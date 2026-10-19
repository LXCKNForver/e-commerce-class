"""单元05：工具注册表（Registry）与调度器（Dispatcher）。

之前工具说明写在 TOOL_SCHEMAS、执行白名单写在 TOOL_HANDLERS，两处需要手工同步；
一旦改了其中一处，就会出现"模型能选但程序不认"或"程序能跑但模型不知道"。

本模块把工具的名称、用途、参数说明和处理函数合并成一项 ToolSpec 登记：
- schemas() 由登记项生成，模型看到的清单 = 程序可执行的清单；
- call() 做白名单与参数校验，未注册的名称绝不映射到任何函数；
- dispatch() 是调度器：批量处理一批请求，单条失败不影响其他请求，
  并把 call_id 原样带回，供 tool_loop 按 ID 回填结果。

新增工具只需在 build_registry() 里加一行 register()，其余代码不用改。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from app.tools import (
    ToolError,
    check_return_eligibility,
    draft_manual_ticket,
    lookup_policy_doc,
    query_order,
)


# ---------------------------------------------------------------- 登记项


@dataclass(frozen=True)
class ToolParam:
    """一个参数的登记信息；类型目前只用到 string，留作扩展点。"""

    name: str
    description: str
    type: str = "string"


@dataclass(frozen=True)
class ToolSpec:
    """一项工具的登记信息。

    description 告诉模型"什么情况下该用它"；
    usage 是写给课堂与组长的登记用途，不发送给模型；
    handler 是程序实际调用的 Python 函数，也不会发送给模型。
    """

    name: str
    description: str
    params: tuple[ToolParam, ...]
    handler: Callable[..., dict]
    usage: str = ""
    read_only: bool = True

    @property
    def param_names(self) -> list[str]:
        return [param.name for param in self.params]

    def schema(self) -> dict[str, Any]:
        """把登记项转换成模型接口要求的 function schema。

        additionalProperties=False 禁止额外参数；required 声明必填项。
        这只是"如何请求工具"，真正执行时 call() 还会再校验一次。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {"type": param.type, "description": param.description}
                        for param in self.params
                    },
                    "required": self.param_names,
                    "additionalProperties": False,
                },
            },
        }


# ---------------------------------------------------------------- 调用日志


@dataclass
class DispatchLog:
    """按轮次记录每次调度，供课堂核对输入、输出与异常提示。"""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, **entry: Any) -> None:
        self.entries.append(entry)

    def clear(self) -> None:
        self.entries.clear()

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def failed(self) -> int:
        return sum(1 for item in self.entries if item.get("status") != "ok")

    def summary(self) -> dict[str, int]:
        return {"total": self.total, "ok": self.total - self.failed, "error": self.failed}

    def render(self) -> str:
        if not self.entries:
            return "（本次会话还没有调度记录）"
        lines = ["轮次  工具                        状态    耗时ms  说明"]
        for item in self.entries:
            lines.append(
                f"{item['round']:<5}{item['name']:<28}{item['status']:<8}"
                f"{item['elapsed_ms']:<8}{item['detail']}"
            )
        stat = self.summary()
        lines.append(f"合计 {stat['total']} 次调用：成功 {stat['ok']}，被拒绝或异常 {stat['error']}。")
        return "\n".join(lines)


# ---------------------------------------------------------------- 注册表


class ToolRegistry:
    """保存允许执行的工具，并统一负责参数检查和分发。"""

    def __init__(self) -> None:
        # 用工具名称作键，收到模型请求后可以直接查到对应的 ToolSpec。
        self._items: dict[str, ToolSpec] = {}
        self.log = DispatchLog()

    # ---------------------------------------------------------- 登记

    def register(self, spec: ToolSpec) -> None:
        """登记一项工具；名称重复或缺少关键配置时立即报错。"""
        if not spec.name or not spec.description:
            raise ValueError("工具名称与用途说明不能为空")
        if not spec.params:
            raise ValueError(f"{spec.name} 至少需要一个参数")
        if not callable(spec.handler):
            raise ValueError(f"{spec.name} 的处理函数不可调用")
        if spec.name in self._items:
            raise ValueError(f"工具重复注册：{spec.name}")
        self._items[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._items)

    def schemas(self) -> list[dict[str, Any]]:
        """模型看到的清单直接来自登记项，与可执行范围天然一致。"""
        return [spec.schema() for spec in self._items.values()]

    def get(self, name: str) -> ToolSpec | None:
        return self._items.get(name)

    def describe(self) -> str:
        """给课堂看的登记表：名称、用途、参数、是否只读。"""
        lines = []
        for index, spec in enumerate(self._items.values(), start=1):
            params = "、".join(spec.param_names)
            lines.append(f"{index}. {spec.name}（{'只读' if spec.read_only else '写入'}）")
            lines.append(f"   用途：{spec.usage or spec.description}")
            lines.append(f"   参数：{params}")
        return "\n".join(lines)

    # ---------------------------------------------------------- 单条执行

    def call(self, name: str, arguments: Any) -> dict:
        """执行单个工具请求：先确认工具存在，再核对参数，最后调用 handler。"""
        spec = self._items.get(name)
        if spec is None:
            # 模型给出的名称只是请求；未注册的名称绝不能映射到任意函数。
            raise ToolError(f"未知工具：{name}；可用工具：{', '.join(sorted(self._items))}")
        if not isinstance(arguments, dict):
            raise ToolError(f"{name} 的 arguments 必须是 JSON 对象")

        allowed = set(spec.param_names)
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ToolError(f"{name} 不支持参数：{', '.join(unknown)}")
        missing = sorted(allowed - set(arguments))
        if missing:
            raise ToolError(f"{name} 缺少必填参数：{', '.join(missing)}")
        for key in spec.param_names:
            value = arguments[key]
            if not isinstance(value, str) or not value.strip():
                # 空字符串和纯空格无法用于查询；同时避免把数字等类型交给处理函数。
                raise ToolError(f"{name} 的参数 {key} 必须是非空字符串")

        clean = {key: arguments[key].strip() for key in spec.param_names}
        return spec.handler(**clean)

    # ---------------------------------------------------------- 调度器

    def dispatch(
        self,
        calls: list[dict[str, Any]],
        *,
        max_calls: int = 4,
        round_index: int = 0,
    ) -> list[dict[str, Any]]:
        """逐项处理模型的一批请求，为每项返回可回填的结果。

        任何一条失败都只变成该项的可见错误，不影响同批其他请求，
        也不让主程序中断——这是单元05"工具异常不会让主程序直接崩溃"的关键。
        """
        if max_calls <= 0:
            raise ValueError("max_calls必须大于0")
        if len(calls) > max_calls:
            raise ToolError(f"单轮工具调用超过上限：{max_calls}")

        results: list[dict[str, Any]] = []
        for call in calls:
            name = call.get("name")
            arguments = call.get("arguments")
            started = perf_counter()
            try:
                result = self.call(name, arguments)
                status, detail = "ok", "执行成功"
                error: str | None = None
            except ToolError as exc:
                # 校验类失败：未知工具、缺参、多参、类型不对。
                result, status, detail, error = {"error": str(exc)}, "拒绝", str(exc), str(exc)
            except Exception as exc:  # noqa: BLE001 - 兜底：处理函数抛出的任何异常都不向上冒泡
                message = f"{type(exc).__name__}: {exc}"
                result, status, detail, error = {"error": f"工具执行异常：{message}"}, "异常", message, message

            elapsed_ms = round((perf_counter() - started) * 1000, 2)
            self.log.add(
                round=round_index,
                call_id=call.get("call_id"),
                name=name,
                arguments=arguments if isinstance(arguments, dict) else None,
                status=status,
                detail=detail,
                elapsed_ms=elapsed_ms,
            )
            # call_id 原样带回，供 tool_loop 与模型发起的请求逐条配对。
            results.append(
                {
                    "call_id": call.get("call_id"),
                    "round": round_index,
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                    "status": status,
                    "error": error,
                    "elapsed_ms": elapsed_ms,
                }
            )
        return results


# ---------------------------------------------------------------- 组装


def build_registry() -> ToolRegistry:
    """组装本项目的全部受控工具，作为统一入口。

    新增工具只在这里加一行 register()，模型说明与执行白名单同时生效。
    """
    registry = ToolRegistry()

    registry.register(ToolSpec(
        name="query_order",
        description=(
            "按订单号查询模拟订单的订单状态、物流状态、签收日期、是否拆封和售后状态。"
            "用户询问订单状态、物流进度或要求核对订单事实时使用；只读，不修改任何数据。"
        ),
        params=(ToolParam("order_id", "完整订单编号，例如 SO-2001。"),),
        handler=query_order,
        usage="只读核对订单事实，是所有售后判断的取数入口",
    ))

    registry.register(ToolSpec(
        name="check_return_eligibility",
        description=(
            "按样例售后政策判断订单是否可退。同时考虑签收时间（7 天期限）与商品类别，"
            "定制商品、已拆封易耗品、质量争议与投诉一律转人工。"
            "用户询问'能不能退货/退款''符合不符合条件'时使用。"
        ),
        params=(
            ToolParam("order_id", "完整订单编号，例如 SO-2001。"),
            ToolParam("reason", "顾客提出的售后原因，例如'质量问题''不想要了''物流延迟'。"),
        ),
        handler=check_return_eligibility,
        usage="按样例政策给出退货结论与依据，争议场景转人工",
    ))

    registry.register(ToolSpec(
        name="draft_manual_ticket",
        description=(
            "生成待人工确认的工单草稿。用于越权请求、纠纷投诉、资料不全等需要人工介入的场景；"
            "只返回草稿，不写入真实系统。"
        ),
        params=(
            ToolParam("order_id", "完整订单编号，例如 SO-2001。"),
            ToolParam("issue", "需要人工处理的问题描述。"),
        ),
        handler=draft_manual_ticket,
        usage="越权与纠纷的出路：只出草稿，不退款不改地址",
    ))

    # 单元05 新增的第二个业务工具：让【政策依据】引用到真实的样例政策原文，
    # 而不是把 DOC 编号写死在代码里。
    registry.register(ToolSpec(
        name="lookup_policy_doc",
        description=(
            "按来源编号（例如 DOC-02-05）读取课程样例政策原文的规则与说明、使用边界两节。"
            "需要引用政策依据或解释某个 DOC 编号时使用；只摘录原文，不判断是否适用于当前订单。"
        ),
        params=(ToolParam("doc_id", "来源编号，格式 DOC-02-05。"),),
        handler=lookup_policy_doc,
        usage="只读取样例政策原文，支撑【政策依据】引用",
    ))

    return registry


_DEFAULT_REGISTRY: ToolRegistry | None = None


def default_registry() -> ToolRegistry:
    """进程内共享一份注册表，避免每次调用都重新登记。"""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_registry()
    return _DEFAULT_REGISTRY
