"""单元04/05：模型请求工具 -> 注册表校验执行 -> 结果回填 -> 模型最终回答。

闭环要点：
1. assistant 消息必须原样带上 tool_calls，否则下一次调用会失去上下文；
2. tool 消息必须按 tool_call_id 回填，模型才能把结果与请求对上；
3. 参数校验失败时不让程序崩溃，而是把明确的错误信息回填给模型，
   让它有机会补全参数重试 —— 这正是"缺少参数时给出清晰错误"的验收路径。

单元05 起调度改由 ToolRegistry 负责：模型可调用的清单来自 registry.schemas()，
执行边界来自 registry.dispatch()，单次问答同时受 max_rounds（轮次）与
max_calls（调用总数）约束，每次调度都写入 registry.log 供课堂核对。
"""

from __future__ import annotations

import json

from app.prompts import STRONG_SYSTEM_PROMPT
from app.registry import ToolRegistry, default_registry
from app.tools import ToolError

SYSTEM_PROMPT = (
    STRONG_SYSTEM_PROMPT
    + """

工具使用要求：
1. 用户提到订单号时，先调用工具读取模拟订单，不要凭记忆编造订单字段。
2. 询问能否退货/退款时调用 check_return_eligibility，并如实转述工具给出的依据与期限算式。
3. 越权请求（退款、改地址、改物流、删投诉、判定责任）或投诉争议，调用 draft_manual_ticket 生成草稿。
4. 需要解释某个 DOC 编号或引用样例政策原文时，调用 lookup_policy_doc 读取该来源，不要凭记忆复述条款。
5. 工具返回的字段才是可引用的事实；工具报错时先修正参数重试，仍失败则如实说明资料不足。"""
)


def _build_tool_message(call_id: str, name: str, payload: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": json.dumps(payload, ensure_ascii=False),
    }


def answer_with_registry(
    llm,
    messages: list[dict],
    registry: ToolRegistry,
    *,
    max_rounds: int = 4,
    max_calls: int = 6,
    require_tool_call: bool = True,
) -> tuple[str, list[dict]]:
    """单元05 调度主循环：模型请求 -> 注册表调度 -> 结果回填 -> 模型作答。"""
    if not messages:
        raise ValueError("messages不能为空")

    dialog: list[dict] = messages
    trace: list[dict] = []

    for round_index in range(1, max_rounds + 1):
        message = llm.complete(dialog, registry.schemas())
        calls = getattr(message, "tool_calls", None) or []

        if not calls:
            content = (getattr(message, "content", None) or "").strip()
            if not content:
                raise RuntimeError("模型未生成最终回答")
            if not trace and require_tool_call:
                raise RuntimeError("模型没有发起工具调用，无法证明本次演示形成闭环")
            return content, trace

        if len(trace) + len(calls) > max_calls:
            raise ToolError(f"本次问答工具调用超过上限：{max_calls}")

        dialog.append(message.model_dump(exclude_none=True))

        requests: list[dict] = []
        for call in calls:
            if getattr(call, "type", "function") != "function":
                raise ToolError(f"不支持的调用类型：{call.type}")
            try:
                arguments = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                arguments = None
            requests.append({"call_id": call.id, "name": call.function.name, "arguments": arguments})

        for step in registry.dispatch(requests, max_calls=max_calls, round_index=round_index):
            trace.append(step)
            dialog.append(_build_tool_message(step["call_id"], step["name"], step["result"]))

    raise RuntimeError(f"连续{max_rounds}轮工具调用后仍未得到最终回答")


def answer_with_tools(
    llm,
    messages: list[dict],
    *,
    max_rounds: int = 4,
    max_calls: int = 6,
    require_tool_call: bool = True,
    registry: ToolRegistry | None = None,
) -> tuple[str, list[dict]]:
    return answer_with_registry(
        llm,
        messages,
        registry or default_registry(),
        max_rounds=max_rounds,
        max_calls=max_calls,
        require_tool_call=require_tool_call,
    )


def answer_question(
    llm,
    question: str,
    *,
    max_rounds: int = 4,
    registry: ToolRegistry | None = None,
) -> tuple[str, list[dict]]:
    if not question.strip():
        raise ValueError("问题不能为空")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return answer_with_registry(
        llm,
        messages,
        registry or default_registry(),
        max_rounds=max_rounds,
        require_tool_call=True,
    )


def format_trace(trace: list[dict]) -> str:
    """把调用轨迹渲染成课堂可直接截图的美化文本。"""
    if not trace:
        return "  （本次未发生工具调用）"
    lines = []
    for index, step in enumerate(trace, start=1):
        status = step.get("status", "ok")
        status_icon = "✅" if status == "ok" else "❌"
        lines.append(f"  [{status_icon} 第{index}次调用] 工具：{step['name']}（第 {step.get('round', '-')} 轮）")
        lines.append(f"     ├─ 生成参数：{json.dumps(step['arguments'], ensure_ascii=False)}")
        lines.append(f"     ├─ 执行状态：{status}（耗时 {step.get('elapsed_ms', '-')}ms）")

        res_str = json.dumps(step['result'], ensure_ascii=False)
        if len(res_str) > 100:
            res_str = res_str[:100] + "... [已截断]"
        lines.append(f"     └─ 返回结果：{res_str}")

        if step.get("error"):
            lines.append(f"     ⚠️ 校验提示：{step['error']}")
    return "\n".join(lines)