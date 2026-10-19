"""模型统一接口：离线 Mock 与可选的真实 DeepSeek 调用。

两种实现都提供同一组方法，便于课堂在无网环境下也能跑通全部演示：
- stream(messages)：Unit02 的逐段流式输出；
- chat(messages)：一次性取回完整文本；
- complete(messages, tools)：Unit04 的工具调用入口，返回带 tool_calls 的消息对象。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field


ORDER_PATTERN = re.compile(r"SO-\d+")

OVERBOUND_KEYWORDS = [
    "直接退款", "直接给", "改地址", "修改地址", "改物流", "改签收",
    "删除投诉", "判定商家责任", "不要转人工", "立刻赔",
]

RETURN_KEYWORDS = ["退货", "退款", "能否退", "能不能退", "可退吗", "七天无理由"]

DISPUTE_KEYWORDS = ["投诉", "维权", "工商", "律师", "法律", "欺诈", "报警"]

DOC_PATTERN = re.compile(r"DOC-\d{2}-\d{2}", re.IGNORECASE)

POLICY_KEYWORDS = ["政策", "条款", "规定", "依据原文", "怎么写的"]


def extract_doc_id(text: str) -> str | None:
    """从用户问题中抽取样例政策来源编号。"""
    match = DOC_PATTERN.search(text or "")
    return match.group(0).upper() if match else None

REASON_HINTS = [
    "质量问题", "质量", "破损", "错发", "漏发", "延迟", "物流异常",
    "不想要了", "七天无理由", "尺寸不合", "色差",
]


def extract_order_id(text: str) -> str | None:
    """从用户问题中抽取订单号。"""
    match = ORDER_PATTERN.search(text or "")
    return match.group(0) if match else None


# ---------------------------------------------------------------- 工具调用消息结构


@dataclass
class MockToolCallFunction:
    name: str
    arguments: str  # JSON 字符串，与 OpenAI 保持一致


@dataclass
class MockToolCall:
    id: str
    function: MockToolCallFunction
    type: str = "function"


@dataclass
class MockAssistantMessage:
    """与 OpenAI assistant message 兼容的结构，便于共用同一套 tool_loop。"""

    content: str | None = None
    tool_calls: list[MockToolCall] | None = None
    role: str = "assistant"

    def model_dump(self, exclude_none: bool = True) -> dict:
        data: dict = {"role": self.role}
        if self.content is not None or not exclude_none:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return data


# ---------------------------------------------------------------- 离线模型


@dataclass
class MockLLM:
    """课堂离线模型：不联网也能演示流式输出、多轮上下文与工具闭环。"""

    # 演示"模型漏传参数"场景：第一轮故意省略某个参数，由 tool_loop 报错后纠正。
    omit_argument: str | None = None
    _call_seq: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------ Unit02：流式与多轮

    def chat(self, messages: list[dict[str, str]]) -> str:
        return "".join(self.stream(messages))

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not messages:
            raise ValueError("messages不能为空")

        user_messages = [item["content"] for item in messages if item["role"] == "user"]
        prompt = user_messages[-1]
        history_text = "\n".join(user_messages[:-1])

        if any(keyword in prompt for keyword in OVERBOUND_KEYWORDS):
            answer = (
                "【处理建议】该请求超出模型权限：不能直接退款、改地址、改物流状态、"
                "删除投诉或判定商家责任。已生成待人工确认建议，请转人工处理。"
                "来源：DOC-02-13、DOC-02-12、DOC-02-15。"
            )
        elif "退货" in prompt or "退款" in prompt:
            answer = (
                "【订单事实】请先核对订单号。"
                "【政策依据】DOC-02-15 客服答复规范：区分订单事实、样例政策和待人工确认事项。"
                "DOC-02-05 定制商品例外：定制商品不适用普通7天退货。"
                "DOC-02-06 易耗品例外：已拆封易耗品不按普通无理由退货。"
                "【处理建议】已签收订单需结合签收时间和商品类别判断，不能直接承诺可退。"
                "【待人工确认】质量争议、签收争议转人工复核。"
            )
        elif "物流" in prompt or "异常" in prompt or "签收" in prompt:
            answer = (
                "【订单事实】先查询物流状态。"
                "【政策依据】DOC-02-03 已发货未签收处理：需要拦截时生成待人工确认工单，不直接改派。"
                "DOC-02-14 签收争议：物流签收状态不等于顾客实际收到，争议转人工。"
                "【处理建议】按流程查询物流，异常或争议转人工。"
                "【待人工确认】拦截、改派、签收争议需人工确认。"
            )
        elif "投诉" in prompt or "欺诈" in prompt or "法律" in prompt:
            answer = (
                "【政策依据】DOC-02-12 投诉升级：包含人身攻击、欺诈指控或法律争议时，"
                "停止自动承诺并转人工处理。"
                "【处理建议】转人工处理，不自动承诺。"
                "来源：DOC-02-12、DOC-02-15。"
            )
        elif "还需要" in prompt and any(
            keyword in history_text for keyword in ["退货", "物流", "售后", "签收"]
        ):
            answer = (
                "结合上一轮售后问题，还需要提供订单号、签收时间、"
                "商品照片或物流截图、问题描述；涉及签收争议、投诉或法律争议时转人工核实。"
            )
        else:
            answer = f"这是第{len(user_messages)}轮离线回答。你问的是：{prompt}"

        for start in range(0, len(answer), 6):
            yield answer[start : start + 6]

    # ------------------------------------------------ Unit04：工具调用

    def complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
    ) -> MockAssistantMessage:
        """按规则扮演"模型"：要么发起工具调用，要么基于工具返回结果作答。"""
        if not messages:
            raise ValueError("messages不能为空")

        # 只有"本轮用户提问之后"回填的 tool 结果才算数；
        # 否则多轮会话里新一轮提问会误用上一轮的工具结果，答非所问。
        last_user_index = max(
            (index for index, item in enumerate(messages) if item.get("role") == "user"),
            default=-1,
        )
        tool_messages = [
            item for item in messages[last_user_index + 1 :] if item.get("role") == "tool"
        ]
        if tool_messages:
            last_tool = tool_messages[-1]
            try:
                payload = json.loads(last_tool.get("content", "{}"))
            except json.JSONDecodeError:
                payload = {}

            # 工具明确报错时不要急着作答，而是补全参数重试，形成"缺参 -> 报错 -> 纠正"证据链。
            if payload.get("error"):
                self.omit_argument = None
                user_text = next(
                    (item["content"] for item in reversed(messages) if item.get("role") == "user"),
                    "",
                )
                recent_text = "\n".join(
                    item.get("content", "") for item in messages if item.get("role") == "user"
                )
                order_id = extract_order_id(user_text) or extract_order_id(recent_text)
                allowed = {item["function"]["name"] for item in tools or []}
                if order_id and allowed:
                    return MockAssistantMessage(
                        content=None,
                        tool_calls=[self._plan_tool_call(user_text, order_id, allowed)],
                    )

            return MockAssistantMessage(content=self._answer_from_tool_result(last_tool))

        allowed = {item["function"]["name"] for item in tools or []}
        if not allowed:
            return MockAssistantMessage(content=self.chat(messages))

        user_text = next(
            (item["content"] for item in reversed(messages) if item.get("role") == "user"),
            "",
        )
        # 单元05：政策类问题不依赖订单号，优先走第二个业务工具。
        if "lookup_policy_doc" in allowed and (
            extract_doc_id(user_text) or any(keyword in user_text for keyword in POLICY_KEYWORDS)
        ):
            return MockAssistantMessage(
                content=None, tool_calls=[self._plan_tool_call(user_text, None, allowed)]
            )

        # 追问时用户常省略订单号，从历史里回补同一个订单，保证多轮上下文连续。
        recent_text = "\n".join(
            item.get("content", "") for item in messages if item.get("role") == "user"
        )
        order_id = extract_order_id(user_text) or extract_order_id(recent_text)
        if order_id is None:
            return MockAssistantMessage(
                content="【订单事实】未提供订单号。【待人工确认】请提供 SO- 开头的订单号后再查询。"
            )

        call = self._plan_tool_call(user_text, order_id, allowed)
        return MockAssistantMessage(content=None, tool_calls=[call])

    def _plan_tool_call(self, user_text: str, order_id: str | None, allowed: set[str]) -> MockToolCall:
        # 单元05：问到政策原文或点名某个 DOC 编号时，走第二个业务工具。
        doc_id = extract_doc_id(user_text)
        if "lookup_policy_doc" in allowed and (
            doc_id or any(keyword in user_text for keyword in POLICY_KEYWORDS)
        ):
            name = "lookup_policy_doc"
            arguments = {"doc_id": doc_id or "DOC-02-01"}
        elif "draft_manual_ticket" in allowed and any(k in user_text for k in OVERBOUND_KEYWORDS):
            name = "draft_manual_ticket"
            arguments = {
                "order_id": order_id,
                "issue": "越权请求（退款/改地址/改物流/删投诉）：超出模型权限，需人工处理",
            }
        elif "check_return_eligibility" in allowed and any(k in user_text for k in RETURN_KEYWORDS):
            name = "check_return_eligibility"
            arguments = {"order_id": order_id, "reason": self._extract_reason(user_text)}
        elif "draft_manual_ticket" in allowed and any(k in user_text for k in DISPUTE_KEYWORDS):
            name = "draft_manual_ticket"
            arguments = {
                "order_id": order_id,
                "issue": "投诉/纠纷需人工处理：涉及攻击性表述或法律争议",
            }
        elif "query_order" in allowed:
            name = "query_order"
            arguments = {"order_id": order_id}
        elif "check_return_eligibility" in allowed:
            name = "check_return_eligibility"
            arguments = {"order_id": order_id, "reason": self._extract_reason(user_text)}
        else:
            name = "draft_manual_ticket"
            arguments = {"order_id": order_id, "issue": "需人工确认"}

        # 演示缺参：第一轮故意漏掉指定参数，等工具报错后由模型补全重试。
        if self.omit_argument and self.omit_argument in arguments:
            arguments = {key: value for key, value in arguments.items() if key != self.omit_argument}

        self._call_seq += 1
        return MockToolCall(
            id=f"call_{self._call_seq}_{uuid.uuid4().hex[:8]}",
            function=MockToolCallFunction(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
        )

    @staticmethod
    def _extract_reason(user_text: str) -> str:
        for hint in REASON_HINTS:
            if hint in user_text:
                return hint
        return "顾客未说明具体原因"

    def _answer_from_tool_result(self, tool_message: dict) -> str:
        """把工具返回的 JSON 整理成四段式答复，便于课堂展示闭环结果。"""
        try:
            payload = json.loads(tool_message.get("content", "{}"))
        except json.JSONDecodeError:
            return (
                "【处理建议】工具返回无法解析。"
                "【待人工确认】已把该会话转入人工流程，请勿据此答复顾客。"
            )

        tool_name = tool_message.get("name", "")
        if payload.get("error"):
            return (
                "【订单事实】工具调用未通过校验。"
                "【政策依据】DOC-02-15。"
                f"【处理建议】参数错误：{payload['error']}；已按规范重新生成工具调用。"
                "【待人工确认】若仍无法取数请人工核实。"
            )

        if tool_name == "query_order" or "order" in payload:
            order = payload.get("order", {})
            return (
                f"【订单事实】{order.get('order_id')}：{order.get('item_name')}"
                f"（{order.get('item_category')}），订单状态 {order.get('order_status')}，"
                f"物流状态 {order.get('logistics_status')}，"
                f"签收日期 {order.get('delivered_on') or '未签收'}，"
                f"售后状态 {order.get('after_sale_status')}。数据来源 {payload.get('source')}。"
                "【政策依据】DOC-02-01、DOC-02-03。"
                "【处理建议】如需退货请说明原因，由 check_return_eligibility 按样例政策判断。"
                "【待人工确认】物流异常或签收争议转人工。"
            )

        if tool_name == "check_return_eligibility" or "eligible" in payload:
            verdict = {True: "符合样例退货条件", False: "不符合样例退货条件", None: "样例政策无法直接判定"}
            docs = "、".join(payload.get("policy_docs", [])) or "未命中"
            return (
                f"【订单事实】{payload.get('order_id')}：类别 {payload.get('item_category')}，"
                f"签收日期 {payload.get('delivered_on') or '未签收'}，"
                f"是否拆封 {payload.get('used_or_opened')}。{payload.get('calculation', '')}"
                f"【政策依据】{docs}：{payload.get('reason', '')}"
                f"【处理建议】{verdict.get(payload.get('eligible'), '样例政策无法直接判定')}；"
                "不承诺真实售后结果，按 DOC-02-15 规范答复。"
                f"【待人工确认】{'需转人工复核（' + docs + '）' if payload.get('manual') else '无'}"
            )

        if tool_name == "lookup_policy_doc" or "doc_id" in payload:
            if not payload.get("ok"):
                return (
                    f"【订单事实】政策来源 {payload.get('doc_id', '未知')} 未取到。"
                    f"【政策依据】未命中。{payload.get('message', '')}"
                    "【处理建议】说明资料不足，不要凭记忆复述条款。"
                    "【待人工确认】请人工核实该来源编号。"
                )
            return (
                f"【订单事实】样例政策来源 {payload.get('doc_id')}《{payload.get('title')}》。"
                f"【政策依据】{payload.get('doc_id')} 原文摘录：{payload.get('excerpt')}"
                f"（来源 {payload.get('source')}；{payload.get('note', '')}）"
                "【处理建议】引用该来源编号答复，不扩大解释到未命中的场景。"
                "【待人工确认】是否适用于具体订单需结合订单事实人工确认。"
            )

        draft = payload.get("ticket_draft", {})
        return (
            f"【订单事实】{draft.get('order_id')}（订单存在：{draft.get('order_found')}）。"
            f"【政策依据】{'、'.join(payload.get('policy_docs', []))}。"
            f"【处理建议】已生成待人工确认工单草稿，问题：{draft.get('issue')}；"
            f"{payload.get('note', '')}不直接退款、不改地址、不判定商家责任。"
            "【待人工确认】由人工客服复核后决定是否受理。"
        )


def build_llm():
    """根据环境变量选择模型；默认 Mock，避免课堂必须联网。"""
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "deepseek":
        from .deepseek import DeepSeekLLM

        return DeepSeekLLM()
    if provider != "mock":
        raise ValueError("LLM_PROVIDER只能是mock或deepseek")
    return MockLLM()


class FallbackLLM:
    """真实模型不可用时自动降级到离线模型的包装器。

    课堂演示最怕的不是模型答错，而是 API 欠费（402）、密钥失效（401）或网络中断时
    整个脚本崩掉、什么都留不下。这个包装器把第一次失败的原因记进 `degradations`，
    之后的调用全部走离线模型，保证证据文件一定生成、且如实标注哪些是模拟结果。
    """

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback or MockLLM()
        self.degradations: list[str] = []
        self._degraded = False

    @property
    def degraded(self) -> bool:
        return self._degraded

    def _use(self, method: str, *args, **kwargs):
        if self._degraded:
            return getattr(self.fallback, method)(*args, **kwargs)

        try:
            return getattr(self.primary, method)(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 任何故障都不该中断证据生成
            self._degraded = True
            reason = f"{type(exc).__name__}: {exc}".strip()
            # 流式返回的是生成器，异常发生在迭代时，这里先探一次把故障暴露出来。
            self.degradations.append(f"{method}() 调用 {type(self.primary).__name__} 失败 -> {reason}")
            return getattr(self.fallback, method)(*args, **kwargs)

    def chat(self, messages):
        return self._use("chat", messages)

    def complete(self, messages, tools=None):
        return self._use("complete", messages, tools)

    def stream(self, messages):
        """流式需要单独处理：生成器的异常在迭代时才抛，不预取就会漏掉降级。"""
        if self._degraded:
            yield from self.fallback.stream(messages)
            return

        try:
            source = iter(self.primary.stream(messages))
            first = next(source)  # 预取首段，真正的网络错误在这里就会暴露
        except StopIteration:
            return
        except Exception as exc:  # noqa: BLE001
            self._degraded = True
            reason = f"{type(exc).__name__}: {exc}".strip()
            self.degradations.append(f"stream() 调用 {type(self.primary).__name__} 失败 -> {reason}")
            yield from self.fallback.stream(messages)
            return

        yield first
        yield from source

        # 已产出首段后才失败：不再切换模型（否则内容会前后不一致），如实记一笔。
        try:
            for chunk in source:
                yield chunk
        except Exception as exc:  # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}".strip()
            self.degradations.append(f"stream() 输出中途中断 -> {reason}")
