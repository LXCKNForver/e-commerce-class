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
    "质量问题", "质量", "破损", "坏了", "错发", "漏发", "延迟", "物流异常",
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
                "【处理建议】非常抱歉，该请求超出了我的服务权限。作为智能客服，我无法直接为您办理退款、修改地址或物流状态，也无权判定商家责任或删除投诉记录。\n\n"
                "【待人工确认】已为您记录该诉求，并转接至人工客服为您跟进处理，请您稍候。\n"
                "（来源：DOC-02-13、DOC-02-12、DOC-02-15）"
            )
        elif "退货" in prompt or "退款" in prompt:
            answer = (
                "【订单事实】请先核对您的订单号。\n\n"
                "【政策依据】根据 DOC-02-15 客服答复规范，我们需要区分订单事实、样例政策和待人工确认事项。\n"
                "此外，请留意：定制商品不适用普通7天退货（DOC-02-05），已拆封易耗品不按普通无理由退货（DOC-02-06）。\n\n"
                "【处理建议】已签收订单需结合签收时间和商品类别进行综合判断，在此暂无法直接向您承诺是否可以退货。\n\n"
                "【待人工确认】如涉及质量争议或签收争议，我们将为您转交人工复核。"
            )
        elif "物流" in prompt or "异常" in prompt or "签收" in prompt:
            answer = (
                "【订单事实】我先为您查询物流状态。\n\n"
                "【政策依据】根据 DOC-02-03 已发货未签收处理规则，若需要拦截，我们会生成待人工确认的工单，不会直接改派。\n"
                "根据 DOC-02-14 签收争议规则，物流显示“已签收”不等于您实际收到，若有争议需转人工。\n\n"
                "【处理建议】若遇物流异常，建议您先提供物流单号或异常截图，按流程我们会为您转交人工核实。\n\n"
                "【待人工确认】拦截、改派或签收争议等事宜，均需人工客服进一步确认。"
            )
        elif "投诉" in prompt or "欺诈" in prompt or "法律" in prompt:
            answer = (
                "【政策依据】根据 DOC-02-12 投诉升级规则，当投诉包含人身攻击、欺诈指控或法律争议时，我们将停止自动承诺并转人工处理。\n\n"
                "【处理建议】非常抱歉给您带来不佳的体验，该情况我已为您记录并转接人工客服专员为您处理。\n\n"
                "（来源：DOC-02-12、DOC-02-15）"
            )
        elif "还需要" in prompt and any(
            keyword in history_text for keyword in ["退货", "物流", "售后", "签收"]
        ):
            answer = (
                "结合上一轮的售后问题，为了更快为您处理，您可能还需要提供：\n"
                "· 完整的订单编号\n"
                "· 签收时间\n"
                "· 商品照片或物流异常截图\n"
                "· 具体的问题描述\n\n"
                "若涉及签收争议、投诉或法律争议，系统会为您转接人工核实。"
            )
        else:
            answer = f"您好，这是第 {len(user_messages)} 轮对话。您咨询的是：{prompt}。请问有什么可以帮您？"

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

        if "lookup_policy_doc" in allowed and (
            extract_doc_id(user_text) or any(keyword in user_text for keyword in POLICY_KEYWORDS)
        ):
            return MockAssistantMessage(
                content=None, tool_calls=[self._plan_tool_call(user_text, None, allowed)]
            )

        recent_text = "\n".join(
            item.get("content", "") for item in messages if item.get("role") == "user"
        )

        # 先看本轮用户有没有提供单号
        order_id = extract_order_id(user_text)

        # 如果本轮没提供，且问了全局问题，直接拦截
        if not order_id:
            if any(k in user_text for k in ["有哪些", "所有", "全部", "我的订单"]):
                return MockAssistantMessage(
                    content="【订单事实】非常抱歉，当前系统未关联您的专属账号，无法直接展示名下的所有订单记录。\n\n【处理建议】请您提供具体的 SO- 开头的订单编号（例如 SO-2001），我立刻为您单独查询。"
                )

            # 【修复】：既然不是全局查询，客户在对话中省略单号是非常常见的，直接继承上一轮单号
            order_id = extract_order_id(recent_text)

        # 如果找了一圈还是没有
        if order_id is None:
            return MockAssistantMessage(
                content="【订单事实】您好，系统暂未识别到有效订单号。\n\n【待人工确认】请提供 SO- 开头的订单编号后再试，以便我为您查询。"
            )

        call = self._plan_tool_call(user_text, order_id, allowed)
        return MockAssistantMessage(content=None, tool_calls=[call])

    def _plan_tool_call(self, user_text: str, order_id: str | None, allowed: set[str]) -> MockToolCall:
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
        # 增加“坏了”、“破损”等触发退货资格判断
        elif "check_return_eligibility" in allowed and (
            any(k in user_text for k in RETURN_KEYWORDS)
            or any(k in user_text for k in ["坏了", "破损", "不能用", "有问题"])
        ):
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
        """把工具返回的 JSON 整理成四段式答复，语气委婉专业，并在段落间加空行。"""
        try:
            payload = json.loads(tool_message.get("content", "{}"))
        except json.JSONDecodeError:
            return (
                "【处理建议】非常抱歉，系统在处理您的问题时出现了一些异常。\n\n"
                "【待人工确认】已为您转接人工客服，请您稍候。"
            )

        tool_name = tool_message.get("name", "")

        # 处理订单不存在或政策不存在的情况
        if payload.get("ok") is False:
            message = payload.get("message", "未找到相关记录")
            return (
                f"【订单事实】非常抱歉，系统提示：{message}\n\n"
                "【处理建议】请您核对一下提供的信息是否正确。\n\n"
                "【待人工确认】如确认无误但仍有疑问，建议由人工客服为您进一步核实。"
            )

        if payload.get("error"):
            return (
                "【订单事实】抱歉，由于信息校验未通过，暂时无法直接处理您的诉求。\n\n"
                "【政策依据】DOC-02-15 客服答复规范。\n\n"
                f"【处理建议】系统提示：{payload['error']}。我们会为您重新核实并跟进。\n\n"
                "【待人工确认】如问题仍未解决，建议由人工客服进一步为您服务。"
            )

        if tool_name == "query_order" or "order" in payload:
            order = payload.get("order", {})

            # 【修复】：根据不同物流状态，提供体贴入微的话术
            if order.get("order_status") == "运输中" or order.get("logistics_status") == "派送中":
                advice = "您的包裹目前正在派送中，物流小哥正在马不停蹄地为您配送，请您再耐心等待一下，保持电话畅通哦。"
            elif order.get("order_status") == "待发货":
                advice = "您的订单正在仓库加急打包中，我们会尽快为您安排发货，请耐心等待。"
            else:
                advice = "请问您遇到的是物流进度问题，还是希望申请退换货呢？如果需要退货，请告诉我具体原因，我会为您进一步评估。"

            return (
                f"【订单事实】已为您核实订单 {order.get('order_id')} 的当前信息：\n"
                f"· 商品信息：{order.get('item_name')}（{order.get('item_category')}）\n"
                f"· 订单状态：{order.get('order_status')}\n"
                f"· 物流状态：{order.get('logistics_status')}\n"
                f"· 签收日期：{order.get('delivered_on') or '暂未签收'}\n"
                f"· 售后状态：{order.get('after_sale_status')}\n\n"
                "【政策依据】关于商品的售后，可参考样例政策 DOC-02-01、DOC-02-03 等条款。\n\n"
                f"【处理建议】{advice}\n\n"
                "【待人工确认】如涉及物流异常、地址修改或签收争议，系统将为您转接人工处理。"
            )

        if tool_name == "check_return_eligibility" or "eligible" in payload:
            verdict = {True: "符合样例退货条件", False: "暂不符合样例退货条件", None: "样例政策无法直接判定"}
            docs = "、".join(payload.get("policy_docs", [])) or "未命中相关条款"
            calc = payload.get('calculation', '暂未签收，无签收天数').replace('（', '\n  （')

            reason_text = payload.get('reason', '')
            if "质量" in reason_text or "争议" in reason_text:
                advice = (
                    "非常抱歉听到商品出现了损坏或质量问题，给您带来了不便。\n"
                    "  由于涉及质量问题，按平台流程需要人工介入为您核实，已为您申请优先处理通道。"
                )
            elif payload.get('eligible') is True:
                advice = "根据现有政策，该订单初步符合退货条件。您可以在订单页面提交退货申请，最终结果以实际审核为准。"
            elif payload.get('eligible') is False:
                advice = "根据样例政策，该订单由于商品类别或拆封情况暂不支持无理由退货。如果您认为有其他特殊情况，建议联系人工客服协助。"
            else:
                advice = "该情况较为特殊，为了保障您的权益，已为您转接人工客服进一步核实。"

            return (
                f"【订单事实】已为您核实订单 {payload.get('order_id')}：\n"
                f"· 商品类别：{payload.get('item_category')}\n"
                f"· 签收日期：{payload.get('delivered_on') or '暂未签收'}\n"
                f"· 是否拆封：{'是' if payload.get('used_or_opened') else '否'}\n"
                f"· 时间计算：{calc}\n\n"
                f"【政策依据】命中条款 {docs}：\n"
                f"  {reason_text}\n\n"
                f"【处理建议】{advice}\n\n"
                f"【待人工确认】{'由于涉及特殊争议或例外条款，已为您生成优先处理通道。' if payload.get('manual') else '无。感谢您的理解！'}"
            )

        if tool_name == "lookup_policy_doc" or "doc_id" in payload:
            return (
                f"【订单事实】已为您调取样例政策来源 {payload.get('doc_id')}《{payload.get('title')}》。\n\n"
                f"【政策依据】{payload.get('doc_id')} 原文摘录：\n"
                f"  {payload.get('excerpt')}\n"
                f"  （来源 {payload.get('source')}；{payload.get('note', '')}）\n\n"
                "【处理建议】以上政策供您参考，具体是否适用于您的订单，需结合订单事实进一步评估。\n\n"
                "【待人工确认】如需确认适用性，可为您转接人工客服核实。"
            )

        draft = payload.get("ticket_draft", {})
        return (
            f"【订单事实】已核实订单 {draft.get('order_id')}（订单存在：{'是' if draft.get('order_found') else '否'}）。\n\n"
            f"【政策依据】依据 {'、'.join(payload.get('policy_docs', []))}，该诉求涉及特殊权限或争议场景。\n\n"
            f"【处理建议】非常抱歉，我作为智能助手无法直接为您执行“退款、改地址或判定责任”等操作。\n"
            f"  已为您生成一份待人工确认的工单草稿，问题描述：{draft.get('issue')}。\n"
            f"  {payload.get('note', '')}\n\n"
            "【待人工确认】请您耐心等待，人工客服将尽快介入为您处理。"
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
    """真实模型不可用时自动降级到离线模型的包装器。"""

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
        except Exception as exc:
            self._degraded = True
            reason = f"{type(exc).__name__}: {exc}".strip()
            self.degradations.append(f"{method}() 调用 {type(self.primary).__name__} 失败 -> {reason}")
            return getattr(self.fallback, method)(*args, **kwargs)

    def chat(self, messages):
        return self._use("chat", messages)

    def complete(self, messages, tools=None):
        return self._use("complete", messages, tools)

    def stream(self, messages):
        if self._degraded:
            yield from self.fallback.stream(messages)
            return

        try:
            source = iter(self.primary.stream(messages))
            first = next(source)
        except StopIteration:
            return
        except Exception as exc:
            self._degraded = True
            reason = f"{type(exc).__name__}: {exc}".strip()
            self.degradations.append(f"stream() 调用 {type(self.primary).__name__} 失败 -> {reason}")
            yield from self.fallback.stream(messages)
            return

        yield first
        yield from source

        try:
            for chunk in source:
                yield chunk
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}".strip()
            self.degradations.append(f"stream() 输出中途中断 -> {reason}")