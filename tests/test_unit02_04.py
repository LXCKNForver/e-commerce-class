"""单元测试：单元02 多轮与流式、单元03 提示词模板、单元04 工具闭环。"""

import json
import unittest

from app.conversation import Conversation
from app.data_store import OrderRepository
from app.demo_questions import NORMAL_QUESTIONS, OVERBOUND_QUESTIONS
from app.llm import MockLLM
from app.prompts import (
    STRONG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    WEAK_SYSTEM_PROMPT,
    build_messages,
    format_order_context,
    render_template,
)
from app.tools import TOOL_SCHEMAS, ToolError, draft_manual_ticket, execute_tool_call, query_order
from app.tool_loop import answer_with_tools

REPO = OrderRepository()


# ---------------------------------------------------------------- 单元02


class TestUnit02Conversation(unittest.TestCase):
    def test_conversation_records_and_clears_history(self):
        chat = Conversation()
        chat.add("user", "第一个问题")
        chat.add("assistant", "第一个回答")
        self.assertEqual(len(chat.history), 2)
        self.assertEqual(chat.turns, 1)
        chat.clear()
        self.assertEqual(chat.history, [])

    def test_stream_yields_multiple_chunks(self):
        messages = [{"role": "user", "content": "订单SO-2001已签收，我能退货吗？"}]
        chunks = list(MockLLM().stream(messages))
        self.assertGreater(len(chunks), 1)
        self.assertIn("退货", "".join(chunks))

    def test_stream_mode_injects_order_facts(self):
        """流式模式没有工具可用，必须靠模板注入订单事实，否则模型只会说资料不足。"""
        from main import AgentSession

        session = AgentSession(MockLLM())
        question = "订单SO-2001已签收，我能退货吗？"
        session.ask_stream(question)

        self.assertEqual(session.chat.history[0]["content"], question)
        self.assertEqual(session.last_order_id, "SO-2001")
        self.assertIn("SO-2001", session.resolve_context("那物流有异常怎么办？"))

    def test_follow_up_uses_history(self):
        messages = [
            {"role": "user", "content": "订单SO-2001已签收，我能退货吗？"},
            {"role": "assistant", "content": "已收到退货咨询。"},
            {"role": "user", "content": "还需要什么？"},
        ]
        self.assertIn("上一轮", MockLLM().chat(messages))

    def test_history_is_trimmed_in_pairs(self):
        chat = Conversation(max_messages=4)
        for index in range(4):
            chat.add_pair(f"问题{index}", f"回答{index}")
        self.assertEqual(len(chat.history), 4)
        self.assertEqual(chat.history[0]["content"], "问题2")

    def test_empty_message_is_rejected(self):
        with self.assertRaises(ValueError):
            Conversation().add("user", "   ")

    def test_render_when_history_empty(self):
        self.assertIn("没有历史", Conversation().render())


# ---------------------------------------------------------------- 单元03


class TestUnit03Prompts(unittest.TestCase):
    def test_weak_and_strong_prompts_are_different(self):
        self.assertNotEqual(WEAK_SYSTEM_PROMPT, STRONG_SYSTEM_PROMPT)
        self.assertIn("DOC-02-15", STRONG_SYSTEM_PROMPT)
        self.assertLess(len(WEAK_SYSTEM_PROMPT), 50)

    def test_missing_variable_raises_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            render_template("问题：{question}", context="x")
        self.assertIn("question", str(ctx.exception))

    def test_build_messages_uses_same_context_for_both_qualities(self):
        context = "- order_id: SO-2001"
        weak = build_messages("能退货吗？", context, quality="weak")
        strong = build_messages("能退货吗？", context, quality="strong")
        self.assertIn(context, weak[1]["content"])
        self.assertIn(context, strong[1]["content"])
        self.assertEqual(weak[0]["content"], WEAK_SYSTEM_PROMPT)
        self.assertEqual(strong[0]["content"], STRONG_SYSTEM_PROMPT)

    def test_invalid_quality_is_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("x", "y", quality="unknown")

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            build_messages("  ", "y")

    def test_missing_order_renders_explicit_notice(self):
        self.assertIn("未查到", format_order_context(None))

    def test_order_context_contains_all_key_fields(self):
        context = format_order_context(REPO.get_order("SO-2001"))
        for field in ["SO-2001", "2026-10-12", "键盘"]:
            self.assertIn(field, context)


# ---------------------------------------------------------------- 单元04


class TestUnit04ToolSchemaAndValidation(unittest.TestCase):
    def test_schema_names_match_task_card(self):
        # 单元04 指定三个工具必须都在；单元05 之后注册表还会登记政策查询工具。
        names = {item["function"]["name"] for item in TOOL_SCHEMAS}
        self.assertTrue({"query_order", "check_return_eligibility", "draft_manual_ticket"} <= names)

    def test_missing_argument_is_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            execute_tool_call({"name": "query_order", "arguments": {}})
        self.assertIn("order_id", str(ctx.exception))

    def test_missing_reason_argument_is_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            execute_tool_call(
                {"name": "check_return_eligibility", "arguments": {"order_id": "SO-2001"}}
            )
        self.assertIn("reason", str(ctx.exception))

    def test_unknown_argument_is_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            execute_tool_call(
                {
                    "name": "check_return_eligibility",
                    "arguments": {"order_id": "SO-2001", "reason": "质量问题", "amount": 999},
                }
            )
        self.assertIn("amount", str(ctx.exception))

    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            execute_tool_call({"name": "refund_order", "arguments": {"order_id": "SO-2001"}})
        self.assertIn("未知工具", str(ctx.exception))

    def test_blank_argument_is_rejected(self):
        with self.assertRaises(ToolError):
            execute_tool_call({"name": "query_order", "arguments": {"order_id": "  "}})

    def test_valid_call_returns_order(self):
        result = execute_tool_call({"name": "query_order", "arguments": {"order_id": "SO-2001"}})
        self.assertTrue(result["ok"])
        self.assertEqual(result["order"]["order_status"], "已签收")

    def test_ticket_draft_never_writes_business_system(self):
        draft = draft_manual_ticket("SO-2002", "要求拦截")
        self.assertEqual(draft["ticket_draft"]["status"], "待人工确认")
        self.assertIn("未写入", draft["note"])


class TestUnit04ToolLoop(unittest.TestCase):
    def test_successful_round_trip(self):
        llm = MockLLM()
        answer, trace = answer_with_tools(
            llm, [{"role": "user", "content": "订单SO-2001已签收，我能退货吗？"}]
        )
        self.assertTrue(trace)
        self.assertIn("【订单事实】", answer)
        self.assertIn("DOC-02", answer)
        self.assertTrue(any(step["name"] == "check_return_eligibility" for step in trace))

    def test_tool_result_is_filled_back_with_matching_call_id(self):
        llm = MockLLM()
        dialog = [{"role": "user", "content": "请查询订单SO-2001的状态"}]
        answer_with_tools(llm, dialog)
        call_ids = [
            item["tool_call_id"] for item in dialog if item["role"] == "tool"
        ]
        expected = [
            call["id"] for item in dialog if item["role"] == "assistant"
            for call in item.get("tool_calls", [])
        ]
        self.assertTrue(call_ids)
        self.assertEqual(call_ids, expected)

    def test_missing_argument_is_fixed_in_next_round(self):
        llm = MockLLM(omit_argument="reason")
        _, trace = answer_with_tools(
            llm, [{"role": "user", "content": "订单SO-2001已签收，我想退货，质量问题。"}]
        )
        self.assertTrue(any(step.get("error") for step in trace), "应出现一次参数校验失败")
        self.assertTrue(trace[-1]["result"].get("ok"), "最后一轮应取到工具结果")

    def test_no_order_id_does_not_fake_a_tool_result(self):
        llm = MockLLM()
        messages = [{"role": "user", "content": "你好"}]
        # 验收模式：没有工具调用就不能算闭环，必须显式暴露。
        with self.assertRaises(RuntimeError):
            answer_with_tools(llm, list(messages))
        # 交互模式：允许模型直接作答（例如请用户补充订单号）。
        answer, trace = answer_with_tools(llm, list(messages), require_tool_call=False)
        self.assertEqual(trace, [])
        self.assertIn("订单号", answer)

    def test_overbound_question_produces_ticket_draft_only(self):
        llm = MockLLM()
        answer, trace = answer_with_tools(llm, [{"role": "user", "content": OVERBOUND_QUESTIONS[0]}])
        self.assertTrue(any(step["name"] == "draft_manual_ticket" for step in trace))
        self.assertIn("草稿", answer)
        self.assertNotIn("已退款", answer)

    def test_tool_payload_is_serialisable(self):
        result = query_order("SO-2001")
        self.assertIsInstance(json.dumps(result, ensure_ascii=False), str)


# ---------------------------------------------------------------- 项目数据一致性


class TestProjectBoundaries(unittest.TestCase):
    def test_demo_questions_count(self):
        self.assertEqual(len(NORMAL_QUESTIONS), 3)
        self.assertEqual(len(OVERBOUND_QUESTIONS), 2)

    def test_so2001_matches_records(self):
        order = REPO.get_order("SO-2001")
        self.assertEqual(order["logistics_status"], "已签收")
        self.assertEqual(order["delivered_on"], "2026-10-12")

    def test_unknown_order_is_reported_not_invented(self):
        result = query_order("SO-9999")
        self.assertFalse(result["ok"])
        self.assertIn("未找到", result["message"])


class TestPromptBoundaries(unittest.TestCase):
    """提示词里必须写死的安全边界，缺一条就可能被模型绕过。"""

    def test_system_prompt_contains_boundaries(self):
        self.assertIn("DOC-02-13", SYSTEM_PROMPT)
        self.assertIn("待人工确认", SYSTEM_PROMPT)
        self.assertIn("不直接退款", SYSTEM_PROMPT)

    def test_overbound_question_is_rejected_by_mock(self):
        """越权请求必须被拒答并转人工。

        断言只查语义（拒绝 + 转人工 + 引用边界政策），不写死具体措辞，
        否则每次润色客服话术都会误报为失败。
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": OVERBOUND_QUESTIONS[0]},
        ]
        answer = MockLLM().chat(messages)
        self.assertIn("权限", answer)
        self.assertTrue(
            any(word in answer for word in ("无法", "不能", "无权")),
            f"越权回答里必须出现拒绝动作，实际是：{answer}",
        )
        self.assertIn("人工客服", answer)
        self.assertIn("DOC-02-13", answer)

    def test_logistics_question_returns_manual_ticket_hint(self):
        """物流异常要给出工单/转人工的出口，同样不写死措辞。"""
        answer = MockLLM().chat([{"role": "user", "content": "物流异常怎么办？"}])
        self.assertIn("工单", answer)
        self.assertIn("人工", answer)

    def test_mock_stream_can_be_joined_as_answer(self):
        answer = "".join(MockLLM().stream([{"role": "user", "content": "订单SO-2001已签收，我能退货吗？"}]))
        self.assertIn("退货", answer)


if __name__ == "__main__":
    unittest.main()
