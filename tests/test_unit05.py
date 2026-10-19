"""单元05 验收测试：注册表、调度器、轮次/异常/日志，以及第二个业务工具。"""

from __future__ import annotations

import unittest

from app.llm import FallbackLLM, MockLLM
from app.registry import (
    DispatchLog,
    ToolParam,
    ToolRegistry,
    ToolSpec,
    build_registry,
)
from app.tool_loop import answer_with_registry
from app.tools import ToolError, lookup_policy_doc


class TestUnit05Registry(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def test_at_least_two_business_tools_registered(self):
        self.assertGreaterEqual(len(self.registry.names()), 2)
        for name in ("query_order", "lookup_policy_doc"):
            self.assertIn(name, self.registry.names())

    def test_schemas_come_from_registry_only(self):
        """模型可见清单与可执行清单必须同源，避免两套名单不一致。"""
        schema_names = {item["function"]["name"] for item in self.registry.schemas()}
        self.assertEqual(schema_names, set(self.registry.names()))

    def test_duplicate_registration_is_rejected(self):
        spec = ToolSpec(
            name="query_order",
            description="重复登记测试",
            params=(ToolParam("order_id", "订单号"),),
            handler=lambda order_id: {},
        )
        with self.assertRaises(ValueError):
            self.registry.register(spec)

    def test_spec_without_params_is_rejected(self):
        with self.assertRaises(ValueError):
            ToolRegistry().register(
                ToolSpec(name="bad", description="缺参数", params=(), handler=lambda: {})
            )

    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            self.registry.call("refund_order", {"order_id": "SO-2001"})
        self.assertIn("未知工具", str(ctx.exception))

    def test_missing_argument_is_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            self.registry.call("check_return_eligibility", {"order_id": "SO-2001"})
        self.assertIn("reason", str(ctx.exception))

    def test_extra_argument_is_rejected(self):
        with self.assertRaises(ToolError) as ctx:
            self.registry.call(
                "query_order", {"order_id": "SO-2001", "amount": 100}
            )
        self.assertIn("amount", str(ctx.exception))

    def test_non_string_argument_is_rejected(self):
        with self.assertRaises(ToolError):
            self.registry.call("query_order", {"order_id": 2001})

    def test_registered_tool_runs(self):
        result = self.registry.call("query_order", {"order_id": " SO-2001 "})
        self.assertTrue(result["ok"])
        self.assertEqual(result["order"]["order_id"], "SO-2001")

    def test_describe_lists_usage_and_params(self):
        text = self.registry.describe()
        self.assertIn("query_order", text)
        self.assertIn("lookup_policy_doc", text)
        self.assertIn("参数", text)


class TestUnit05Dispatcher(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_registry()

    def test_batch_mixes_success_and_rejection(self):
        """单条失败不影响同批其他请求，全部请求都有对应结果。"""
        calls = [
            {"call_id": "c1", "name": "query_order", "arguments": {"order_id": "SO-2001"}},
            {"call_id": "c2", "name": "refund_order", "arguments": {}},
            {"call_id": "c3", "name": "query_order", "arguments": {}},
        ]
        results = self.registry.dispatch(calls, round_index=1)

        self.assertEqual([item["call_id"] for item in results], ["c1", "c2", "c3"])
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[1]["status"], "拒绝")
        self.assertEqual(results[2]["status"], "拒绝")
        self.assertIn("error", results[1]["result"])

    def test_handler_exception_is_contained(self):
        """处理函数抛出非 ToolError 异常时，变成可读结果而不向上冒泡。"""

        def boom(order_id: str) -> dict:
            raise RuntimeError("模拟数据库中断")

        registry = ToolRegistry()
        registry.register(ToolSpec(
            name="unstable",
            description="异常兜底测试",
            params=(ToolParam("order_id", "订单号"),),
            handler=boom,
        ))
        results = registry.dispatch(
            [{"call_id": "x1", "name": "unstable", "arguments": {"order_id": "SO-2001"}}],
            round_index=1,
        )
        self.assertEqual(results[0]["status"], "异常")
        self.assertIn("RuntimeError", results[0]["result"]["error"])

    def test_max_calls_limit(self):
        calls = [
            {"call_id": f"c{i}", "name": "query_order", "arguments": {"order_id": "SO-2001"}}
            for i in range(3)
        ]
        with self.assertRaises(ToolError):
            self.registry.dispatch(calls, max_calls=2)

    def test_invalid_max_calls(self):
        with self.assertRaises(ValueError):
            self.registry.dispatch([], max_calls=0)

    def test_dispatch_log_records_every_call(self):
        self.registry.dispatch(
            [
                {"call_id": "c1", "name": "query_order", "arguments": {"order_id": "SO-2001"}},
                {"call_id": "c2", "name": "delete_order", "arguments": {}},
            ],
            round_index=2,
        )
        log = self.registry.log
        self.assertEqual(log.total, 2)
        self.assertEqual(log.failed, 1)
        self.assertEqual(log.entries[0]["round"], 2)
        self.assertIn("状态", log.render())


class TestUnit05DispatchLog(unittest.TestCase):
    def test_empty_log_renders_hint(self):
        self.assertIn("还没有调度记录", DispatchLog().render())

    def test_summary_counts(self):
        log = DispatchLog()
        log.add(round=1, call_id="a", name="query_order", arguments=None, status="ok", detail="", elapsed_ms=1.0)
        log.add(round=1, call_id="b", name="x", arguments=None, status="拒绝", detail="", elapsed_ms=1.0)
        self.assertEqual(log.summary(), {"total": 2, "ok": 1, "error": 1})
        log.clear()
        self.assertEqual(log.total, 0)


class TestUnit05SecondTool(unittest.TestCase):
    def test_lookup_policy_doc_returns_source_text(self):
        result = lookup_policy_doc("DOC-02-05")
        self.assertTrue(result["ok"])
        self.assertEqual(result["doc_id"], "DOC-02-05")
        self.assertIn("定制", result["excerpt"])
        self.assertIn("模拟数据/knowledge", result["source"])

    def test_lookup_policy_doc_is_case_insensitive(self):
        self.assertTrue(lookup_policy_doc("doc-02-01")["ok"])

    def test_lookup_policy_doc_missing_source(self):
        result = lookup_policy_doc("DOC-99-99")
        self.assertFalse(result["ok"])
        self.assertIn("未找到", result["message"])

    def test_lookup_policy_doc_bad_format(self):
        result = lookup_policy_doc("退货规则")
        self.assertFalse(result["ok"])
        self.assertIn("格式不正确", result["message"])

    def test_lookup_policy_doc_never_judges_order(self):
        """政策工具只摘录原文，不对订单下结论。"""
        result = lookup_policy_doc("DOC-02-05")
        self.assertNotIn("eligible", result)
        self.assertIn("未判断是否适用", result["note"])


class TestUnit05LoopWithRegistry(unittest.TestCase):
    def test_two_tools_are_both_dispatched(self):
        """单元05 完成标准：两个业务工具都能被调度。"""
        llm = MockLLM()
        registry = build_registry()

        messages = [{"role": "user", "content": "订单SO-2001已签收，我能退货吗？"}]
        answer, trace = answer_with_registry(llm, messages, registry, require_tool_call=True)
        self.assertTrue(trace)
        self.assertEqual(trace[0]["status"], "ok")
        self.assertTrue(answer)

        messages2 = [{"role": "user", "content": "请给我看看 DOC-02-05 的政策原文怎么写。"}]
        answer2, trace2 = answer_with_registry(llm, messages2, registry, require_tool_call=True)
        self.assertEqual(trace2[0]["name"], "lookup_policy_doc")
        self.assertIn("DOC-02-05", answer2)

    def test_new_turn_does_not_reuse_previous_tool_result(self):
        """多轮会话里新一轮提问必须重新调度，不能拿上一轮的工具结果作答。"""
        llm = MockLLM()
        registry = build_registry()
        dialog: list[dict] = []

        dialog.append({"role": "user", "content": "请给我看看 DOC-02-05 的政策原文"})
        _, trace1 = answer_with_registry(llm, dialog, registry, require_tool_call=False)
        self.assertEqual(trace1[-1]["name"], "lookup_policy_doc")

        dialog.append({"role": "user", "content": "订单SO-2001已签收，我能退货吗？"})
        answer2, trace2 = answer_with_registry(llm, dialog, registry, require_tool_call=False)
        self.assertEqual(trace2[-1]["name"], "check_return_eligibility")
        self.assertIn("SO-2001", answer2)

    def test_max_rounds_stops_runaway_model(self):

        class LoopLLM:
            """每轮都请求工具，永远不给最终回答。"""

            def __init__(self) -> None:
                self.calls = 0

            def complete(self, messages, tools=None):
                self.calls += 1
                from app.llm import MockAssistantMessage, MockToolCall, MockToolCallFunction

                return MockAssistantMessage(
                    content=None,
                    tool_calls=[MockToolCall(
                        id=f"call_{self.calls}",
                        function=MockToolCallFunction(
                            name="query_order",
                            arguments='{"order_id": "SO-2001"}',
                        ),
                    )],
                )

        llm = LoopLLM()
        with self.assertRaises(RuntimeError) as ctx:
            answer_with_registry(
                llm, [{"role": "user", "content": "查订单SO-2001"}], build_registry(), max_rounds=2
            )
        self.assertIn("2轮", str(ctx.exception))
        self.assertEqual(llm.calls, 2)

    def test_no_tool_call_in_strict_mode_is_exposed(self):
        llm = MockLLM()
        with self.assertRaises(RuntimeError):
            answer_with_registry(llm, [{"role": "user", "content": "你好"}], build_registry())
        # 交互模式放宽，允许直接作答。
        answer, trace = answer_with_registry(
            llm, [{"role": "user", "content": "你好"}], build_registry(), require_tool_call=False
        )
        self.assertEqual(trace, [])
        self.assertIn("订单号", answer)


class TestUnit05Fallback(unittest.TestCase):
    """API 欠费/密钥失效/断网时，证据生成不能整段崩掉。"""

    @staticmethod
    def _broken_llm():
        class Broken:
            def chat(self, messages):
                raise RuntimeError("模拟 402 Insufficient Balance")

            def complete(self, messages, tools=None):
                raise RuntimeError("模拟 402 Insufficient Balance")

            def stream(self, messages):
                raise RuntimeError("模拟 402 Insufficient Balance")
                yield  # pragma: no cover - 仅为让本方法成为生成器

        return Broken()

    def test_chat_falls_back_and_records_reason(self):
        llm = FallbackLLM(self._broken_llm())
        answer = llm.chat([{"role": "user", "content": "你好"}])
        self.assertTrue(llm.degraded)
        self.assertEqual(len(llm.degradations), 1)
        self.assertIn("402", llm.degradations[0])
        self.assertIn("chat()", llm.degradations[0])
        self.assertTrue(answer)  # 离线模型仍然给出了回答

    def test_complete_falls_back(self):
        llm = FallbackLLM(self._broken_llm())
        message = llm.complete([{"role": "user", "content": "查订单SO-2001"}], [])
        self.assertTrue(llm.degraded)
        self.assertIn("complete()", llm.degradations[0])
        self.assertIsNotNone(message)

    def test_stream_falls_back_even_when_error_raised_on_iteration(self):
        llm = FallbackLLM(self._broken_llm())
        chunks = list(llm.stream([{"role": "user", "content": "订单SO-2001能退货吗？"}]))
        self.assertTrue(llm.degraded)
        self.assertIn("stream()", llm.degradations[0])
        self.assertTrue("".join(chunks))

    def test_degrades_only_once_and_reuses_fallback(self):
        """第一次失败后不应反复试探远程模型，避免拖慢生成。"""
        llm = FallbackLLM(self._broken_llm())
        llm.chat([{"role": "user", "content": "你好"}])
        llm.chat([{"role": "user", "content": "再问一次"}])
        self.assertEqual(len(llm.degradations), 1)

    def test_healthy_llm_never_degrades(self):
        llm = FallbackLLM(MockLLM())
        llm.chat([{"role": "user", "content": "订单SO-2001已签收，我能退货吗？"}])
        self.assertFalse(llm.degraded)
        self.assertEqual(llm.degradations, [])


if __name__ == "__main__":
    unittest.main()
