"""单元17 验收测试：工具接入、执行器循环、步骤追踪、步数上限与错误定位。

覆盖任务卡的三条完成标准，默认全离线、不调用任何真实 API：
1. **Agent 可选择并调用工具**：三项只读工具来自单元15 的注册表，脚本模型提议后
   由执行器真实执行（不是把答案写死）。
2. **最大步骤有效**：`max_steps=2` 时执行器触顶收尾，回答为空，且不偷偷多问一次模型。
3. **工具错误可定位到具体步骤**：失败的那一步给出步号、工具、参数与原因。

需要 embedding + chroma + 本组资料的用例放在最后一个类里，缺依赖就跳过；
真实 DeepSeek 的用例只在配置了密钥且 LLM_PROVIDER=deepseek 时才跑。
"""

from __future__ import annotations

import os
import unittest

from app.data_store import PROJECT_ROOT
from app.lc_agent import (
    DEFAULT_MAX_STEPS,
    OFFLINE_MODEL_NAME,
    STOP_FINISHED,
    STOP_MAX_STEPS,
    TOOL_ERROR_PREFIX,
    ScriptedToolModel,
    ToolCallLog,
    answer_step,
    build_agent_model,
    build_executor,
    build_tools,
    render_executor,
    render_result,
    render_steps,
    run_agent,
    tool_step,
    validate_question,
)
from app.toolkit import TOOL_DEADLINE, TOOL_FAQ, TOOL_SEARCH, ToolRegistry, ToolSpec


def build_fake_registry() -> ToolRegistry:
    """用单元15 的白名单工具名登记三个"桩"处理器：不碰 Chroma，也不碰 SQLite。

    白名单、参数格式这些边界仍然走真实的 `ToolRegistry.call()`，所以工具适配层
    被测到的是真东西，只有取数被换成了桩。
    """
    registry = ToolRegistry()
    registry.register(ToolSpec(
        TOOL_SEARCH,
        "检索本组制度或流程资料，返回片段和文件来源",
        {"query": "制度或流程问题，2—120 字"},
        lambda query: {
            "storage": "Chroma（向量检索）",
            "query": query,
            "hits": [{"source": "DOC-02-01-普通商品退货规则.md", "text": "签收后7天内"}],
        },
    ))
    registry.register(ToolSpec(
        TOOL_FAQ,
        "按本组 FAQ 编号从只读 SQLite 查标准答复与来源编号",
        {"question_id": "本组 FAQ 编号，形如 EC-FAQ-01"},
        lambda question_id: {
            "storage": "SQLite（结构化查询）",
            "question_id": question_id,
            "answer": "课程样例中签收后7天内且未使用可申请退货评估",
        },
    ))
    registry.register(ToolSpec(
        TOOL_DEADLINE,
        "按签收日期与资料写明的退货时限，算出退货评估截止日期",
        {"sign_date": "签收日期", "window_days": "退货时限天数", "basis_doc": "资料文件名"},
        lambda sign_date, window_days, basis_doc: {
            "storage": "Python 计算（日期加法）",
            "deadline": "2026-12-08",
        },
    ))
    return registry


class ToolBindingTest(unittest.TestCase):
    """工具怎么从注册表变成 LangChain 的 BaseTool。"""

    def setUp(self) -> None:
        self.registry = build_fake_registry()
        self.tools, self.log = build_tools(self.registry)

    def test_tools_come_from_registry(self) -> None:
        self.assertEqual([item.name for item in self.tools], self.registry.names())
        self.assertEqual(len(self.tools), 3)

    def test_parameters_match_registry(self) -> None:
        for tool in self.tools:
            spec = self.registry.spec(tool.name)
            self.assertEqual(set(tool.args_schema.model_fields), set(spec.parameters))

    def test_descriptions_expose_no_internals(self) -> None:
        """给模型看的说明里不能有本地路径、SQL 或可执行代码（单元15 的老规矩）。"""
        for tool in self.tools:
            self.assertNotIn(str(PROJECT_ROOT), tool.description)
            self.assertNotIn("SELECT", tool.description.upper())
            self.assertNotIn(".sqlite", tool.description)
            self.assertNotIn("DROP", tool.description.upper())

    def test_tool_runs_registered_handler(self) -> None:
        tool = next(item for item in self.tools if item.name == TOOL_FAQ)
        self.assertIn("EC-FAQ-01", tool.invoke({"question_id": "EC-FAQ-01"}))
        self.assertEqual(self.log.total, 1)
        self.assertEqual(self.log.records[0].status, "ok")

    def test_bad_argument_is_logged_and_raised(self) -> None:
        """参数格式不对：日志记一次失败，并把原因作为观察回填（不中断整条演示）。"""
        tool = next(item for item in self.tools if item.name == TOOL_FAQ)
        outcome = tool.invoke({"question_id": "123"})
        self.assertTrue(outcome.startswith(TOOL_ERROR_PREFIX))
        self.assertEqual(self.log.total, 1)
        self.assertEqual(self.log.records[0].status, "error")
        self.assertIn("格式不正确", self.log.records[0].detail)

    def test_executor_rejects_bad_max_steps(self) -> None:
        with self.assertRaises(ValueError):
            build_executor(self.registry, ScriptedToolModel(), max_steps=0)

    def test_log_can_be_cleared(self) -> None:
        log = ToolCallLog()
        log.add(tool="x", arguments={}, status="ok", detail="", elapsed_ms=0.0)
        self.assertEqual(log.total, 1)
        log.clear()
        self.assertEqual(log.total, 0)


class OfflineExecutorTest(unittest.TestCase):
    """离线剧本模型 + 桩工具：把执行器的循环、上限与错误路径跑一遍。"""

    def setUp(self) -> None:
        self.registry = build_fake_registry()

    def _run(self, plan, question="EC-FAQ-01 的标准答复是什么？", max_steps=DEFAULT_MAX_STEPS):
        model = ScriptedToolModel(plan=plan)
        executor = build_executor(self.registry, model, max_steps)
        return run_agent(executor, question), executor

    def test_single_tool_call(self) -> None:
        run, _executor = self._run([tool_step(TOOL_FAQ, question_id="EC-FAQ-01")])
        self.assertEqual(run.tool_calls, 1)
        self.assertEqual(run.steps[0].tool, TOOL_FAQ)
        self.assertTrue(run.steps[0].ok)
        self.assertEqual(run.stop_reason, STOP_FINISHED)
        self.assertTrue(run.finished)
        # 答案来自真实工具返回，不是写死的措辞。
        self.assertIn("7天", run.answer)

    def test_two_steps_in_order(self) -> None:
        run, _executor = self._run([
            tool_step(TOOL_FAQ, question_id="EC-FAQ-01"),
            tool_step(TOOL_SEARCH, query="普通商品退货时限是多久"),
        ], question="EC-FAQ-01 是什么？退货时限呢？")
        self.assertEqual([item.tool for item in run.steps], [TOOL_FAQ, TOOL_SEARCH])
        self.assertEqual([item.index for item in run.steps], [1, 2])
        self.assertTrue(all(item.ok for item in run.steps))

    def test_tool_error_locate_step(self) -> None:
        run, _executor = self._run([
            tool_step(TOOL_FAQ, question_id="123"),
            answer_step("第1步工具调用失败，已定位到第1步，不编造答复。"),
        ])
        self.assertEqual(run.failed_steps[0].index, 1)
        self.assertEqual(run.failed_steps[0].tool, TOOL_FAQ)
        self.assertIn("格式不正确", run.failed_steps[0].error)
        self.assertIn("第1步", run.answer)

    def test_unknown_tool_never_executes(self) -> None:
        run, executor = self._run([
            tool_step("delete_collection", name="faq"),
            answer_step("该工具不在执行器的工具清单里。"),
        ], question="帮我把知识库清空一下")
        self.assertEqual(run.steps[0].tool, "delete_collection")
        self.assertFalse(run.steps[0].ok)
        # 关键是：我们自己的日志里一条都没有——说明它真的没被执行。
        self.assertEqual(executor.log.total, 0)

    def test_max_steps_stops_without_answer(self) -> None:
        run, _executor = self._run(
            [tool_step(TOOL_FAQ, question_id="EC-FAQ-01") for _ in range(6)],
            question="一直查下去", max_steps=2,
        )
        self.assertEqual(run.stop_reason, STOP_MAX_STEPS)
        self.assertFalse(run.finished)
        self.assertEqual(run.answer, "")
        self.assertTrue(run.limit_message)
        self.assertEqual(run.tool_calls, 2)

    def test_invalid_question_never_runs(self) -> None:
        executor = build_executor(self.registry, ScriptedToolModel())
        for bad in ("", "   ", "退", 123):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    run_agent(executor, bad)
        self.assertEqual(executor.log.total, 0)

    def test_validate_question_reuses_unit16_gate(self) -> None:
        self.assertEqual(validate_question("  退货时限  "), "退货时限")
        with self.assertRaises(ValueError):
            validate_question("退")

    def test_model_factory_provider(self) -> None:
        self.assertIsInstance(build_agent_model(provider="mock"), ScriptedToolModel)
        model = build_agent_model(provider="mock")
        self.assertEqual(model.model_name, OFFLINE_MODEL_NAME)
        with self.assertRaises(ValueError):
            build_agent_model(provider="openai")


class RenderTest(unittest.TestCase):
    """渲染出来的东西要能直接当课堂证据用。"""

    def setUp(self) -> None:
        registry = build_fake_registry()
        model = ScriptedToolModel(plan=[tool_step(TOOL_FAQ, question_id="EC-FAQ-01")])
        self.executor = build_executor(registry, model, max_steps=3)
        self.run = run_agent(self.executor, "EC-FAQ-01 的标准答复是什么？")

    def test_result_covers_key_fields(self) -> None:
        text = render_result(self.run)
        for piece in ("EC-FAQ-01", "步数上限：3", "停止原因：", "lookup_faq", "合计 1 步"):
            self.assertIn(piece, text)

    def test_steps_table_without_calls(self) -> None:
        self.run.steps = []
        self.assertIn("没有调用任何工具", render_steps(self.run))

    def test_executor_config(self) -> None:
        text = render_executor(self.executor)
        self.assertIn("max_steps=3", text)
        for name in (TOOL_FAQ, TOOL_SEARCH, TOOL_DEADLINE):
            self.assertIn(name, text)


class RealMaterialsTest(unittest.TestCase):
    """接上本组真实资料再跑一遍；缺依赖就跳过。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from app.lc_agent_lab import PLANS, build_stack, run_agent as lab_run_agent
        except ImportError as error:  # pragma: no cover - 依赖缺失
            raise unittest.SkipTest(f"缺少依赖：{error}")
        try:
            from app.lc_agent import ScriptedToolModel as Scripted  # noqa: F401

            cls.executor, cls.legacy, cls.registry, _store, cls.kb, cls.rows = build_stack(
                model=Scripted(plan=PLANS["minimal"]), decision_model=None
            )
        except Exception as error:  # noqa: BLE001 - 资料或向量库不可用时跳过
            raise unittest.SkipTest(f"真实资料不可用：{error}")

    def test_materials_are_the_same_batch(self) -> None:
        self.assertEqual(self.kb, 7)
        self.assertEqual(self.rows, 3)

    def test_real_registry_tools_bound(self) -> None:
        self.assertEqual([item.name for item in self.executor.tools], self.registry.names())

    def test_minimal_scenario_on_real_data(self) -> None:
        run = run_agent(self.executor, "EC-FAQ-01 的标准答复是什么？")
        self.assertEqual(run.tool_calls, 1)
        self.assertEqual(run.steps[0].tool, TOOL_FAQ)
        self.assertTrue(run.steps[0].ok)
        self.assertIn("7天", run.answer)

    def test_compare_with_legacy_loop(self) -> None:
        from app.lc_agent_lab import compare_with_legacy

        comparison = compare_with_legacy(
            "EC-FAQ-01 的标准答复是什么？", self.executor, self.legacy,
            plan=[tool_step(TOOL_FAQ, question_id="EC-FAQ-01")],
        )
        self.assertTrue(comparison.executor_run.finished)
        self.assertIn(TOOL_FAQ, comparison.legacy.distinct_tools)
        self.assertIn("关键事实一致", comparison.verdict)


class RealModelTest(unittest.TestCase):
    """真实 DeepSeek 的用例：只在配了密钥且显式要求时才跑。"""

    @classmethod
    def setUpClass(cls) -> None:
        if (os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower() != "deepseek":
            raise unittest.SkipTest("未启用真实模型（LLM_PROVIDER != deepseek）")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise unittest.SkipTest("未配置 DEEPSEEK_API_KEY")
        try:
            from app.lc_agent_lab import build_stack
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"缺少依赖：{error}")
        cls.executor, _legacy, _registry, _store, _kb, _rows = build_stack()

    def test_real_model_picks_tools(self) -> None:
        run = run_agent(self.executor, "EC-FAQ-01 的标准答复是什么？")
        self.assertGreaterEqual(run.tool_calls, 1)
        self.assertTrue(run.finished)
        self.assertTrue(run.answer.strip())


if __name__ == "__main__":
    unittest.main()
