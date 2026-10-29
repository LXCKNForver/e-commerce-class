"""单元18 验收测试：路由分支、顺序步骤、每步输入输出、安全兜底。

覆盖任务卡的三条完成标准，默认全离线、不调用任何真实 API：
1. **至少两条分支都能到达**：知识 / FAQ 编号 / 日期计算三条分支各跑一次，
   出口字段（route、来源、答复）都由真实代码路径产出。
2. **每步输入输出可查看**：步骤表里每步都记了输入字段、输出字段与说明，
   且序号连续；出问题的那一步能从说明里看出来。
3. **未知类型进入安全兜底**：三条规则都没命中时不检索、不调模型（模型调用 0 次），
   只给固定话术；分支到了但没证据的路径同样 0 次模型调用。

需要 embedding + chroma + 本组资料的用例放在 `RealMaterialsTest` 里，缺依赖就跳过；
真实 DeepSeek 的用例只在配置了密钥且 LLM_PROVIDER=deepseek 时才跑。
"""

from __future__ import annotations

import os
import unittest
from typing import NamedTuple

from app.flow_chain import (
    FALLBACK_ANSWER,
    NO_EVIDENCE_ANSWER,
    ROUTE_DEADLINE,
    ROUTE_FAQ,
    ROUTE_KNOWLEDGE,
    ROUTE_UNKNOWN,
    STOP_FALLBACK,
    STOP_FINISHED,
    STOP_NO_EVIDENCE,
    STEP_FALLBACK,
    STEP_FINISH,
    STEP_NO_EVIDENCE,
    STEP_ROUTE,
    STEP_SUMMARY,
    ScriptedFlowModel,
    StepLog,
    build_flow_model,
    build_workflow,
    collect_evidence,
    render_concepts,
    render_flow,
    render_routes,
    render_result,
    render_steps,
    route_question,
    run_flow,
)
from app.lc_chain import MIN_SIMILARITY, validate_question
from app.toolkit import TOOL_DEADLINE, TOOL_FAQ, TOOL_SEARCH, ToolRegistry, ToolSpec

FAQ_QUESTION = "EC-FAQ-01 的标准答复是什么？"
KNOWLEDGE_QUESTION = "普通商品退货有哪些要求？"
DEADLINE_QUESTION = "2026年12月1日签收的普通商品，退货评估最晚到哪天？"
UNKNOWN_QUESTION = "帮我写一首关于秋天的诗"


class FakeHit(NamedTuple):
    """检索命中的替身：字段与 `app.retriever.SearchResult` 同名同义。"""

    source: str
    text: str
    distance: float = 0.2
    similarity: float = 0.8


class FakeService:
    """检索服务的替身：只按给定命中返回，不碰 Chroma。

    相似度是可控的——"分支到了但没证据"这条路径只能靠低相似度造出来，
    真实的最近邻永远会返回几条，光靠换个问题演示不出空结果。
    """

    def __init__(self, hits: list[FakeHit] | None = None) -> None:
        self.hits = list(hits if hits is not None else DEFAULT_HITS)
        self.queries: list[str] = []

    def search(self, query: str, project_id: str, top_k: int = 3) -> list[FakeHit]:
        self.queries.append(query)
        # project_id 必须传：与真实服务一样，不允许"查全部"。
        if not project_id:
            raise ValueError("project_id 不能为空")
        return self.hits[:top_k]


DEFAULT_HITS = [
    FakeHit("模拟数据/单元12项目资料/电商售后服务智能体项目组/DOC-02-01-普通商品退货规则.md",
            "本课程样例中，普通商品签收后7天内且未使用可申请退货评估。"),
    FakeHit("模拟数据/单元12项目资料/电商售后服务智能体项目组/常见售后问题.csv",
            "问题编号：EC-FAQ-01；标准答复：签收后7天内且未使用可申请退货评估"),
]


def build_fake_registry(deadline_ok: bool = True, faq_found: bool = True) -> ToolRegistry:
    """用单元15 的白名单工具名登记三个桩处理器：白名单与参数格式仍是真代码。"""

    def lookup(question_id: str) -> dict:
        if not faq_found:
            return {
                "storage": "SQLite（结构化查询）",
                "source": "常见售后问题.csv",
                "question_id": question_id,
                "status": "未找到",
            }
        return {
            "storage": "SQLite（结构化查询）",
            "source": "常见售后问题.csv",
            "question_id": question_id,
            "question": "普通商品签收后还能退货吗",
            "answer": "课程样例中签收后7天内且未使用可申请退货评估，最终由人工复核",
            "source_id": "DOC-02-01",
        }

    def deadline(sign_date: str, window_days: str, basis_doc: str) -> dict:
        if not deadline_ok:
            raise ValueError(f"{basis_doc} 里没有“{window_days}天”这个时限")
        return {
            "storage": "Python 计算（日期加法）",
            "sign_date": sign_date,
            "window_days": window_days,
            "deadline": "2026-12-08",
            "basis_doc": basis_doc,
            "basis_text": "，普通商品签收后7天内",
        }

    registry = ToolRegistry()
    registry.register(ToolSpec(
        TOOL_SEARCH,
        "检索本组制度或流程资料，返回片段和文件来源",
        {"query": "制度或流程问题，2—120 字"},
        lambda query: {"storage": "Chroma（向量检索）", "query": query, "hits": []},
    ))
    registry.register(ToolSpec(
        TOOL_FAQ,
        "按本组 FAQ 编号从只读 SQLite 查标准答复与来源编号",
        {"question_id": "本组 FAQ 编号，形如 EC-FAQ-01"},
        lookup,
    ))
    registry.register(ToolSpec(
        TOOL_DEADLINE,
        "按签收日期与资料写明的退货时限，算出退货评估截止日期",
        {"sign_date": "签收日期", "window_days": "退货时限天数", "basis_doc": "资料文件名"},
        deadline,
    ))
    return registry


def build_parts(
    hits: list[FakeHit] | None = None,
    deadline_ok: bool = True,
    faq_found: bool = True,
):
    """装一套能跑的工作流：桩注册表 + 桩检索 + 离线模型。"""
    registry = build_fake_registry(deadline_ok=deadline_ok, faq_found=faq_found)
    service = FakeService(hits)
    model = ScriptedFlowModel()
    parts = build_workflow(registry, service, model)
    return parts, registry, service, model


# ---------------------------------------------------------------- ① 路由


class RoutingTest(unittest.TestCase):
    """路由是纯规则：同样的问题永远同一条分支，且不消耗模型调用。"""

    def test_faq_route_by_ticket_number(self) -> None:
        decision = route_question(FAQ_QUESTION)
        self.assertEqual(decision.route, ROUTE_FAQ)
        self.assertEqual(decision.matched, "EC-FAQ-01")
        self.assertIn("EC-FAQ-01", decision.reason)

    def test_deadline_route_needs_date_and_due_word(self) -> None:
        self.assertEqual(route_question(DEADLINE_QUESTION).route, ROUTE_DEADLINE)
        # 只有日期、没问时限：不该走计算分支。
        self.assertEqual(route_question("2026年12月1日你们几点上班").route, ROUTE_UNKNOWN)

    def test_knowledge_route_by_business_word(self) -> None:
        decision = route_question(KNOWLEDGE_QUESTION)
        self.assertEqual(decision.route, ROUTE_KNOWLEDGE)
        self.assertEqual(decision.matched, "退货")

    def test_unknown_route_goes_to_fallback(self) -> None:
        for question in (UNKNOWN_QUESTION, "今天天气怎么样", "你好"):
            with self.subTest(question=question):
                self.assertEqual(route_question(question).route, ROUTE_UNKNOWN)

    def test_priority_faq_before_deadline(self) -> None:
        """同时命中编号与日期时先走编号：顺序即优先级，别让规则互相抢。"""
        decision = route_question("EC-FAQ-02：2026年12月1日签收的商品还能退吗，最晚到哪天？")
        self.assertEqual(decision.route, ROUTE_FAQ)

    def test_route_is_reproducible(self) -> None:
        first = route_question(KNOWLEDGE_QUESTION)
        second = route_question("  " + KNOWLEDGE_QUESTION + "  ")
        self.assertEqual((first.route, first.matched), (second.route, second.matched))


# ---------------------------------------------------------------- ② 分支运行


class OfflineWorkflowTest(unittest.TestCase):
    """四条分支 + 两条异常路径，全部离线跑；每条都核对出口字段。"""

    def test_knowledge_branch_reaches_two_model_calls(self) -> None:
        parts, _registry, service, model = build_parts()
        result = run_flow(KNOWLEDGE_QUESTION, parts)
        self.assertEqual(result.route, ROUTE_KNOWLEDGE)
        self.assertTrue(result.has_evidence)
        self.assertEqual(result.stop_reason, STOP_FINISHED)
        self.assertEqual(result.model_calls, 2)
        self.assertEqual(model.calls, 2)
        self.assertEqual(service.queries, [KNOWLEDGE_QUESTION])
        self.assertIn("DOC-02-01-普通商品退货规则.md", result.sources)
        self.assertIn("7天", result.answer)

    def test_faq_branch_uses_readonly_query(self) -> None:
        parts, _registry, service, _model = build_parts()
        result = run_flow(FAQ_QUESTION, parts)
        self.assertEqual(result.route, ROUTE_FAQ)
        self.assertEqual(result.tool, TOOL_FAQ)
        self.assertEqual(result.sources, ["常见售后问题.csv"])
        self.assertTrue(result.finished)
        # 编号分支不问向量库：一次检索都没有发生。
        self.assertEqual(service.queries, [])

    def test_deadline_branch_computes_with_python(self) -> None:
        parts, _registry, _service, _model = build_parts()
        result = run_flow(DEADLINE_QUESTION, parts)
        self.assertEqual(result.route, ROUTE_DEADLINE)
        self.assertIn(TOOL_DEADLINE, result.tool)
        self.assertIn("2026-12-08", result.evidence)
        self.assertEqual(result.sources, ["DOC-02-01-普通商品退货规则.md"])
        self.assertTrue(result.finished)

    def test_unknown_branch_calls_no_model(self) -> None:
        parts, _registry, service, model = build_parts()
        result = run_flow(UNKNOWN_QUESTION, parts)
        self.assertEqual(result.route, ROUTE_UNKNOWN)
        self.assertEqual(result.stop_reason, STOP_FALLBACK)
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(model.calls, 0)
        self.assertEqual(service.queries, [])
        self.assertEqual(result.answer, FALLBACK_ANSWER)
        self.assertIn(STEP_FALLBACK, [step.name for step in result.steps])

    def test_no_evidence_branch_calls_no_model(self) -> None:
        low = [FakeHit("模拟数据/单元12项目资料/电商售后服务智能体项目组/DOC-02-07-物流异常说明.md",
                       "物流轨迹长期没有更新时应人工核实", similarity=0.31)]
        parts, _registry, _service, model = build_parts(low)
        result = run_flow(KNOWLEDGE_QUESTION, parts)
        self.assertEqual(result.route, ROUTE_KNOWLEDGE)
        self.assertFalse(result.has_evidence)
        self.assertEqual(result.stop_reason, STOP_NO_EVIDENCE)
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(model.calls, 0)
        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertIn(str(MIN_SIMILARITY), result.evidence_detail)
        self.assertIn(STEP_NO_EVIDENCE, [step.name for step in result.steps])

    def test_invalid_date_is_not_guessed(self) -> None:
        parts, _registry, _service, _model = build_parts()
        result = run_flow("2026年13月45日签收的商品，退货最晚到哪天？", parts)
        self.assertEqual(result.route, ROUTE_DEADLINE)
        self.assertFalse(result.has_evidence)
        self.assertIn("日期不合法", result.evidence_detail)
        self.assertEqual(result.model_calls, 0)

    def test_missing_window_days_blocks_calculation(self) -> None:
        """资料里没写“N天”就不做日期加法：天数必须来自记录，不来自模型。"""
        no_window = [FakeHit("模拟数据/单元12项目资料/电商售后服务智能体项目组/DOC-02-04-质量问题处理.md",
                             "质量问题需保留订单编号、商品照片和问题描述")]
        parts, _registry, _service, _model = build_parts(no_window)
        result = run_flow(DEADLINE_QUESTION, parts)
        self.assertFalse(result.has_evidence)
        self.assertIn("N天", result.evidence_detail)
        self.assertEqual(result.model_calls, 0)

    def test_missing_faq_id_is_not_faked(self) -> None:
        parts, _registry, _service, model = build_parts(faq_found=False)
        result = run_flow("EC-FAQ-09 的标准答复是什么？", parts)
        self.assertEqual(result.route, ROUTE_FAQ)
        self.assertFalse(result.has_evidence)
        self.assertIn("库里没有编号 EC-FAQ-09", result.evidence_detail)
        self.assertEqual(model.calls, 0)

    def test_deadline_tool_rejection_is_reported(self) -> None:
        parts, _registry, _service, _model = build_parts(deadline_ok=False)
        result = run_flow(DEADLINE_QUESTION, parts)
        self.assertFalse(result.has_evidence)
        self.assertIn("计算工具拒绝", result.evidence_detail)

    def test_invalid_question_never_runs(self) -> None:
        parts, _registry, service, model = build_parts()
        cases: list[object] = ["", "   ", "退", 123, "退货" * 80]
        for raw in cases:
            with self.subTest(raw=repr(raw)[:20]):
                with self.assertRaises(ValueError):
                    run_flow(raw, parts)
        self.assertEqual(parts.log.total, 0)
        self.assertEqual(service.queries, [])
        self.assertEqual(model.calls, 0)

    def test_validate_question_reuses_unit16_gate(self) -> None:
        """守门规则只有一份：本单元直接用单元16 的 validate_question。"""
        self.assertEqual(validate_question(KNOWLEDGE_QUESTION), KNOWLEDGE_QUESTION)
        with self.assertRaises(ValueError):
            validate_question("")

    def test_steps_reset_between_runs(self) -> None:
        parts, _registry, _service, _model = build_parts()
        first = run_flow(KNOWLEDGE_QUESTION, parts)
        second = run_flow(UNKNOWN_QUESTION, parts)
        self.assertEqual(len(first.steps), 4)
        self.assertEqual(len(second.steps), 3)
        self.assertEqual(parts.log.total, 3)


# ---------------------------------------------------------------- ③ 步骤表


class StepTraceTest(unittest.TestCase):
    """每步输入输出可查看：步骤表要说清"谁、吃进什么、吐出什么"。"""

    def setUp(self) -> None:
        parts, _registry, _service, _model = build_parts()
        self.result = run_flow(KNOWLEDGE_QUESTION, parts)

    def test_steps_are_in_order(self) -> None:
        names = [step.name for step in self.result.steps]
        self.assertEqual(
            names, [STEP_ROUTE, "② 取证据（按分支查其中一类资料）", STEP_SUMMARY, STEP_FINISH]
        )
        self.assertEqual([step.index for step in self.result.steps], [1, 2, 3, 4])

    def test_every_step_has_input_and_output(self) -> None:
        for step in self.result.steps:
            with self.subTest(step=step.name):
                self.assertGreater(len(step.input_keys), 0)
                self.assertGreater(len(step.output_keys), 0)
                self.assertTrue(step.detail)
                self.assertGreaterEqual(step.elapsed_ms, 0)

    def test_fields_accumulate_not_replaced(self) -> None:
        """只加不改：后一步的输出字段一定多于前一步，问题字段一路都在。"""
        counts = [len(step.output_keys) for step in self.result.steps]
        self.assertEqual(counts, sorted(counts))
        self.assertIn("question", self.result.steps[-1].output_keys)

    def test_route_step_says_why(self) -> None:
        self.assertIn("退货", self.result.steps[0].detail)

    def test_log_time_lookup(self) -> None:
        log = StepLog()
        log.record(STEP_ROUTE, ("question",), ("question", "route"), "x", 3)
        self.assertEqual(log.time_of(STEP_ROUTE), 3)
        self.assertEqual(log.total, 1)
        log.clear()
        self.assertEqual(log.total, 0)


# ---------------------------------------------------------------- ④ 渲染与配置


class RenderTest(unittest.TestCase):
    def setUp(self) -> None:
        parts, _registry, _service, _model = build_parts()
        self.parts = parts
        self.result = run_flow(KNOWLEDGE_QUESTION, parts)

    def test_result_covers_key_fields(self) -> None:
        text = render_result(self.result)
        for piece in ("分支：knowledge", "取证据：", "模型调用：2 次", "停止原因：正常完成", "答复："):
            self.assertIn(piece, text)

    def test_steps_table_lists_each_step(self) -> None:
        text = render_steps(self.result)
        self.assertIn("共 4 步", text)
        self.assertIn(STEP_SUMMARY, text)

    def test_routes_table_has_four_branches(self) -> None:
        text = render_routes()
        for route in (ROUTE_FAQ, ROUTE_DEADLINE, ROUTE_KNOWLEDGE, ROUTE_UNKNOWN):
            self.assertIn(route, text)

    def test_flow_diagram_marks_fallback(self) -> None:
        text = render_flow()
        self.assertIn("unknown", text)
        self.assertIn("安全兜底", text)

    def test_concepts_and_questions(self) -> None:
        text = render_concepts()
        self.assertIn("Chain 链式编排", text)
        self.assertIn("项目问题清单", text)

    def test_model_factory_provider(self) -> None:
        self.assertIsInstance(build_flow_model("mock"), ScriptedFlowModel)
        with self.assertRaises(ValueError):
            build_flow_model("openai")


# ---------------------------------------------------------------- ⑤ 真实资料


class RealMaterialsTest(unittest.TestCase):
    """接上本组真实资料再跑一遍五条路径；缺依赖就跳过。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from app.flow_lab import build_stack
        except ImportError as error:  # pragma: no cover - 依赖缺失
            raise unittest.SkipTest(f"缺少依赖：{error}")
        try:
            cls.parts, cls.registry, cls.service, cls.kb, cls.rows = build_stack()
        except Exception as error:  # noqa: BLE001 - 资料或向量库不可用时跳过
            raise unittest.SkipTest(f"真实资料不可用：{error}")

    def test_materials_are_the_same_batch(self) -> None:
        self.assertEqual(self.kb, 7)
        self.assertEqual(self.rows, 3)

    def test_knowledge_branch_on_real_data(self) -> None:
        result = run_flow(KNOWLEDGE_QUESTION, self.parts)
        self.assertEqual(result.route, ROUTE_KNOWLEDGE)
        self.assertTrue(result.has_evidence)
        self.assertIn("7天", result.answer)

    def test_faq_branch_on_real_data(self) -> None:
        result = run_flow(FAQ_QUESTION, self.parts)
        self.assertEqual(result.tool, TOOL_FAQ)
        self.assertIn("7天", result.evidence)
        self.assertTrue(result.finished)

    def test_deadline_branch_on_real_data(self) -> None:
        result = run_flow(DEADLINE_QUESTION, self.parts)
        self.assertEqual(result.route, ROUTE_DEADLINE)
        self.assertIn("2026-12-08", result.evidence)
        self.assertTrue(result.finished)

    def test_all_branches_reachable(self) -> None:
        from app.flow_lab import run_class

        run = run_class()
        self.assertGreaterEqual(len(run.reached_routes), 3)
        self.assertIn(ROUTE_UNKNOWN, run.reached_routes)
        # 兜底与无证据两条路径都不许调模型。
        self.assertIn("unknown", run.zero_model_calls)
        self.assertIn("no_evidence", run.zero_model_calls)


class RealModelTest(unittest.TestCase):
    """真实 DeepSeek 的用例：要显式开启（RUN_REAL_MODEL=1）且配了密钥才跑。

    刻意不用 `LLM_PROVIDER` 当开关：那个变量会被别的用例从 `.env` 里读进来，
    整套测试的总数就会随执行顺序变——"多少项通过"是要写进证据文档的数字，不能飘。
    真实模型的效果在 `/flow` 里跑一遍看就行，这里只留一条最小回归。
    """

    @classmethod
    def setUpClass(cls) -> None:
        if (os.getenv("RUN_REAL_MODEL", "") or "").strip() != "1":
            raise unittest.SkipTest("未显式开启真实模型用例（RUN_REAL_MODEL=1）")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise unittest.SkipTest("未配置 DEEPSEEK_API_KEY")
        try:
            from app.flow_lab import build_stack
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"缺少依赖：{error}")
        cls.parts, _registry, _service, _kb, _rows = build_stack(
            model=build_flow_model("deepseek")
        )

    def test_real_model_summarises_on_knowledge_branch(self) -> None:
        result = run_flow(KNOWLEDGE_QUESTION, self.parts)
        self.assertTrue(result.finished)
        self.assertEqual(result.model_calls, 2)
        self.assertIn("7天", result.answer)
        self.assertTrue(result.sources)

    def test_real_model_still_skipped_on_fallback(self) -> None:
        """兜底分支跟模型没关系：真实模型下同样是 0 次调用。"""
        result = run_flow(UNKNOWN_QUESTION, self.parts)
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(result.answer, FALLBACK_ANSWER)


if __name__ == "__main__":
    unittest.main()
