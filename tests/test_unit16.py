"""单元16 验收测试：LCEL 链的六段流转、证据门槛、守门与同题对照。

覆盖五件事，默认全离线、不调用任何真实 API：
1. **输入守门**：空、空白、过短、过长、非字符串——五种坏输入都停在跑链之前。
2. **证据门槛**：相似度不达标的命中不进 Prompt；没有证据时上下文是明确的占位，
   离线模型的答复只能是"资料不足"。
3. **assign 只加字段**：问题一路都在，所以最后仍能打印来源；链返回的字段齐不齐。
4. **六段各有输入输出**：`inspect_stages=True` 时逐段记录的形状与组件名对得上。
5. **同题对照**：与单元15 的 ToolAgent 比对的是来源与数字这类硬事实，不比措辞。

需要 embedding + chroma + 本组资料的用例放在最后一个类里，缺依赖就跳过；
那个类显式传入离线模型与脚本决策模型，跑测试不会联网。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.lc_chain import (
    ANSWER_KEY,
    CONTEXT_KEY,
    EMPTY_CONTEXT,
    HITS_KEY,
    INSUFFICIENT_MARK,
    MAX_QUESTION_CHARS,
    MIN_SIMILARITY,
    OFFLINE_MODEL_NAME,
    QUESTION_KEY,
    SOURCES_KEY,
    STAGE_TABLE,
    TOP_K,
    ChainResult,
    OfflineChatModel,
    build_chain,
    build_chat_model,
    filter_hits,
    format_context,
    hits_to_dicts,
    render_hits,
    render_result,
    render_stage_table,
    run_chain,
    select_sources,
    validate_question,
)
from app.chain_lab import (
    DEFAULT_QUESTION,
    INSUFFICIENT_QUESTION,
    INVALID_CASES,
    build_stack,
    compare_with_agent,
    key_facts,
    render_comparison,
    render_invalid,
    render_reusable,
    render_run,
    render_scenario,
    run_class,
    run_invalid,
    run_scenario,
)
from app.retriever import SearchResult
from app.tool_agent import ScriptedToolModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "e-commerce-after-sales"
KNOWLEDGE_DIR = PROJECT_ROOT / "模拟数据" / "单元12项目资料" / "电商售后服务智能体项目组"

MD_NAME = "DOC-02-01-普通商品退货规则.md"
MD_SOURCE = f"模拟数据/单元12项目资料/电商售后服务智能体项目组/{MD_NAME}"
CSV_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/常见售后问题.csv"
PDF_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/售后受理材料清单.pdf"

GOOD_TEXT = "本课程样例中，普通商品签收后7天内且未使用可申请退货评估。"
WEAK_TEXT = "课程虚构资料，仅供教学演示。"


def _hit(source: str, text: str = GOOD_TEXT, similarity: float = 0.70) -> dict:
    return {"source": source, "text": text, "distance": 0.6, "similarity": similarity}


def _search(hits: list[dict]):
    """假检索函数：链只需要一个"问题进去、片段出来"的可调用对象。"""
    return lambda question: hits


def _parts(hits: list[dict], **kwargs) -> object:
    return build_chain(_search(hits), OfflineChatModel(), **kwargs)


# ---------------------------------------------------------------- 输入守门


class TestValidateQuestion(unittest.TestCase):
    def test_normal_question_passes(self) -> None:
        self.assertEqual(validate_question("普通商品退货有哪些要求？"), "普通商品退货有哪些要求？")

    def test_whitespace_is_collapsed(self) -> None:
        self.assertEqual(validate_question("退货  时限\n是多久"), "退货 时限 是多久")

    def test_empty_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能为空"):
            validate_question("")

    def test_blank_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能为空"):
            validate_question("   \t ")

    def test_too_short_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少 2 个字"):
            validate_question("退")

    def test_too_long_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "最多 120 个字"):
            validate_question("退" * (MAX_QUESTION_CHARS + 1))

    def test_non_string_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须是字符串"):
            validate_question(123)

    def test_every_invalid_case_is_rejected(self) -> None:
        rows = run_invalid(_parts([]))
        self.assertEqual(len(rows), len(INVALID_CASES))
        for _shown, outcome in rows:
            self.assertTrue(outcome.startswith("已拒绝"), outcome)


# ---------------------------------------------------------------- 证据门槛


class TestEvidenceGate(unittest.TestCase):
    def test_hits_above_threshold_are_kept(self) -> None:
        hits = [_hit(MD_SOURCE), _hit(CSV_SOURCE, similarity=0.60)]
        self.assertEqual(len(filter_hits(hits)), 2)

    def test_hits_below_threshold_are_dropped(self) -> None:
        hits = [_hit(MD_SOURCE, similarity=0.40), _hit(CSV_SOURCE, similarity=0.30)]
        self.assertEqual(filter_hits(hits), [])

    def test_missing_similarity_is_not_killed(self) -> None:
        hits = [{"source": MD_SOURCE, "text": GOOD_TEXT, "distance": 0.6, "similarity": None}]
        self.assertEqual(len(filter_hits(hits)), 1)

    def test_empty_hits_give_the_placeholder_context(self) -> None:
        self.assertEqual(format_context([]), EMPTY_CONTEXT)

    def test_weak_hits_give_the_placeholder_context(self) -> None:
        self.assertEqual(format_context([_hit(MD_SOURCE, similarity=0.20)]), EMPTY_CONTEXT)

    def test_context_is_numbered_and_carries_the_source(self) -> None:
        text = format_context([_hit(MD_SOURCE), _hit(CSV_SOURCE, WEAK_TEXT, 0.60)])
        self.assertIn("[1] 来源：", text)
        self.assertIn(MD_NAME, text)
        self.assertIn("[2] 来源：", text)

    def test_sources_are_deduplicated_and_normalized(self) -> None:
        hits = [_hit(MD_SOURCE), _hit(MD_SOURCE), _hit(CSV_SOURCE, WEAK_TEXT, 0.60)]
        self.assertEqual(select_sources(hits), [MD_NAME, "常见售后问题.csv"])

    def test_sources_are_empty_when_evidence_is_weak(self) -> None:
        self.assertEqual(select_sources([_hit(MD_SOURCE, similarity=0.1)]), [])

    def test_threshold_is_a_demo_value_not_accuracy(self) -> None:
        """门槛的语义要写在注释里：这里用断言把默认值钉住，改动时必须同步文档。"""
        self.assertEqual(MIN_SIMILARITY, 0.55)


# ---------------------------------------------------------------- 链的组装


class TestChainAssembly(unittest.TestCase):
    def setUp(self) -> None:
        self.hits = [_hit(MD_SOURCE), _hit(CSV_SOURCE, WEAK_TEXT, 0.60), _hit(PDF_SOURCE, WEAK_TEXT, 0.56)]
        self.parts = _parts(self.hits)

    def test_chain_returns_all_five_fields(self) -> None:
        values = self.parts.chain.invoke({QUESTION_KEY: "普通商品退货有哪些要求"})
        self.assertEqual(
            sorted(values), sorted([QUESTION_KEY, HITS_KEY, CONTEXT_KEY, SOURCES_KEY, ANSWER_KEY])
        )

    def test_question_survives_to_the_end(self) -> None:
        values = self.parts.chain.invoke({QUESTION_KEY: "普通商品退货有哪些要求"})
        self.assertEqual(values[QUESTION_KEY], "普通商品退货有哪些要求")

    def test_answer_is_a_plain_string(self) -> None:
        values = self.parts.chain.invoke({QUESTION_KEY: "普通商品退货有哪些要求"})
        self.assertIsInstance(values[ANSWER_KEY], str)

    def test_sources_come_from_the_hits(self) -> None:
        values = self.parts.chain.invoke({QUESTION_KEY: "普通商品退货有哪些要求"})
        self.assertEqual(values[SOURCES_KEY], [MD_NAME, "常见售后问题.csv", "售后受理材料清单.pdf"])

    def test_retrieve_stage_returns_only_hits(self) -> None:
        self.assertEqual(self.parts.retrieve.invoke({QUESTION_KEY: "退货"}), self.hits)

    def test_assign_keeps_the_original_input(self) -> None:
        values = self.parts.with_hits.invoke({QUESTION_KEY: "退货"})
        self.assertEqual(sorted(values), sorted([QUESTION_KEY, HITS_KEY]))

    def test_prompt_uses_question_and_context(self) -> None:
        rendered = self.parts.prompt.invoke({
            QUESTION_KEY: "退货时限",
            CONTEXT_KEY: "[1] 来源：a\n内容：b",
        })
        messages = rendered.to_messages()
        self.assertEqual(len(messages), 2)
        self.assertIn("退货时限", messages[1].content)
        self.assertIn("[1] 来源：a", messages[1].content)

    def test_parser_converts_the_message_to_text(self) -> None:
        from langchain_core.messages import AIMessage

        self.assertEqual(self.parts.parser.invoke(AIMessage(content="签收后7天内")), "签收后7天内")

    def test_custom_top_k_is_respected_by_the_search_function(self) -> None:
        """检索条数由传入的检索函数决定；链不偷偷改 K。"""
        seen: dict[str, int] = {}

        def search(question: str, top_k: int = TOP_K) -> list[dict]:
            seen[question] = top_k
            return self.hits[:top_k]

        parts = build_chain(search, OfflineChatModel())
        parts.chain.invoke({QUESTION_KEY: "退货"})
        self.assertEqual(seen["退货"], TOP_K)


# ---------------------------------------------------------------- 离线模型


class TestOfflineChatModel(unittest.TestCase):
    def test_offline_model_answers_from_evidence(self) -> None:
        model = OfflineChatModel()
        result = model.invoke([("human", "用户问题：退货\n\n检索资料：\n[1] 来源：a.md\n内容：签收后7天内")])
        self.assertIn("签收后7天内", result.content)
        self.assertIn("a.md", result.content)

    def test_offline_model_refuses_when_context_is_empty(self) -> None:
        model = OfflineChatModel()
        result = model.invoke([("human", "用户问题：清关\n\n检索资料：\n" + EMPTY_CONTEXT)])
        self.assertIn(INSUFFICIENT_MARK, result.content)

    def test_offline_model_is_a_runnable_chat_model(self) -> None:
        from langchain_core.language_models.chat_models import BaseChatModel

        self.assertIsInstance(OfflineChatModel(), BaseChatModel)

    def test_model_factory_picks_the_offline_model(self) -> None:
        self.assertIsInstance(build_chat_model(provider="mock"), OfflineChatModel)

    def test_model_factory_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "mock 或 deepseek"):
            build_chat_model(provider="openai")

    def test_model_factory_explains_missing_key(self) -> None:
        import os

        saved = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
                build_chat_model(provider="deepseek")
        finally:
            if saved is not None:
                os.environ["DEEPSEEK_API_KEY"] = saved


# ---------------------------------------------------------------- 跑链


class TestRunChain(unittest.TestCase):
    def setUp(self) -> None:
        self.hits = [_hit(MD_SOURCE), _hit(CSV_SOURCE, WEAK_TEXT, 0.60)]
        self.parts = _parts(self.hits)

    def test_plain_run_collects_the_evidence(self) -> None:
        result = run_chain("普通商品退货有哪些要求", self.parts)
        self.assertEqual(result.question, "普通商品退货有哪些要求")
        self.assertTrue(result.has_evidence)
        self.assertEqual(result.sources, [MD_NAME, "常见售后问题.csv"])
        self.assertEqual(len(result.hits), 2)

    def test_stage_inspection_records_all_six_stages(self) -> None:
        result = run_chain("普通商品退货有哪些要求", self.parts, inspect_stages=True)
        self.assertEqual(len(result.stages), len(STAGE_TABLE))
        self.assertEqual(result.message_kind, "AIMessage")
        self.assertGreater(result.prompt_chars, 0)

    def test_two_run_modes_agree_on_the_evidence(self) -> None:
        one = run_chain("普通商品退货有哪些要求", self.parts)
        two = run_chain("普通商品退货有哪些要求", self.parts, inspect_stages=True)
        self.assertEqual(one.sources, two.sources)
        self.assertEqual(one.context, two.context)
        self.assertEqual(one.answer, two.answer)

    def test_insufficient_evidence_leads_to_a_refusal(self) -> None:
        parts = _parts([_hit(MD_SOURCE, similarity=0.2)])
        result = run_chain("跨境保税商品的清关时效是多久", parts)
        self.assertFalse(result.has_evidence)
        self.assertEqual(result.sources, [])
        self.assertIn(INSUFFICIENT_MARK, result.answer)
        self.assertTrue(result.insufficient)

    def test_invalid_question_never_reaches_the_chain(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能为空"):
            run_chain("   ", self.parts)

    def test_model_name_is_recorded(self) -> None:
        result = run_chain("普通商品退货有哪些要求", self.parts)
        self.assertIn(OFFLINE_MODEL_NAME, result.model_name)


# ---------------------------------------------------------------- 关键事实与渲染


class TestKeyFacts(unittest.TestCase):
    def test_facts_are_numbers_and_file_names(self) -> None:
        facts = key_facts("签收后7天内可申请", MD_SOURCE)
        self.assertIn("7天", facts)
        self.assertIn(MD_NAME, facts)

    def test_facts_are_deduplicated(self) -> None:
        self.assertEqual(key_facts("7天", "7天"), ["7天"])

    def test_no_facts_when_nothing_to_compare(self) -> None:
        self.assertEqual(key_facts("暂无法确认"), [])


class TestRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.hits = [_hit(MD_SOURCE), _hit(CSV_SOURCE, WEAK_TEXT, 0.60)]
        self.parts = _parts(self.hits)
        self.result = run_chain("普通商品退货有哪些要求", self.parts, inspect_stages=True)

    def test_stage_table_lists_six_components(self) -> None:
        text = render_stage_table()
        for spec in STAGE_TABLE:
            self.assertIn(spec.component, text)
        self.assertIn("RunnableLambda", text)
        self.assertIn("StrOutputParser", text)

    def test_result_contains_question_sources_and_answer(self) -> None:
        text = render_result(self.result)
        self.assertIn("普通商品退货有哪些要求", text)
        self.assertIn(MD_NAME, text)
        self.assertIn("回答：", text)

    def test_hits_render_shows_similarity(self) -> None:
        text = render_hits(self.result)
        self.assertIn("0.700", text)
        self.assertIn("采用为证据的来源", text)

    def test_invalid_render_marks_every_row_rejected(self) -> None:
        text = render_invalid(run_invalid(self.parts))
        for _shown, outcome in run_invalid(self.parts):
            self.assertIn(outcome, text)

    def test_scenario_render_shows_the_error(self) -> None:
        scenario = run_scenario("answer", "   ", self.parts)
        self.assertIn("输入未通过校验", render_scenario(scenario))
        self.assertIsNone(scenario.result)

    def test_reusable_points_mention_the_retrieval_function(self) -> None:
        text = render_reusable()
        self.assertIn("RetrievalService.search", text)
        self.assertIn("ToolAgent（单元15）", text)


# ---------------------------------------------------------------- 真实依赖


class TestWithRealVectorStore(unittest.TestCase):
    """需要 embedding + chroma + 本组资料：四节课按任务卡顺序离线跑一遍。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from app.parse_lab import EXPECTED_RECORDS
        except ImportError as error:  # pragma: no cover - 依赖缺失
            raise unittest.SkipTest(f"缺少依赖：{error}")

        if not (KNOWLEDGE_DIR / "常见售后问题.csv").is_file():
            raise unittest.SkipTest("本组资料目录不在仓库中")

        cls._dir = Path(tempfile.mkdtemp(prefix="chroma-unit16-cls-"))
        # 显式传离线模型与脚本决策模型：跑测试不打真实 API。
        cls.lab_run = run_class(
            database_path=cls._dir,
            model=OfflineChatModel(),
            decision_model=ScriptedToolModel(),
        )
        cls.constants = {"kb": EXPECTED_RECORDS}
        cls.render_run = staticmethod(render_run)  # type: ignore[assignment]

    @classmethod
    def tearDownClass(cls) -> None:
        folder = getattr(cls, "_dir", None)
        if folder is not None:
            shutil.rmtree(folder, True)

    def test_knowledge_base_has_the_expected_records(self) -> None:
        self.assertEqual(self.lab_run.kb_records, self.constants["kb"])

    def test_three_scenarios_ran(self) -> None:
        self.assertEqual(
            [item.key for item in self.lab_run.scenarios], ["steps", "answer", "empty"]
        )

    def test_answer_scenario_found_real_evidence(self) -> None:
        scenario = self.lab_run.scenario("answer")
        assert scenario is not None and scenario.result is not None
        self.assertTrue(scenario.result.has_evidence)
        self.assertTrue(scenario.result.sources)
        self.assertFalse(scenario.result.insufficient)

    def test_empty_scenario_refuses_to_answer(self) -> None:
        scenario = self.lab_run.scenario("empty")
        assert scenario is not None and scenario.result is not None
        self.assertFalse(scenario.result.has_evidence)
        self.assertIn(INSUFFICIENT_MARK, scenario.result.answer)

    def test_steps_scenario_records_six_stages(self) -> None:
        scenario = self.lab_run.scenario("steps")
        assert scenario is not None and scenario.result is not None
        self.assertEqual(len(scenario.result.stages), len(STAGE_TABLE))

    def test_comparison_agrees_on_the_key_facts(self) -> None:
        comparison = self.lab_run.comparison
        assert comparison is not None
        self.assertTrue(comparison.chain_facts)
        self.assertTrue(set(comparison.chain_facts) & set(comparison.baseline_facts))
        self.assertIn("关键事实一致", comparison.verdict)

    def test_render_contains_the_required_sections(self) -> None:
        text = self.render_run(self.lab_run)
        self.assertIn("单元16", text)
        self.assertIn("逐段流转", text)
        self.assertIn("组合链一次运行", text)
        self.assertIn("资料不足", text)
        self.assertIn("无效输入", text)
        self.assertIn("同题对照", text)
        self.assertIn(f"{self.constants['kb']} 条", text)


class TestCompareWithAgentOffline(unittest.TestCase):
    """对照逻辑本身：用假检索也能验证"比的是硬事实、不是措辞"。"""

    def setUp(self) -> None:
        from app.toolkit import build_registry
        from app.faq_db import prepare_faq_db, count_faq

        self._faq_dir = Path(tempfile.mkdtemp(prefix="sqlite-unit16-cmp-"))
        db_path = self._faq_dir / "faq.db"
        prepare_faq_db(KNOWLEDGE_DIR / "常见售后问题.csv", db_path)
        self.faq_rows = count_faq(db_path)

        class FakeRetriever:
            def search(self, question: str, project_id: str = PROJECT_ID, top_k: int = TOP_K):
                return [
                    SearchResult(
                        document_id=f"{project_id}:doc-1",
                        text=GOOD_TEXT,
                        source=MD_SOURCE,
                        project_id=project_id,
                        doc_type="knowledge",
                        distance=0.4,
                        similarity=0.8,
                    )
                ]

        self.service = FakeRetriever()
        self.registry = build_registry(
            self.service, PROJECT_ID, KNOWLEDGE_DIR / "常见售后问题.csv", db_path, KNOWLEDGE_DIR
        )
        self.parts = _parts([_hit(MD_SOURCE, GOOD_TEXT, 0.8)])

    def tearDown(self) -> None:
        shutil.rmtree(self._faq_dir, True)

    def test_faq_db_has_the_expected_rows(self) -> None:
        self.assertEqual(self.faq_rows, 3)

    def test_comparison_verdict_mentions_the_shared_fact(self) -> None:
        comparison = compare_with_agent(
            DEFAULT_QUESTION, self.parts, self.service, decision_model=ScriptedToolModel()
        )
        self.assertIn("关键事实", comparison.verdict)
        self.assertIsNotNone(comparison.baseline)

    def test_comparison_render_shows_both_paths(self) -> None:
        comparison = compare_with_agent(
            DEFAULT_QUESTION, self.parts, self.service, decision_model=ScriptedToolModel()
        )
        text = render_comparison(comparison)
        self.assertIn("LCEL 链（单元16）", text)
        self.assertIn("单元15 ToolAgent", text)
        self.assertIn("结论：", text)

    def test_insufficient_question_still_runs_the_chain(self) -> None:
        parts = _parts([_hit(MD_SOURCE, WEAK_TEXT, 0.2)])
        comparison = compare_with_agent(
            INSUFFICIENT_QUESTION, parts, self.service, decision_model=ScriptedToolModel()
        )
        self.assertTrue(comparison.chain.insufficient)


if __name__ == "__main__":
    unittest.main()
