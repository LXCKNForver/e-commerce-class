"""单元15 验收测试：工具白名单、三类数据源、循环边界与轨迹渲染。

覆盖四件事，均不调用真实 API：
1. **放行前校验**：工具白名单、字段多一个少一个、参数格式、注入串、类型——
   这些全部发生在执行之前，是"模型只能提议"的证据。
2. **三类数据源**：检索走 Chroma（假检索器）、FAQ 走只读参数化 SQL、
   日期走 Python 计算且天数必须先在资料原文里出现。
3. **循环与停止**：单工具一次答完、组合问题连续调用两个不同工具、
   拒绝路径停在原地、轮次耗尽不补答案、finish 来源必须真观察过。
4. **编排与渲染**：`ToolKitRun` 的场景封装与四段输出是否含课堂要交的字段。

需要 embedding + chroma 的真实入库用例单独放在最后一个类里，缺依赖就跳过。
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app.faq_db import (
    FAQ_TABLE,
    REQUIRED_FIELDS,
    SELECT_BY_ID,
    connect_readonly,
    count_faq,
    fetch_faq,
    prepare_faq_db,
    read_faq_rows,
)
from app.retriever import SearchResult
from app.tool_agent import (
    DEFAULT_MAX_STEPS,
    FINISHED,
    FINISH_TOOL,
    INSUFFICIENT_MARK,
    MIN_MAX_STEPS,
    REJECTED,
    STOPPED_BY_LIMIT,
    ScriptedRebelModel,
    ScriptedToolModel,
    ToolAgent,
    build_tool_decision_model,
    render_result,
    render_trace,
    split_items,
)
from app.toolkit import (
    ALLOWED_TOOLS,
    ARGUMENT_RULES,
    TOOL_DEADLINE,
    TOOL_FAQ,
    TOOL_SEARCH,
    ToolRegistry,
    ToolRejected,
    ToolSpec,
    authorize_tool,
    build_registry,
    compute_return_deadline,
    lookup_faq,
    render_rejections,
    render_tool_list,
    search_knowledge,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "e-commerce-after-sales"
KNOWLEDGE_DIR = (
    PROJECT_ROOT / "模拟数据" / "单元12项目资料" / "电商售后服务智能体项目组"
)
FAQ_FILE = KNOWLEDGE_DIR / "常见售后问题.csv"

MD_NAME = "DOC-02-01-普通商品退货规则.md"
MD_SOURCE = f"模拟数据/单元12项目资料/电商售后服务智能体项目组/{MD_NAME}"
PDF_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/售后受理材料清单.pdf"

SINGLE_QUESTION = "EC-FAQ-01 的标准答复是什么？"
COMBO_QUESTION = "EC-FAQ-01 的标准答复是什么？另外普通商品退货时限是多久？"
CALC_QUESTION = "2026年12月1日签收的普通商品，退货评估最晚到哪天？"


def _hit(source: str, text: str = "签收后7天内且未使用可申请退货评估。") -> SearchResult:
    return SearchResult(
        document_id=f"{PROJECT_ID}:{Path(source).name}:row:2",
        text=text,
        source=source,
        project_id=PROJECT_ID,
        doc_type=Path(source).suffix.lstrip("."),
        distance=0.5,
    )


class FakeRetriever:
    """按关键词返回固定证据，并记录每次调用的 query 与 project_id。"""

    def __init__(self, hits: list[SearchResult] | None = None):
        self.hits = hits if hits is not None else [_hit(MD_SOURCE), _hit(PDF_SOURCE)]
        self.queries: list[str] = []
        self.projects: list[str] = []

    def search(self, question: str, project_id: str = PROJECT_ID, top_k: int = 3,
               source: str | None = None, doc_type: str | None = None) -> list[SearchResult]:
        self.queries.append(question)
        self.projects.append(project_id)
        return list(self.hits)


class EmptyRetriever(FakeRetriever):
    """什么都不命中：验证"没观察到数据时只能答资料不足"。"""

    def search(self, question: str, project_id: str = PROJECT_ID, top_k: int = 3,
               source: str | None = None, doc_type: str | None = None) -> list[SearchResult]:
        self.queries.append(question)
        self.projects.append(project_id)
        return []


def make_registry(tmp: Path, retriever: FakeRetriever | None = None) -> ToolRegistry:
    """真实注册表 + 临时 FAQ 库：三条工具都在，但落点可控。"""
    db_path = tmp / "faq.db"
    prepare_faq_db(FAQ_FILE, db_path)
    return build_registry(
        retriever or FakeRetriever(), PROJECT_ID, FAQ_FILE, db_path, KNOWLEDGE_DIR
    )


def make_agent(tmp: Path, retriever: FakeRetriever | None = None, model=None):
    registry = make_registry(tmp, retriever)
    decision = model if model is not None else ScriptedToolModel()
    return ToolAgent(decision, registry), registry, decision


class ScriptedModel:
    """按剧本逐轮返回动作，并记录收到的对话历史（用来看观察有没有回填）。"""

    def __init__(self, script: list[dict]):
        self.script = script
        self.seen_messages: list[list[dict]] = []
        self.calls: int = 0

    def decide(self, messages: list[dict]) -> str:
        self.calls += 1
        self.seen_messages.append(list(messages))
        index = min(self.calls - 1, len(self.script) - 1)
        return json.dumps(self.script[index], ensure_ascii=False)


# ---------------------------------------------------------------- ① 放行前校验


class TestAuthorizeTool(unittest.TestCase):
    """模型提议与程序执行之间的那道门。"""

    def test_unknown_tool_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "工具未授权"):
            authorize_tool("delete_collection", {"name": "faq"})

    def test_sql_injection_tool_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "工具未授权"):
            authorize_tool("lookup_faq; DROP TABLE faq", {})

    def test_non_dict_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "JSON 对象"):
            authorize_tool(TOOL_SEARCH, "退货时限")  # type: ignore[arg-type]

    def test_missing_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "只接受参数"):
            authorize_tool(TOOL_FAQ, {})

    def test_extra_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "只接受参数"):
            authorize_tool(TOOL_SEARCH, {"query": "退货时限", "project_id": "other"})

    def test_bad_faq_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "格式不正确"):
            authorize_tool(TOOL_FAQ, {"question_id": "123"})

    def test_faq_id_injection_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "格式不正确"):
            authorize_tool(TOOL_FAQ, {"question_id": "EC-FAQ-01; DROP TABLE faq"})

    def test_short_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "格式不正确"):
            authorize_tool(TOOL_SEARCH, {"query": "退"})

    def test_integer_window_days_is_accepted(self) -> None:
        clean = authorize_tool(
            TOOL_DEADLINE,
            {"sign_date": "2026-12-01", "window_days": 7, "basis_doc": MD_NAME},
        )
        self.assertEqual(clean["window_days"], "7")

    def test_boolean_argument_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "类型不正确"):
            authorize_tool(TOOL_FAQ, {"question_id": True})  # type: ignore[dict-item]

    def test_clean_arguments_are_stripped(self) -> None:
        clean = authorize_tool(TOOL_FAQ, {"question_id": "  EC-FAQ-01  "})
        self.assertEqual(clean["question_id"], "EC-FAQ-01")

    def test_every_allowed_tool_has_rules(self) -> None:
        self.assertEqual(set(ARGUMENT_RULES), set(ALLOWED_TOOLS))


class TestToolRegistry(unittest.TestCase):
    """注册表的契约：模型看到的清单 = 程序肯执行的清单。"""

    def test_register_outside_whitelist_fails(self) -> None:
        registry = ToolRegistry()
        with self.assertRaisesRegex(ValueError, "不在白名单"):
            registry.register(ToolSpec("shell", "任意命令", {}, lambda: {}))

    def test_duplicate_registration_fails(self) -> None:
        registry = ToolRegistry()
        spec = ToolSpec(TOOL_SEARCH, "检索", {"query": "问题"}, lambda query: {})
        registry.register(spec)
        with self.assertRaisesRegex(ValueError, "重复注册"):
            registry.register(spec)

    def test_descriptions_do_not_leak_paths(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolSpec(TOOL_FAQ, "按编号查", {"question_id": "编号"}, lambda q: {}))
        text = registry.descriptions()
        self.assertIn(TOOL_FAQ, text)
        self.assertNotIn(str(PROJECT_ROOT), text)
        self.assertNotIn("SELECT", text)

    def test_spec_of_unknown_tool_raises(self) -> None:
        registry = ToolRegistry()
        with self.assertRaisesRegex(ToolRejected, "工具未授权"):
            registry.spec("delete_collection")

    def test_call_validates_before_executing(self) -> None:
        registry = ToolRegistry()
        calls: list[str] = []
        registry.register(
            ToolSpec(
                TOOL_SEARCH,
                "检索",
                {"query": "问题"},
                lambda query: calls.append(query) or {"hits": []},
            )
        )
        with self.assertRaises(ToolRejected):
            registry.call(TOOL_SEARCH, {"query": "x"})
        self.assertEqual(calls, [])


# ---------------------------------------------------------------- ② 三类数据源


class TestFaqDatabase(unittest.TestCase):
    """结构化那一侧：只读连接 + 参数化 SQL。"""

    def setUp(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="sqlite-unit15-"))
        self.addCleanup(shutil.rmtree, self._dir, True)
        self.db_path = self._dir / "faq.db"
        prepare_faq_db(FAQ_FILE, self.db_path)

    def test_rows_match_the_csv(self) -> None:
        self.assertEqual(count_faq(self.db_path), len(read_faq_rows(FAQ_FILE)))

    def test_required_fields_are_present(self) -> None:
        for row in read_faq_rows(FAQ_FILE):
            self.assertTrue(set(REQUIRED_FIELDS).issubset(row))

    def test_fetch_by_id(self) -> None:
        row = fetch_faq(self.db_path, "EC-FAQ-01")
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["question_id"], "EC-FAQ-01")
        self.assertTrue(row["answer"])
        self.assertTrue(row["source_id"])

    def test_unknown_id_returns_none(self) -> None:
        self.assertIsNone(fetch_faq(self.db_path, "EC-FAQ-99"))

    def test_connection_is_readonly(self) -> None:
        connection = connect_readonly(self.db_path)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute(f"DELETE FROM {FAQ_TABLE}")
        finally:
            connection.close()

    def test_sql_is_parameterised(self) -> None:
        self.assertIn("?", SELECT_BY_ID)
        self.assertIn("WHERE question_id = ?", SELECT_BY_ID)

    def test_rerun_does_not_duplicate(self) -> None:
        prepare_faq_db(FAQ_FILE, self.db_path)
        prepare_faq_db(FAQ_FILE, self.db_path)
        self.assertEqual(count_faq(self.db_path), 3)

    def test_missing_csv_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_faq_rows(self._dir / "不存在.csv")


class TestLookupFaqTool(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="sqlite-unit15-"))
        self.addCleanup(shutil.rmtree, self._dir, True)
        self.db_path = self._dir / "faq.db"
        prepare_faq_db(FAQ_FILE, self.db_path)

    def test_found_row_carries_source(self) -> None:
        observation = lookup_faq(self.db_path, FAQ_FILE, "EC-FAQ-01")
        self.assertEqual(observation["question_id"], "EC-FAQ-01")
        self.assertEqual(observation["source"], FAQ_FILE.name)
        self.assertTrue(observation["source_id"])

    def test_missing_row_says_not_found(self) -> None:
        observation = lookup_faq(self.db_path, FAQ_FILE, "EC-FAQ-99")
        self.assertEqual(observation["status"], "未找到")
        self.assertNotIn("answer", observation)


class TestSearchTool(unittest.TestCase):
    def test_hits_carry_sources_only(self) -> None:
        retriever = FakeRetriever([_hit(MD_SOURCE)])
        observation = search_knowledge(retriever, PROJECT_ID, "退货时限")  # type: ignore[arg-type]
        self.assertEqual(retriever.projects, [PROJECT_ID])
        self.assertEqual([hit["source"] for hit in observation["hits"]], [MD_SOURCE])

    def test_empty_hits_are_reported_as_empty(self) -> None:
        observation = search_knowledge(EmptyRetriever(), PROJECT_ID, "查不到的东西")  # type: ignore[arg-type]
        self.assertEqual(observation["hits"], [])


class TestDeadlineTool(unittest.TestCase):
    """计算工具：天数必须来自资料原文，算的人必须是 Python。"""

    def test_deadline_is_computed_by_python(self) -> None:
        observation = compute_return_deadline(KNOWLEDGE_DIR, MD_NAME, "2026-12-01", "7")
        self.assertEqual(observation["deadline"], "2026-12-08")
        self.assertEqual(observation["basis_doc"], MD_NAME)

    def test_window_must_exist_in_the_document(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "没有“30天”"):
            compute_return_deadline(KNOWLEDGE_DIR, MD_NAME, "2026-12-01", "30")

    def test_basis_doc_must_be_a_real_file(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "不在本组资料中"):
            compute_return_deadline(KNOWLEDGE_DIR, "随便写的文件.md", "2026-12-01", "7")

    def test_invalid_date_is_rejected(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "签收日期不合法"):
            compute_return_deadline(KNOWLEDGE_DIR, MD_NAME, "2026-13-40", "7")

    def test_no_path_traversal(self) -> None:
        with self.assertRaisesRegex(ToolRejected, "不在本组资料中"):
            compute_return_deadline(KNOWLEDGE_DIR, "../../../secret.md", "2026-12-01", "7")


# ---------------------------------------------------------------- ③ 循环与停止


class TestToolAgentLoop(unittest.TestCase):
    """四类场景：单工具、连续调用、拒绝、轮次耗尽。"""

    def setUp(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="toolkit-unit15-"))
        self.addCleanup(shutil.rmtree, self._dir, True)

    def test_single_tool_finishes(self) -> None:
        agent, _registry, _model = make_agent(self._dir)
        result = agent.run(SINGLE_QUESTION)
        self.assertTrue(result.finished)
        self.assertEqual(result.tool_names, [TOOL_FAQ])
        self.assertEqual(result.steps, 2)
        self.assertIn(FAQ_FILE.name, "、".join(result.sources))

    def test_combo_uses_two_different_tools(self) -> None:
        agent, _registry, _model = make_agent(self._dir)
        result = agent.run(COMBO_QUESTION)
        self.assertTrue(result.finished)
        self.assertEqual(result.distinct_tools, [TOOL_FAQ, TOOL_SEARCH])

    def test_calc_uses_search_then_deadline(self) -> None:
        agent, _registry, _model = make_agent(self._dir)
        result = agent.run(CALC_QUESTION)
        self.assertTrue(result.finished)
        self.assertEqual(result.distinct_tools, [TOOL_SEARCH, TOOL_DEADLINE])
        self.assertIn("2026-12-08", result.answer)

    def test_unknown_tool_is_rejected_before_execution(self) -> None:
        agent, registry, _ = make_agent(self._dir)
        agent.decision_model = ScriptedRebelModel("delete_collection", {"name": "faq"})
        result = agent.run("请处理一下这个售后问题")
        self.assertTrue(result.rejected)
        self.assertEqual(result.stop_reason, REJECTED)
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(result.answer, "")
        self.assertIn("工具未授权", result.reject_reason)

    def test_bad_argument_is_rejected_before_execution(self) -> None:
        agent, _registry, _ = make_agent(self._dir)
        agent.decision_model = ScriptedRebelModel(TOOL_FAQ, {"question_id": "123"})
        result = agent.run("请处理一下这个售后问题")
        self.assertTrue(result.rejected)
        self.assertEqual(result.tool_calls, 0)
        self.assertIn("格式不正确", result.reject_reason)

    def test_max_steps_leaves_no_answer(self) -> None:
        script = [
            {"reason": "查一次", "action": TOOL_SEARCH, "arguments": {"query": "退货时限"}},
            {"reason": "再查一次", "action": TOOL_SEARCH, "arguments": {"query": "材料清单"}},
        ]
        agent, _registry, model = make_agent(self._dir, model=ScriptedModel(script))
        result = agent.run("退货时限？另外材料清单？", MIN_MAX_STEPS)
        self.assertTrue(result.limited)
        self.assertEqual(result.stop_reason, STOPPED_BY_LIMIT)
        self.assertEqual(result.answer, "")
        self.assertEqual(result.decision_calls, MIN_MAX_STEPS)
        self.assertEqual(model.calls, MIN_MAX_STEPS)

    def test_observation_is_fed_back_to_the_model(self) -> None:
        # 脚本模型把"看到的东西"记下来：第 2 轮收到的消息里必须带观察前缀。
        script = [
            {"reason": "查编号", "action": TOOL_FAQ, "arguments": {"question_id": "EC-FAQ-01"}},
            {
                "reason": "汇总",
                "action": FINISH_TOOL,
                "answer": "签收后7天内可申请退货评估。",
                "sources": [],
            },
        ]
        agent, _registry, model = make_agent(self._dir, model=ScriptedModel(script))
        agent.run(SINGLE_QUESTION)
        self.assertTrue(model_seen_observation(model))

    def test_finish_sources_must_be_observed(self) -> None:
        agent, _registry, _ = make_agent(self._dir)
        script = [
            {
                "reason": "编一个来源",
                "action": FINISH_TOOL,
                "answer": "签收后7天内可申请。",
                "sources": ["根本没查过的文件.md"],
            }
        ]
        agent.decision_model = ScriptedModel(script)
        with self.assertRaisesRegex(ValueError, "未观察到的来源"):
            agent.run(SINGLE_QUESTION)

    def test_finish_without_observation_must_say_insufficient(self) -> None:
        agent, _registry, _ = make_agent(self._dir)
        script = [
            {"reason": "直接作答", "action": FINISH_TOOL, "answer": "7天内可以退。", "sources": []}
        ]
        agent.decision_model = ScriptedModel(script)
        with self.assertRaisesRegex(ValueError, INSUFFICIENT_MARK):
            agent.run(SINGLE_QUESTION)

    def test_empty_retrieval_answers_insufficient(self) -> None:
        agent, _registry, _ = make_agent(self._dir, retriever=EmptyRetriever())
        result = agent.run("普通商品退货时限是多久？")
        self.assertTrue(result.finished)
        self.assertIn(INSUFFICIENT_MARK, result.answer)
        self.assertEqual(result.sources, [])

    def test_parse_decision_requires_json_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "有效 JSON"):
            ToolAgent.parse_decision("我想查一下退货时限")
        with self.assertRaisesRegex(ValueError, "JSON 对象"):
            ToolAgent.parse_decision("[1, 2]")

    def test_question_and_steps_are_validated(self) -> None:
        agent, _registry, _ = make_agent(self._dir)
        with self.assertRaisesRegex(ValueError, "问题不能为空"):
            agent.run("   ")
        with self.assertRaisesRegex(ValueError, "整数"):
            agent.run(SINGLE_QUESTION, "3")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "至少"):
            agent.run(SINGLE_QUESTION, MIN_MAX_STEPS - 1)


def model_seen_observation(model) -> bool:
    """脚本型模型是否真的收到了带观察前缀的回填消息。"""
    from app.tool_agent import OBSERVATION_PREFIX

    for messages in getattr(model, "seen_messages", []):
        for message in messages:
            if OBSERVATION_PREFIX in str(message.get("content", "")):
                return True
    return False


class TestItemSplit(unittest.TestCase):
    """离线脚本按事项推进的依据；也是演示题是否真有"两个事项"的自检。"""

    def test_combo_question_has_two_items(self) -> None:
        self.assertEqual(len(split_items(COMBO_QUESTION)), 2)

    def test_single_question_has_one_item(self) -> None:
        self.assertEqual(len(split_items("普通商品退货时限是多久？")), 1)

    def test_empty(self) -> None:
        self.assertEqual(split_items("  "), [])


# ---------------------------------------------------------------- ④ 渲染与编排


class TestRendering(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="toolkit-unit15-"))
        self.addCleanup(shutil.rmtree, self._dir, True)

    def test_tool_list_has_four_elements(self) -> None:
        registry = make_registry(self._dir)
        text = render_tool_list(registry)
        for name in (TOOL_SEARCH, TOOL_FAQ, TOOL_DEADLINE):
            self.assertIn(name, text)
        self.assertIn("name=", text)
        self.assertIn("description=", text)
        self.assertIn("parameters=", text)
        self.assertIn("execute=", text)

    def test_rejections_are_all_refused(self) -> None:
        registry = make_registry(self._dir)
        text = render_rejections(registry)
        self.assertNotIn("竟然通过了", text)
        self.assertGreaterEqual(text.count("已拒绝"), 8)
        self.assertIn("DROP TABLE", text)

    def test_render_result_and_trace(self) -> None:
        agent, _registry, _ = make_agent(self._dir)
        result = agent.run(COMBO_QUESTION)
        text = render_result(result)
        self.assertIn(FINISHED, text)
        self.assertIn("工具调用：2 次", text)
        trace = render_trace(result)
        self.assertIn(TOOL_FAQ, trace)
        self.assertIn(TOOL_SEARCH, trace)
        self.assertIn(FINISH_TOOL, trace)

    def test_rejected_result_renders_reason(self) -> None:
        agent, _registry, _ = make_agent(self._dir)
        agent.decision_model = ScriptedRebelModel("drop_all", {})
        text = render_result(agent.run("清理一下知识库"))
        self.assertIn(REJECTED, text)
        self.assertIn("工具未授权", text)


class TestDecisionModelFactory(unittest.TestCase):
    def test_mock_returns_scripted_model(self) -> None:
        model = build_tool_decision_model(None, provider="mock")
        self.assertIsInstance(model, ScriptedToolModel)

    def test_unknown_provider_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "mock 或 deepseek"):
            build_tool_decision_model(None, provider="openai")


# ---------------------------------------------------------------- 真实依赖


class TestWithRealVectorStore(unittest.TestCase):
    """需要 embedding + chroma + 本组资料：四个场景按任务卡顺序真跑一遍。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from app.chroma_store import ChromaVectorStore, resolve_unit15_db_path
            from app.loaders import load_materials
            from app.tool_lab import (
                CALC_QUESTION,
                COMBO_QUESTION,
                EXPECTED_FAQ_ROWS,
                EXPECTED_KB_RECORDS,
                KNOWLEDGE_DIR,
                PROJECT_KEY,
                SINGLE_QUESTION,
                build_kit,
                render_run,
                run_class,
            )
        except ImportError as error:  # pragma: no cover - 依赖缺失
            raise unittest.SkipTest(f"缺少依赖：{error}")

        if not (KNOWLEDGE_DIR / "常见售后问题.csv").is_file():
            raise unittest.SkipTest("本组资料目录不在仓库中")

        cls._dir = Path(tempfile.mkdtemp(prefix="chroma-unit15-cls-"))
        cls._faq_dir = Path(tempfile.mkdtemp(prefix="sqlite-unit15-cls-"))
        # 别把这个字段叫 run：TestCase 自己有 run()，同名会让它变成不可调用的对象。
        cls.kit_run = run_class(database_path=cls._dir, faq_db_path=cls._faq_dir / "faq.db")
        cls.constants = {
            "kb": EXPECTED_KB_RECORDS,
            "faq": EXPECTED_FAQ_ROWS,
            "project": PROJECT_KEY,
        }
        cls.questions = {
            "single": SINGLE_QUESTION,
            "combo": COMBO_QUESTION,
            "calc": CALC_QUESTION,
        }
        cls.render_run = staticmethod(render_run)  # type: ignore[assignment]

    @classmethod
    def tearDownClass(cls) -> None:
        for folder in (getattr(cls, "_dir", None), getattr(cls, "_faq_dir", None)):
            if folder is not None:
                shutil.rmtree(folder, True)

    def test_knowledge_base_has_the_expected_records(self) -> None:
        self.assertEqual(self.kit_run.kb_records, self.constants["kb"])

    def test_faq_db_has_the_expected_rows(self) -> None:
        self.assertEqual(self.kit_run.faq_rows, self.constants["faq"])

    def test_all_four_scenarios_ran(self) -> None:
        keys = [scenario.key for scenario in self.kit_run.scenarios]
        self.assertEqual(keys, ["single", "combo", "calc", "reject"])

    def test_single_scenario_uses_faq(self) -> None:
        scenario = self.kit_run.scenario("single")
        assert scenario is not None
        self.assertTrue(scenario.finished)
        self.assertIn(TOOL_FAQ, scenario.result.tool_names)

    def test_combo_scenario_uses_two_tools(self) -> None:
        scenario = self.kit_run.scenario("combo")
        assert scenario is not None
        self.assertTrue(scenario.finished)
        self.assertEqual(len(scenario.result.distinct_tools), 2)

    def test_calc_scenario_reaches_deadline(self) -> None:
        scenario = self.kit_run.scenario("calc")
        assert scenario is not None
        self.assertTrue(scenario.finished)
        self.assertIn(TOOL_DEADLINE, scenario.result.tool_names)

    def test_reject_scenario_never_executes(self) -> None:
        scenario = self.kit_run.scenario("reject")
        assert scenario is not None
        self.assertEqual(len(scenario.results), 2)
        for result in scenario.results:
            self.assertTrue(result.rejected)
            self.assertEqual(result.tool_calls, 0)
            self.assertEqual(result.answer, "")

    def test_render_contains_the_required_sections(self) -> None:
        text = self.render_run(self.kit_run)
        self.assertIn("单元15", text)
        self.assertIn("单工具", text)
        self.assertIn("连续调用", text)
        self.assertIn("拒绝路径", text)
        self.assertIn("计算工具", text)
        self.assertIn(f"{self.constants['kb']} 条", text)
        self.assertIn(f"{self.constants['faq']} 行", text)


if __name__ == "__main__":
    unittest.main()
