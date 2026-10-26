"""单元14 验收测试：ReAct 循环的边界、停止状态与轨迹渲染。

覆盖三件事，均不调用真实 API：
1. **循环本身**：注入假检索 + 脚本决策模型，覆盖单事项、双事项、轮次耗尽三种停止形态。
2. **执行边界**：动作白名单、来源必须真实观察、无观察时只能答"资料不足"、
   参数校验、JSON 解析——这些是"模型只能提议、Python 才能执行"的证据。
3. **编排与渲染**：`run_scenario` 的场景封装与两段输出是否含课堂要交的字段。

需要 embedding + chroma 的真实入库用例单独放在最后那个类里，缺依赖就跳过。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.agent import (
    ACTIONS,
    DEFAULT_MAX_STEPS,
    FINISHED,
    FINISH_TOOL,
    INSUFFICIENT_MARK,
    MIN_MAX_STEPS,
    ScriptedDecisionModel,
    SEARCH_TOOL,
    STOPPED_BY_LIMIT,
    build_decision_model,
    list_actions,
    render_result,
    render_trace,
    ReActAgent,
    split_items,
)
from app.agent_lab import (
    EXPECTED_KB_RECORDS,
    LIMITED_MAX_STEPS,
    SINGLE_QUESTION,
    WHY_PYTHON_EXECUTES,
    decision_model_name,
    render_comparison,
    render_scenario,
    render_why_python,
    run_scenario,
)
from app.retriever import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "e-commerce-after-sales"

MD_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/DOC-02-01-普通商品退货规则.md"
QUALITY_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/DOC-02-04-质量问题处理.md"
PDF_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/售后受理材料清单.pdf"
CSV_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/常见售后问题.csv"

SINGLE_QUERY = "普通商品签收后还能申请退货评估吗？"
DOUBLE_QUERY = "普通商品签收后多久还能申请退货评估？另外商品有质量问题需要准备哪些材料？"


def _hit(source: str, distance: float = 0.5) -> SearchResult:
    return SearchResult(
        document_id=f"{PROJECT_ID}:{Path(source).name}:row:2",
        text="签收后7天内且未使用可申请退货评估，最终由人工复核。",
        source=source,
        project_id=PROJECT_ID,
        doc_type=Path(source).suffix.lstrip("."),
        distance=distance,
    )


class FakeRetriever:
    """按关键词返回固定证据，并记录每一次调用的 query 与 project_id。"""

    def __init__(self, batches: dict[str, list[SearchResult]] | None = None):
        self.batches = batches or {
            "退货时限": [_hit(CSV_SOURCE, 0.42), _hit(MD_SOURCE, 0.61), _hit(PDF_SOURCE, 0.9)],
            "质量材料": [_hit(CSV_SOURCE, 0.44), _hit(PDF_SOURCE, 0.5), _hit(QUALITY_SOURCE, 0.66)],
        }
        self.queries: list[str] = []
        self.projects: list[str] = []

    def _match(self, query: str) -> list[SearchResult]:
        if "质量" in query or "材料" in query:
            return self.batches["质量材料"]
        return self.batches["退货时限"]

    def search(self, question: str, project_id: str = PROJECT_ID, top_k: int = 3,
               source: str | None = None, doc_type: str | None = None) -> list[SearchResult]:
        self.queries.append(question)
        self.projects.append(project_id)
        return self._match(question)


class EmptyRetriever(FakeRetriever):
    """什么都不命中：用来验证"没观察到资料时只能答资料不足"。"""

    def search(self, question: str, project_id: str = PROJECT_ID, top_k: int = 3,
               source: str | None = None, doc_type: str | None = None) -> list[SearchResult]:
        self.queries.append(question)
        self.projects.append(project_id)
        return []


class RecordingDecisionModel:
    """按给定剧本逐轮返回动作，同时记下它收到的对话历史。"""

    def __init__(self, script: list[dict]):
        self.script = script
        self.seen_messages: list[list[dict]] = []
        self.calls: int = 0

    def decide(self, messages: list[dict]) -> str:
        self.calls += 1
        import json

        self.seen_messages.append(list(messages))
        index = min(self.calls - 1, len(self.script) - 1)
        return json.dumps(self.script[index], ensure_ascii=False)


def make_agent(retriever: FakeRetriever | None = None, scripted: bool = True):
    retriever = retriever or FakeRetriever()
    model = ScriptedDecisionModel() if scripted else None
    return ReActAgent(model, retriever, PROJECT_ID), retriever, model


class TestItemSplit(unittest.TestCase):
    """离线脚本策略的分句依据；顺便也是演示题是否符合"两个事项"的自检。"""

    def test_single_item(self) -> None:
        self.assertEqual(len(split_items(SINGLE_QUERY)), 1)

    def test_double_items(self) -> None:
        self.assertEqual(len(split_items(DOUBLE_QUERY)), 2)

    def test_punctuation_is_trimmed(self) -> None:
        items = split_items(DOUBLE_QUERY)
        self.assertFalse(any(item.endswith("？") for item in items))
        self.assertTrue(items[1].startswith("商品"))

    def test_empty_question(self) -> None:
        self.assertEqual(split_items("   "), [])

    def test_limit_is_respected(self) -> None:
        long_one = "退货时限？另外材料清单？还有发票问题？另外地址修改？"
        self.assertEqual(len(split_items(long_one, limit=2)), 2)


class TestAgentBoundaries(unittest.TestCase):
    """构造与入参校验：越早报错，课堂演示越不会卡在半路。"""

    def test_project_id_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "project_id"):
            ReActAgent(ScriptedDecisionModel(), FakeRetriever(), "   ")

    def test_question_must_not_be_empty(self) -> None:
        agent, _, _ = make_agent()
        with self.assertRaisesRegex(ValueError, "问题不能为空"):
            agent.run("   ")

    def test_max_steps_needs_two_at_least(self) -> None:
        agent, _, _ = make_agent()
        with self.assertRaisesRegex(ValueError, "至少"):
            agent.run(SINGLE_QUERY, MIN_MAX_STEPS - 1)

    def test_max_steps_must_be_integer(self) -> None:
        agent, _, _ = make_agent()
        with self.assertRaisesRegex(ValueError, "整数"):
            agent.run(SINGLE_QUERY, "3")  # type: ignore[arg-type]

    def test_actions_are_the_whitelist(self) -> None:
        self.assertEqual(list_actions(), [SEARCH_TOOL, FINISH_TOOL])
        self.assertEqual(tuple(list_actions()), ACTIONS)


class TestReactLoop(unittest.TestCase):
    """三种停止形态：一次检索就答 / 两轮检索再答 / 轮次耗尽未答。"""

    def test_single_item_finishes_with_one_search(self) -> None:
        agent, retriever, _ = make_agent()
        result = agent.run(SINGLE_QUERY)
        self.assertTrue(result.finished)
        self.assertEqual(result.stop_reason, FINISHED)
        self.assertEqual(result.search_count, 1)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.decision_calls, 2)
        self.assertTrue(result.answer)
        self.assertEqual(retriever.projects, [PROJECT_ID])

    def test_reported_sources_must_be_observed(self) -> None:
        agent, _, _ = make_agent()
        result = agent.run(SINGLE_QUERY)
        for source in result.sources:
            self.assertIn(source, result.observed_sources)

    def test_double_item_triggers_second_search(self) -> None:
        agent, retriever, _ = make_agent()
        result = agent.run(DOUBLE_QUERY)
        self.assertEqual(result.search_count, 2, "双事项问题应出现两轮检索")
        self.assertEqual(result.steps, 3)
        self.assertTrue(result.finished)
        self.assertNotEqual(retriever.queries[0], retriever.queries[1],
                            "第二轮的 query 应与第一轮不同")
        names = [Path(item).name for item in result.observed_sources]
        self.assertIn("DOC-02-04-质量问题处理.md", names)

    def test_max_steps_stops_without_answer(self) -> None:
        """完成标准第三条：限制轮次时必须能识别未完成，而不把半途结果当答案。"""
        agent, _, _ = make_agent()
        result = agent.run(DOUBLE_QUERY, LIMITED_MAX_STEPS)
        self.assertFalse(result.finished)
        self.assertTrue(result.limited)
        self.assertEqual(result.stop_reason, STOPPED_BY_LIMIT)
        self.assertEqual(result.answer, "")
        self.assertEqual(result.sources, [])
        self.assertEqual(result.decision_calls, LIMITED_MAX_STEPS,
                         "轮次耗尽后不允许再偷偷多问一次模型")
        self.assertEqual(result.search_count, 2)

    def test_no_evidence_only_allows_insufficient_answer(self) -> None:
        agent, _, _ = make_agent(EmptyRetriever())
        result = agent.run(SINGLE_QUERY)
        self.assertTrue(result.finished)
        self.assertEqual(result.sources, [])
        self.assertIn(INSUFFICIENT_MARK, result.answer)

    def test_observation_is_fed_back_to_the_model(self) -> None:
        """第二轮决策时必须能看到第一轮的工具观察，否则不叫循环。"""
        script = [
            {"reason": "先查退货时限", "action": SEARCH_TOOL, "arguments": {"query": "退货时限"}},
            {"reason": "依据观察作答", "action": FINISH_TOOL,
             "answer": "7天内未使用可申请退货评估[1]", "sources": [MD_SOURCE]},
        ]
        model = RecordingDecisionModel(script)
        agent = ReActAgent(model, FakeRetriever(), PROJECT_ID)
        result = agent.run(SINGLE_QUERY)
        self.assertTrue(result.finished)
        self.assertEqual(model.calls, 2)
        second_round = model.seen_messages[1]
        self.assertTrue(any("工具观察：" in str(item.get("content", ""))
                            for item in second_round),
                        "第二轮发给模型的消息里必须带上第一轮的工具观察")


class TestPythonExecutionBoundaries(unittest.TestCase):
    """模型只提议、Python 才执行：这些校验全部发生在本地代码里。"""

    def test_unknown_action_is_rejected(self) -> None:
        script = [{"reason": "想直接改库", "action": "delete_everything", "arguments": {}}]
        agent = ReActAgent(RecordingDecisionModel(script), FakeRetriever(), PROJECT_ID)
        with self.assertRaisesRegex(ValueError, "未知动作"):
            agent.run(SINGLE_QUERY)

    def test_missing_arguments_object(self) -> None:
        script = [{"reason": "没给参数", "action": SEARCH_TOOL, "arguments": "退货"}]
        agent = ReActAgent(RecordingDecisionModel(script), FakeRetriever(), PROJECT_ID)
        with self.assertRaisesRegex(ValueError, "arguments 必须是对象"):
            agent.run(SINGLE_QUERY)

    def test_empty_query_is_rejected(self) -> None:
        script = [{"reason": "空查询", "action": SEARCH_TOOL, "arguments": {"query": "  "}}]
        agent = ReActAgent(RecordingDecisionModel(script), FakeRetriever(), PROJECT_ID)
        with self.assertRaisesRegex(ValueError, "query 不能为空"):
            agent.run(SINGLE_QUERY)

    def test_invalid_json_is_rejected(self) -> None:
        class BadJSON:
            calls = 0

            def decide(self, messages: list[dict]) -> str:
                self.calls += 1
                return "好的，我先查一下。<不是 JSON>"

        agent = ReActAgent(BadJSON(), FakeRetriever(), PROJECT_ID)
        with self.assertRaisesRegex(ValueError, "不是合法\\s*JSON|未返回有效 JSON"):
            agent.run(SINGLE_QUERY)

    def test_json_must_be_object(self) -> None:
        class ArrayList:
            def decide(self, messages: list[dict]) -> str:
                return '["search_knowledge"]'

        agent = ReActAgent(ArrayList(), FakeRetriever(), PROJECT_ID)
        with self.assertRaisesRegex(ValueError, "必须是 JSON 对象"):
            agent.run(SINGLE_QUERY)

    def test_finish_without_answer_is_rejected(self) -> None:
        script = [{"reason": "想蒙过去", "action": FINISH_TOOL, "answer": "  ", "sources": []}]
        agent = ReActAgent(RecordingDecisionModel(script), FakeRetriever(), PROJECT_ID)
        with self.assertRaisesRegex(ValueError, "finish 缺少回答"):
            agent.run(SINGLE_QUERY)

    def test_sources_must_come_from_observations(self) -> None:
        script = [
            {"reason": "先查", "action": SEARCH_TOOL, "arguments": {"query": "退货时限"}},
            {"reason": "编一个来源", "action": FINISH_TOOL,
             "answer": "依据 DOC-02-99 可以退货", "sources": ["DOC-02-99-不存在的资料.md"]},
        ]
        agent = ReActAgent(RecordingDecisionModel(script), FakeRetriever(), PROJECT_ID)
        with self.assertRaisesRegex(ValueError, "未检索到的来源"):
            agent.run(SINGLE_QUERY)

    def test_filename_is_accepted_as_source(self) -> None:
        """模型写文件名也算数——口径与单元13 的 normalize_source 一致。"""
        script = [
            {"reason": "先查", "action": SEARCH_TOOL, "arguments": {"query": "退货时限"}},
            {"reason": "引用真实来源", "action": FINISH_TOOL,
             "answer": "7天内未使用可申请退货评估", "sources": ["DOC-02-01-普通商品退货规则.md"]},
        ]
        agent = ReActAgent(RecordingDecisionModel(script), FakeRetriever(), PROJECT_ID)
        result = agent.run(SINGLE_QUERY)
        self.assertEqual([Path(item).name for item in result.sources],
                         ["DOC-02-01-普通商品退货规则.md"])

    def test_sources_must_be_string_list(self) -> None:
        script = [
            {"reason": "先查", "action": SEARCH_TOOL, "arguments": {"query": "退货时限"}},
            {"reason": "来源写错类型", "action": FINISH_TOOL,
             "answer": "可以退货", "sources": [123]},
        ]
        agent = ReActAgent(RecordingDecisionModel(script), FakeRetriever(), PROJECT_ID)
        with self.assertRaisesRegex(ValueError, "文件名列表"):
            agent.run(SINGLE_QUERY)

    def test_no_observation_requires_insufficient_word(self) -> None:
        script = [{"reason": "凭常识作答", "action": FINISH_TOOL,
                   "answer": "一般签收七天都可以退", "sources": []}]
        agent = ReActAgent(RecordingDecisionModel(script), FakeRetriever(), PROJECT_ID)
        with self.assertRaisesRegex(ValueError, INSUFFICIENT_MARK):
            agent.run(SINGLE_QUERY)


class TestDecisionModelFactory(unittest.TestCase):
    def test_mock_provider_returns_scripted_model(self) -> None:
        self.assertIsInstance(build_decision_model("mock"), ScriptedDecisionModel)

    def test_unknown_provider_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
            build_decision_model("gpt")

    def test_scripted_model_name_is_explicit(self) -> None:
        self.assertIn("离线", decision_model_name(ScriptedDecisionModel()))

    def test_default_follows_environment(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"LLM_PROVIDER": "mock"}):
            self.assertIsInstance(build_decision_model(), ScriptedDecisionModel)

    def test_deepseek_provider_needs_key(self) -> None:
        """没有密钥要立刻报错，绝不能静默降级成本地脚本策略。"""
        import os
        from unittest import mock

        try:
            import openai  # noqa: F401
        except ImportError:
            self.skipTest("需要 openai：uv sync --extra deepseek")

        saved_provider = os.environ.pop("LLM_PROVIDER", None)
        try:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DEEPSEEK_API_KEY", None)
                with self.assertRaises(RuntimeError):
                    build_decision_model("deepseek")
        finally:
            if saved_provider:
                os.environ["LLM_PROVIDER"] = saved_provider


class TestLabScenarios(unittest.TestCase):
    """编排层：场景封装 + 两段渲染里必须出现任务卡要求的字段。"""

    def _scenario(self, question: str, max_steps: int, key: str):
        agent = ReActAgent(ScriptedDecisionModel(), FakeRetriever(), PROJECT_ID)
        return run_scenario(key, question, max_steps, agent=agent)[0]

    def test_scenario_wraps_result(self) -> None:
        scenario = self._scenario(SINGLE_QUESTION, DEFAULT_MAX_STEPS, "single")
        self.assertTrue(scenario.finished)
        self.assertEqual(scenario.search_count, 1)
        self.assertEqual(scenario.stop_reason, FINISHED)

    def test_render_scenario_includes_trace_and_result(self) -> None:
        text = render_scenario(self._scenario(DOUBLE_QUERY, DEFAULT_MAX_STEPS, "double"))
        self.assertIn("停止原因", text)
        self.assertIn("轨迹", text)
        self.assertIn("search_knowledge", text)
        self.assertIn("DOC-02-04-质量问题处理.md", text)

    def test_render_comparison_marks_unfinished(self) -> None:
        class FakeRun:
            def __init__(self, scenarios):
                self.scenarios = scenarios

            def scenario(self, key):
                for item in self.scenarios:
                    if item.key == key:
                        return item
                return None

        full = self._scenario(DOUBLE_QUERY, DEFAULT_MAX_STEPS, "double")
        limited = self._scenario(DOUBLE_QUERY, LIMITED_MAX_STEPS, "limited")
        text = render_comparison(FakeRun([full, limited]))
        self.assertIn("双事项问题", text)
        self.assertIn("轮次受限重跑", text)
        self.assertIn(FINISHED, text)
        self.assertIn(STOPPED_BY_LIMIT, text)
        self.assertIn("否（未完成，不予采纳）", text)

    def test_render_why_python_lists_reasons(self) -> None:
        text = render_why_python()
        self.assertIn(f"共 {len(WHY_PYTHON_EXECUTES)} 条：", text)
        for title, _detail in WHY_PYTHON_EXECUTES:
            self.assertIn(title, text)
            self.assertTrue(_detail)

    def test_render_result_and_trace(self) -> None:
        result = self._scenario(SINGLE_QUERY, DEFAULT_MAX_STEPS, "single").result
        self.assertIn("轮次上限", render_result(result))
        trace = render_trace(result)
        self.assertIn(f"1\t", trace)
        self.assertIn(SEARCH_TOOL, trace)
        self.assertIn(FINISH_TOOL, trace)

    def test_limited_render_prints_the_warning(self) -> None:
        result = self._scenario(DOUBLE_QUERY, LIMITED_MAX_STEPS, "limited").result
        self.assertIn("轮次耗尽", render_trace(result))
        self.assertIn("（本轮未产出答案）", render_result(result))


def _chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_chroma_available(), "未安装 chromadb / fastembed，跳过真实入库用例")
class ChromaIntegrationTest(unittest.TestCase):
    """真实链路：本组 7 条资料进 .chroma_unit14 后，Agent 能查到并引用真实来源。"""

    @classmethod
    def setUpClass(cls) -> None:
        from app.agent_lab import build_agent

        cls._dir = Path(tempfile.mkdtemp(prefix="chroma-unit14-cls-"))
        cls.agent, cls.store, cls.count = build_agent(DEFAULT_MAX_STEPS, cls._dir,
                                                      ScriptedDecisionModel())

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._dir, ignore_errors=True)

    def test_knowledge_base_has_expected_records(self) -> None:
        self.assertEqual(self.count, EXPECTED_KB_RECORDS)

    def test_real_retrieval_feeds_finish(self) -> None:
        result = self.agent.run(SINGLE_QUESTION)
        self.assertTrue(result.finished)
        self.assertEqual(result.search_count, 1)
        self.assertTrue(result.sources, "命中真实资料后应当给出来源")
        for source in result.sources:
            self.assertIn(source, result.observed_sources)

    def test_limited_run_on_real_store(self) -> None:
        from app.agent_lab import DOUBLE_QUESTION

        result = self.agent.run(DOUBLE_QUESTION, LIMITED_MAX_STEPS)
        self.assertTrue(result.limited)
        self.assertEqual(result.search_count, 2)


if __name__ == "__main__":
    unittest.main()
