"""单元13 验收测试：两类指标、SQLite 日志与评估闭环。

覆盖三件事：
1. **人工检索指标**（precision/recall）：纯规则、零模型，永远运行；含重复片段、
   路径归一化、空命中、缺标注等边界。
2. **监控落库**（LogStore）：全部跑在临时目录的 SQLite 上，测首页完就删；
   含 feedback 合法值与 ID 校验、summary 均值与反馈计数、评分失败行不进均值。
3. **评估闭环**（run_evaluation）：注入假检索与假模型跑完整流程，不调用任何真实 API；
   RAGAS 只测"发起请求之前的边界拦截"（缺密钥即报错），真实评审不做进测试。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.evaluation import normalize_source, retrieval_scores
from app.eval_lab import (
    EXPECTED_CASES,
    METRIC_EXPLANATIONS,
    read_summary,
    render_case,
    render_metrics_explanation,
    render_run,
    run_evaluation,
    set_feedback,
    validate_cases,
)
from app.monitoring import LogStore, render_rows, render_summary
from app.rag import RecordingLLM, RAGService
from app.retriever import SearchResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "e-commerce-after-sales"

MD_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/DOC-02-01-普通商品退货规则.md"
QUALITY_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/DOC-02-04-质量问题处理.md"
CSV_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/常见售后问题.csv"
PDF_SOURCE = "模拟数据/单元12项目资料/电商售后服务智能体项目组/售后受理材料清单.pdf"


def _ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


class TestRetrievalScores(unittest.TestCase):
    """精密/召回的规则逻辑：这也是任务卡要默写出来的那两个公式。"""

    def test_all_relevant(self) -> None:
        precision, recall = retrieval_scores(
            [MD_SOURCE, CSV_SOURCE], ["DOC-02-01-普通商品退货规则.md", "常见售后问题.csv"]
        )
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)

    def test_path_and_filename_are_equalized(self) -> None:
        """expected_sources 标注文件名，库里存相对路径——归一化后必须判为同一份。"""
        self.assertEqual(normalize_source(MD_SOURCE), "DOC-02-01-普通商品退货规则.md")
        self.assertEqual(normalize_source("DOC-02-01-普通商品退货规则.md"), "DOC-02-01-普通商品退货规则.md")

    def test_windows_separator(self) -> None:
        self.assertEqual(
            normalize_source("模拟数据\\knowledge\\DOC-02-05-定制商品例外.md"),
            "DOC-02-05-定制商品例外.md",
        )

    def test_duplicate_rows_do_not_inflate_recall(self) -> None:
        """同一份 CSV 的三行全命中：Precision 按条计（3/3），Recall 只算一次来源（1/1）。"""
        precision, recall = retrieval_scores(
            [CSV_SOURCE, CSV_SOURCE, CSV_SOURCE], ["常见售后问题.csv"]
        )
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)

    def test_partial_hit(self) -> None:
        """Top-3 里混进一份没标注的资料：Precision 掉一整份，Recall 不受影响。"""
        precision, recall = retrieval_scores(
            [CSV_SOURCE, MD_SOURCE, PDF_SOURCE],
            ["DOC-02-01-普通商品退货规则.md", "常见售后问题.csv"],
        )
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertEqual(recall, 1.0)

    def test_nothing_retrieved(self) -> None:
        """一条都没命中时 Precision 为 0；Recall 为 0（标注的相关来源一个都没找回）。"""
        precision, recall = retrieval_scores([], ["DOC-02-01-普通商品退货规则.md"])
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)

    def test_missing_direct_evidence(self) -> None:
        """Recall@3 低 = 直接证据没进 Top-3，这正是要分析的"低分样例"形态。"""
        precision, recall = retrieval_scores(
            [PDF_SOURCE, PDF_SOURCE], ["DOC-02-01-普通商品退货规则.md", "常见售后问题.csv"]
        )
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)

    def test_empty_expected_raises(self) -> None:
        with self.assertRaises(ValueError):
            retrieval_scores([MD_SOURCE], [])


class TestValidateCases(unittest.TestCase):
    def test_too_few_cases(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少"):
            validate_cases([{"id": "A", "question": "q", "expected_sources": ["a.md"], "reference_answer": "r"}])

    def test_missing_field_reports_index(self) -> None:
        cases = [
            {"id": "A", "question": "q", "expected_sources": ["a.md"], "reference_answer": "r"},
            {"id": "B", "question": "q", "expected_sources": [], "reference_answer": "r"},
            {"id": "C", "question": "", "expected_sources": ["a.md"], "reference_answer": "r"},
        ]
        with self.assertRaisesRegex(ValueError, "第 2 道题缺少字段"):
            validate_cases(cases)

    def test_valid_cases_pass(self) -> None:
        cases = [
            {
                "id": f"Q{i}",
                "question": f"问题{i}",
                "expected_sources": ["DOC-02-01-普通商品退货规则.md"],
                "reference_answer": f"参考答案{i}",
            }
            for i in range(EXPECTED_CASES)
        ]
        self.assertIsNone(validate_cases(cases))

    def test_shipped_cases_are_valid(self) -> None:
        """本组提交的 evaluation_cases.json 必须通过校验，且 expected_sources 非空。"""
        cases = json.loads((PROJECT_ROOT / "evaluation_cases.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), EXPECTED_CASES)
        validate_cases(cases)
        for case in cases:
            self.assertTrue(case["expected_sources"])
            self.assertTrue(case["reference_answer"].strip())


class TestLogStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "rag_monitor.sqlite3"
        self.logs = LogStore(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _record(self, precision: float = 0.67, ragas: dict | None = None) -> int:
        return self.logs.record(
            "EC-RAG-01", "商品在签收后五天可以退货吗？",
            [{"source": MD_SOURCE, "source_name": "DOC-02-01-普通商品退货规则.md", "distance": 0.5, "text": "…"}],
            "参考回答……[1]", precision, 1.0,
            ragas if ragas is not None else {"faithfulness": 1.0, "context_precision": 0.8, "context_recall": 1.0},
            3.21,
        )

    def test_record_returns_increasing_ids(self) -> None:
        first = self._record()
        second = self._record()
        self.assertEqual(second, first + 1)

    def test_rows_keep_evidence(self) -> None:
        log_id = self._record()
        row = self.logs.rows(1)[0]
        self.assertEqual(row["id"], log_id)
        self.assertEqual(row["case_id"], "EC-RAG-01")
        self.assertEqual(json.loads(row["retrieved_json"])[0]["source"], MD_SOURCE)
        self.assertAlmostEqual(row["latency_seconds"], 3.21)

    def test_feedback_ok(self) -> None:
        log_id = self._record()
        self.logs.set_feedback(log_id, "bad")
        row = self.logs.rows(1)[0]
        self.assertEqual(row["user_feedback"], "bad")

    def test_feedback_rejects_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "good"):
            self.logs.set_feedback(1, "maybe")

    def test_feedback_requires_existing_id(self) -> None:
        with self.assertRaisesRegex(KeyError, "日志ID不存在"):
            self.logs.set_feedback(999, "good")

    def test_summary_averages_and_feedback(self) -> None:
        self._record(precision=1.0)
        self._record(precision=0.0)
        self.logs.set_feedback(1, "good")
        self.logs.set_feedback(2, "bad")
        summary = self.logs.summary()
        self.assertEqual(summary["questions"], 2)
        self.assertAlmostEqual(summary["avg_precision_at_3"], 0.5)
        self.assertAlmostEqual(summary["avg_recall_at_3"], 1.0)
        self.assertAlmostEqual(summary["avg_faithfulness"], 1.0)
        self.assertAlmostEqual(summary["avg_latency_seconds"], 3.21)
        self.assertEqual(summary["feedback"], {"good": 1, "bad": 1})

    def test_summary_skips_failed_scoring_rows(self) -> None:
        """评审失败的行如实记 error，绝不混进均值假装是低分。"""
        self._record()  # faithfulness=1.0
        self._record(ragas={"error": "RuntimeError: 未配置密钥"})
        summary = self.logs.summary()
        self.assertEqual(summary["questions"], 2)
        self.assertEqual(summary["avg_faithfulness"], 1.0, "失败行应被剔除，而不是按 0 分拉低均值")

    def test_summary_none_when_all_scoring_failed(self) -> None:
        self._record(ragas={"error": "RuntimeError: 未配置密钥"})
        self._record(ragas={"skipped": "没有检索命中，跳过 RAGAS 评分"})
        summary = self.logs.summary()
        self.assertIsNone(summary["avg_faithfulness"])
        self.assertIsNone(summary["avg_context_precision"])

    def test_summary_on_empty_db(self) -> None:
        summary = self.logs.summary()
        self.assertEqual(summary["questions"], 0)
        self.assertIsNone(summary["avg_precision_at_3"])
        self.assertEqual(summary["feedback"], {})


class FakeLLM:
    """不联网的假模型：回答里带上合法引用编号，够测完整闭环。"""

    def __init__(self, replies: list[str] | None = None):
        self.replies = replies or []
        self.calls = 0

    def chat(self, messages: list[dict]) -> str:
        self.calls += 1
        return self.replies[self.calls - 1] if self.calls <= len(self.replies) else "依据资料作答……[1]"


class FakeRetriever:
    """固定返回三道题各自的 Top-3，命中来源与实际库一致。"""

    def __init__(self, batches: dict[str, list[SearchResult]]):
        self.batches = batches
        self.projects: list[str] = []

    def search(self, question: str, project_id: str = PROJECT_ID, top_k: int = 3,
               source: str | None = None, doc_type: str | None = None) -> list[SearchResult]:
        self.projects.append(project_id)
        return self.batches.get(question, [])


def _hit(source: str, distance: float) -> SearchResult:
    return SearchResult(
        document_id=f"{PROJECT_ID}:{Path(source).name}:row:2",
        text="退货规则原文……",
        source=source,
        project_id=PROJECT_ID,
        doc_type=Path(source).suffix.lstrip("."),
        distance=distance,
    )


class TestEvalLoop(unittest.TestCase):
    """闭环用假检索 + 假模型跑，一行真实 API 都不调用。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.log_db = Path(self._tmp.name) / "monitor.sqlite3"
        self.cases_path = Path(self._tmp.name) / "cases.json"
        cases = [
            {
                "id": "EC-RAG-01",
                "question": "普通商品签收后五天且未使用能退货吗？",
                "expected_sources": ["DOC-02-01-普通商品退货规则.md", "常见售后问题.csv"],
                "reference_answer": "七天无理由需人工复核。",
            },
            {
                "id": "EC-RAG-02",
                "question": "超过七天时限的质量问题怎么办？",
                "expected_sources": ["DOC-02-04-质量问题处理.md", "常见售后问题.csv"],
                "reference_answer": "转人工复核。",
            },
            {
                "id": "EC-RAG-03",
                "question": "物流长期不更新能承诺到达日期吗？",
                "expected_sources": ["DOC-02-07-物流异常说明.md", "常见售后问题.csv"],
                "reference_answer": "不能虚构到达日期。",
            },
        ]
        self.cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _service(self) -> RAGService:
        batches = {
            "普通商品签收后五天且未使用能退货吗？": [
                _hit(CSV_SOURCE, 0.46), _hit(MD_SOURCE, 0.61), _hit(PDF_SOURCE, 0.93)
            ],
            "超过七天时限的质量问题怎么办？": [
                _hit(QUALITY_SOURCE, 0.40), _hit(CSV_SOURCE, 0.55)
            ],
            "物流长期不更新能承诺到达日期吗？": [_hit(PDF_SOURCE, 0.7)],
        }
        llm = FakeLLM([
            "签收七天内且未使用可申请退货评估，是否通过由人工复核……[1][2]",
            "质量问题即使超过七天也应转人工复核……[1][2]",
            "应说明当前物流状态并建议人工核实，不能虚构到达日期……[1]",
        ])
        return RAGService(FakeRetriever(batches), RecordingLLM(llm), top_k=3)

    def test_three_cases_are_logged(self) -> None:
        run = run_evaluation(EXPECTED_CASES, service=self._service(),
                             cases_path=self.cases_path, log_path=self.log_db, with_ragas=False)
        self.assertEqual(run.run_cases, 3)
        self.assertEqual([item.log_id for item in run.results], [1, 2, 3])
        summary = read_summary(self.log_db)
        self.assertEqual(summary["questions"], 3)
        self.assertIsNone(summary["avg_faithfulness"], "未调用评审时不应出现 RAGAS 均值")
        for item in run.results:
            self.assertEqual(item.ragas["skipped"], "本次未调用评审模型")

    def test_precision_and_recall_from_expected_sources(self) -> None:
        run = run_evaluation(3, service=self._service(), cases_path=self.cases_path,
                             log_path=self.log_db, with_ragas=False)
        first, second, third = run.results
        # 第1题：三条命中里两条是标注来源；两份标注来源都找回 → P=2/3, R=1
        self.assertAlmostEqual(first.precision, 2 / 3)
        self.assertEqual(first.recall, 1.0)
        # 第2题：两条都命中 → 满分
        self.assertEqual(second.precision, 1.0)
        self.assertEqual(second.recall, 1.0)
        # 第3题：命中一份没标注的资料 → 两个指标都掉到 0，这就是要分析的低分样例
        self.assertEqual(third.precision, 0.0)
        self.assertEqual(third.recall, 0.0)
        self.assertTrue(third.hits, "低分样例仍应保留证据供复盘")

    def test_limit_one_runs_only_first_case(self) -> None:
        run = run_evaluation(1, service=self._service(), cases_path=self.cases_path,
                             log_path=self.log_db, with_ragas=False)
        self.assertEqual(run.run_cases, 1)
        self.assertEqual(run.total_cases, 3)

    def test_limit_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "大于 0"):
            run_evaluation(0, service=self._service(), cases_path=self.cases_path,
                           log_path=self.log_db, with_ragas=False)

    def test_answer_records_cited_sources(self) -> None:
        run = run_evaluation(1, service=self._service(), cases_path=self.cases_path,
                             log_path=self.log_db, with_ragas=False)
        self.assertEqual(
            run.results[0].sources, [CSV_SOURCE, MD_SOURCE], "引用 [1][2] 应换成两个真实来源"
        )

    def test_render_includes_both_metric_kinds(self) -> None:
        run = run_evaluation(1, service=self._service(), cases_path=self.cases_path,
                             log_path=self.log_db, with_ragas=False)
        text = render_run(run)
        self.assertIn("Precision@3", text)
        self.assertIn("Recall@3", text)
        self.assertIn("日志ID", text)
        for name, *_rest in METRIC_EXPLANATIONS:
            self.assertIn(name, render_metrics_explanation())
        self.assertIn("EC-RAG-01", render_case(run.results[0]))

    def test_render_rows_and_summary_text(self) -> None:
        run = run_evaluation(2, service=self._service(), cases_path=self.cases_path,
                             log_path=self.log_db, with_ragas=False)
        rows_text = render_rows(LogStore(self.log_db).rows(2))
        self.assertIn("日志ID=2", rows_text)
        self.assertIn("常见售后问题.csv", rows_text)
        self.assertIn("平均 Precision@3", render_summary(read_summary(self.log_db)))


class TestRagasBoundary(unittest.TestCase):
    """RAGAS 评审要联网花钱，测试只覆盖"发起请求之前的拦截"。"""

    @unittest.skipUnless(_ragas_available(), "需要 ragas：uv sync --extra evaluation")
    def test_missing_key_is_rejected_before_request(self) -> None:
        import os
        from contextlib import contextmanager

        @contextmanager
        def without_key():
            saved = os.environ.pop("DEEPSEEK_API_KEY", None)
            saved_provider = os.environ.pop("DEEPSEEK_MODEL", None)
            try:
                yield
            finally:
                if saved:
                    os.environ["DEEPSEEK_API_KEY"] = saved
                if saved_provider:
                    os.environ["DEEPSEEK_MODEL"] = saved_provider

        from app.evaluation import ragas_scores

        with without_key(), self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
            ragas_scores("问题", "答案", ["上下文"], "参考答案")


class TestSetFeedbackHelper(unittest.TestCase):
    def test_helper_writes_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "monitor.sqlite3"
            logs = LogStore(db)
            log_id = logs.record("EC-RAG-01", "问题", [], "答案", 1.0, 1.0, {}, 1.0)
            self.assertEqual(set_feedback(log_id, "good", db), log_id)
            self.assertEqual(logs.rows(1)[0]["user_feedback"], "good")


if __name__ == "__main__":
    unittest.main()
