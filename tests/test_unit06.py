"""单元06 验收测试：真实样例文本、向量一一对应、维度一致、边界与误判风险。

不依赖模型的用例永远会跑；需要本地模型的用例在模型缺失时跳过并给出提示，
这样没有拷贝 models/ 的机器也能跑其余单元测试。
"""

from __future__ import annotations

import unittest

from app.embedding import (
    EXPECTED_DIM,
    MODEL_NAME,
    cosine_similarity,
    embed_many,
    model_ready,
    resolve_model_dir,
)
from app.vector_lab import (
    DEMO_QUERY,
    MIN_TEXTS,
    RISK_QUERY,
    boundary_report,
    load_sample_texts,
    misjudge_check,
    run_lab,
)

MODEL_AVAILABLE, MODEL_DETAIL = model_ready()


def require_model(test_case: unittest.TestCase) -> None:
    if not MODEL_AVAILABLE:
        test_case.skipTest(f"本地向量模型不可用：{MODEL_DETAIL}")


class TestUnit06SampleTexts(unittest.TestCase):
    """第2节：本组真实样例文本清单（电商售后场景，带编号与来源）。"""

    def test_at_least_five_real_texts_with_id_and_source(self):
        items = load_sample_texts()
        self.assertGreaterEqual(len(items), MIN_TEXTS)
        for item in items:
            self.assertTrue(item.source_id)
            self.assertTrue(item.text.strip())
            self.assertIn("sample_questions.json", item.source)

    def test_texts_belong_to_ecommerce_after_sale_scene(self):
        """文本必须来自本项目电商售后数据，而不是教师演示的工单/报销场景。"""
        items = load_sample_texts()
        joined = " ".join(item.text for item in items)
        for keyword in ("SO-", "退货", "退款", "商品"):
            self.assertIn(keyword, joined)

    def test_missing_fields_are_rejected(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "sample_questions.json"
            bad.write_text(json.dumps([{"question": "缺少编号"}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_sample_texts(bad)

    def test_too_few_texts_are_rejected(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            small = Path(tmp) / "sample_questions.json"
            payload = [{"question_id": f"Q-{i}", "question": f"问题{i}"} for i in range(3)]
            small.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_sample_texts(small)


class TestUnit06CosineBoundaries(unittest.TestCase):
    """第4节：相似度计算本身的边界。"""

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 0.0])

    def test_empty_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([], [])

    def test_zero_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])

    def test_identical_vector_scores_one(self):
        self.assertAlmostEqual(cosine_similarity([0.6, 0.8], [0.6, 0.8]), 1.0, places=6)

    def test_opposite_vector_scores_minus_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, places=6)


class TestUnit06EmbeddingBoundaries(unittest.TestCase):
    """第4节：向量化入口的边界（空列表、空白文本）。"""

    def test_empty_list_is_rejected(self):
        with self.assertRaises(ValueError):
            embed_many([])

    def test_blank_text_is_rejected_before_model_runs(self):
        with self.assertRaises(ValueError):
            embed_many(["正常问题", "   "])

    def test_non_string_text_is_rejected(self):
        with self.assertRaises(ValueError):
            embed_many(["正常问题", 123])

    def test_boundary_report_records_every_case(self):
        report = boundary_report()
        names = {check["name"] for check in report["checks"]}
        self.assertEqual(
            names,
            {"空列表输入", "含空白文本的一批", "维度不一致求相似度", "零向量求相似度", "同一向量自比"},
        )
        # 前四项必须被拦截，最后一项必须算出 1.0。
        for check in report["checks"][:4]:
            self.assertFalse(check["ok"], check["name"])
        self.assertTrue(report["checks"][-1]["ok"])
        self.assertIn("1.0", report["checks"][-1]["detail"])


class TestUnit06RealEmbedding(unittest.TestCase):
    """第3节：真实本地模型跑通批量向量化与排序。"""

    def setUp(self) -> None:
        require_model(self)

    def test_model_is_bge_small_zh(self):
        self.assertEqual(MODEL_NAME, "BAAI/bge-small-zh-v1.5")

    def test_vectors_match_texts_one_to_one_with_same_dim(self):
        items = load_sample_texts()
        texts = [item.text for item in items]
        vectors = embed_many(texts)

        self.assertEqual(len(vectors), len(texts))
        dims = {len(vector) for vector in vectors}
        self.assertEqual(len(dims), 1)
        self.assertEqual(dims.pop(), EXPECTED_DIM)

    def test_query_and_candidates_share_one_vector_space(self):
        report = run_lab(DEMO_QUERY)
        # 查询句占 1 条，其余每条样例文本各占 1 条。
        self.assertEqual(report["vector_count"], report["text_count"] + 1)
        self.assertEqual(report["dim"], EXPECTED_DIM)
        self.assertTrue(report["one_to_one"])
        self.assertEqual(len(report["ranked"]), report["text_count"])

    def test_ranking_is_sorted_descending(self):
        report = run_lab(DEMO_QUERY)
        scores = [row["score"] for row in report["ranked"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], scores[-1])

    def test_top_three_are_returned(self):
        report = run_lab(DEMO_QUERY)
        self.assertEqual(len(report["top"]), 3)
        self.assertEqual(report["top"][0]["source_id"], report["ranked"][0]["source_id"])

    def test_model_dir_is_resolvable(self):
        self.assertTrue(str(resolve_model_dir()))


class TestUnit06MisjudgeRisk(unittest.TestCase):
    """第4节：解释一个高分误判风险——高分不等于该照搬它的处理方式。"""

    def setUp(self) -> None:
        require_model(self)

    def test_top_match_may_disagree_with_human_route(self):
        case = misjudge_check(RISK_QUERY, "rag")
        self.assertEqual(case["query"], RISK_QUERY)
        self.assertTrue(case["source_id"])
        # 电商售后里"退货运费谁承担"是政策咨询，不该被判成拒答转人工。
        self.assertNotEqual(case["top_route"], "rag")
        self.assertFalse(case["matched"])

    def test_scores_are_close_when_risk_happens(self):
        """风险之所以成立：第一名与第二名的分差很小，排序并不稳定。"""
        report = run_lab(RISK_QUERY)
        gap = report["ranked"][0]["score"] - report["ranked"][1]["score"]
        self.assertGreaterEqual(gap, 0)
        self.assertLess(gap, 0.05)


if __name__ == "__main__":
    unittest.main()
