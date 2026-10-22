"""单元09 验收测试：检索服务的 Top-K、来源过滤、项目隔离、批量与边界。

写入一律发生在临时目录，不碰单元08 的 `.chroma_unit8/` 与本单元 `.chroma_unit9/`。
需要本地向量模型与 chromadb；缺任一项时相关用例跳过并说明原因。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.embedding import MODEL_NAME, model_ready
from app.ingest import load_markdown

MODEL_AVAILABLE, MODEL_DETAIL = model_ready()

try:
    import chromadb  # noqa: F401

    from app.chroma_store import (
        COLLECTION_NAME,
        DEFAULT_UNIT9_DB_PATH,
        PROJECT_ID,
        ChromaVectorStore,
        resolve_unit9_db_path,
    )
    from app.retriever import (
        DEMO_QUESTION,
        DEMO_SOURCE,
        SECOND_QUESTION,
        RetrievalService,
        SearchResult,
        render_results,
    )

    CHROMA_AVAILABLE = True
    CHROMA_DETAIL = ""
except ImportError as exc:  # pragma: no cover - 依赖缺失时才会走这里
    CHROMA_AVAILABLE = False
    CHROMA_DETAIL = str(exc)


def require_service(test_case: unittest.TestCase) -> None:
    if not MODEL_AVAILABLE:
        test_case.skipTest(f"本地向量模型不可用：{MODEL_DETAIL}")
    if not CHROMA_AVAILABLE:
        test_case.skipTest(f"缺少 chromadb：{CHROMA_DETAIL}；请执行 uv sync --extra chroma")


def temp_dir(test_case: unittest.TestCase) -> Path:
    path = Path(tempfile.mkdtemp(prefix="chroma-unit9-"))
    test_case.addCleanup(shutil.rmtree, path, True)
    return path


def build_service(test_case: unittest.TestCase) -> RetrievalService:
    """在临时目录建库并完成首次导入，返回一个可查询的检索服务。"""
    store = ChromaVectorStore(temp_dir(test_case))
    store.upsert_documents(load_markdown())
    return RetrievalService(store)


class TestUnit09DatabaseLayout(unittest.TestCase):
    """第1课：本单元库路径独立于单元08，且可被环境变量覆盖。"""

    def test_unit09_database_is_not_unit08_database(self):
        from app.chroma_store import DEFAULT_DB_PATH

        self.assertNotEqual(DEFAULT_UNIT9_DB_PATH, DEFAULT_DB_PATH)
        self.assertTrue(str(DEFAULT_UNIT9_DB_PATH).endswith(".chroma_unit9"))

    def test_database_dir_can_be_overridden(self):
        import os

        original = os.environ.get("CHROMA_UNIT9_DB_DIR")
        os.environ["CHROMA_UNIT9_DB_DIR"] = "D:/tmp/elsewhere"
        try:
            self.assertEqual(resolve_unit9_db_path(), Path("D:/tmp/elsewhere"))
        finally:
            if original is None:
                os.environ.pop("CHROMA_UNIT9_DB_DIR", None)
            else:
                os.environ["CHROMA_UNIT9_DB_DIR"] = original

    def test_first_import_writes_every_document(self):
        require_service(self)
        store = ChromaVectorStore(temp_dir(self))
        self.assertEqual(store.count(), 0)
        store.upsert_documents(load_markdown())
        self.assertEqual(store.count(), len(load_markdown()))
        self.assertEqual(store.collection.name, COLLECTION_NAME)

    def test_repeated_import_does_not_grow(self):
        require_service(self)
        store = ChromaVectorStore(temp_dir(self))
        records = load_markdown()
        store.upsert_documents(records)
        store.upsert_documents(records)
        self.assertEqual(store.count(), len(records))


class TestUnit09TopK(unittest.TestCase):
    """第2、3课：Top-3/Top-5 数量与排序可复现，返回结构稳定。"""

    def setUp(self):
        require_service(self)
        self.service = build_service(self)

    def test_top_k_controls_return_count(self):
        self.assertEqual(len(self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=3)), 3)
        self.assertEqual(len(self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=5)), 5)

    def test_top3_is_prefix_of_top5(self):
        """同一个问题改 K，前三条必须完全一致——这就是"排序可复现"。"""
        top3 = self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=3)
        top5 = self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=5)
        self.assertEqual([hit.document_id for hit in top3],
                         [hit.document_id for hit in top5[:3]])
        self.assertEqual([hit.distance for hit in top3],
                         [hit.distance for hit in top5[:3]])

    def test_results_are_sorted_by_distance(self):
        distances = [hit.distance for hit in self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=5)]
        self.assertEqual(distances, sorted(distances), "距离应从小到大返回")

    def test_result_fields_are_complete(self):
        hit = self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=1)[0]
        self.assertIsInstance(hit, SearchResult)
        for field in ("document_id", "text", "source", "project_id", "doc_type"):
            self.assertTrue(getattr(hit, field), f"{field} 不能为空")
        self.assertIsInstance(hit.distance, float)
        self.assertGreaterEqual(hit.distance, 0.0)
        self.assertIn("DOC-02-", hit.document_id)
        self.assertEqual(hit.project_id, PROJECT_ID)
        # 原文是整篇 Markdown，不是被截断的片段。
        self.assertIn("DOC-02-", hit.text)

    def test_similarity_agrees_with_distance(self):
        """归一化向量下 2-2cos 应等于 l2 距离，可用来解释"距离的含义"。~/0.01 容差。"""
        hit = self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=1)[0]
        self.assertIsNotNone(hit.similarity)
        self.assertAlmostEqual(2 - 2 * hit.similarity, hit.distance, delta=0.01)


class TestUnit09Filters(unittest.TestCase):
    """第3课：来源过滤只返回该文件，其他项目为空。"""

    def setUp(self):
        require_service(self)
        self.service = build_service(self)

    def test_source_filter_returns_only_that_file(self):
        hits = self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=3, source=DEMO_SOURCE)
        self.assertTrue(hits, "指定来源应至少命中一条")
        for hit in hits:
            self.assertEqual(hit.source, DEMO_SOURCE)
        self.assertLessEqual(len(hits), 3, "命中不足 K 条时实际返回条数会少于 K")

    def test_unknown_source_returns_nothing(self):
        self.assertEqual(
            self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=3, source="不存在的文件.md"),
            [],
        )

    def test_other_project_returns_nothing(self):
        self.assertEqual(self.service.search(DEMO_QUESTION, "another_project", top_k=3), [])

    def test_blank_project_id_is_rejected(self):
        """项目标识是必选过滤条件，留空必须报错而不是退化成"查全部"。"""
        with self.assertRaises(ValueError):
            self.service.search(DEMO_QUESTION, "")


class TestUnit09Batch(unittest.TestCase):
    """第4课：批量查询与边界验证。"""

    def setUp(self):
        require_service(self)
        self.service = build_service(self)

    def test_search_many_keeps_question_order(self):
        questions = [DEMO_QUESTION, SECOND_QUESTION]
        batches = self.service.search_many(questions, PROJECT_ID, top_k=2)
        self.assertEqual(len(batches), len(questions))
        for batch in batches:
            self.assertEqual(len(batch), 2)
        self.assertNotEqual(
            batches[0][0].document_id, batches[1][0].document_id,
            "两个不同问题的首选命中不应完全相同",
        )

    def test_search_equals_first_batch_of_search_many(self):
        """单条接口委托批量接口，两者结果必须一致。"""
        single = self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=3)
        batched = self.service.search_many([DEMO_QUESTION], PROJECT_ID, top_k=3)[0]
        self.assertEqual([hit.document_id for hit in single],
                         [hit.document_id for hit in batched])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.service.search("   ", PROJECT_ID)
        self.assertIn("问题", str(ctx.exception))

    def test_empty_question_list_is_rejected(self):
        with self.assertRaises(ValueError):
            self.service.search_many([], PROJECT_ID)
        with self.assertRaises(ValueError):
            self.service.search_many(["有内容", "  "], PROJECT_ID)

    def test_bad_project_and_top_k_are_rejected(self):
        with self.assertRaises(ValueError):
            self.service.search(DEMO_QUESTION, "   ")
        with self.assertRaises(ValueError):
            self.service.search(DEMO_QUESTION, PROJECT_ID, top_k=0)
        with self.assertRaises(ValueError):
            self.service.search(DEMO_QUESTION, PROJECT_ID, source="")

    def test_empty_collection_returns_empty_batches(self):
        empty = RetrievalService(ChromaVectorStore(temp_dir(self)))
        self.assertEqual(empty.search(DEMO_QUESTION, PROJECT_ID), [])
        self.assertEqual(empty.search_many([DEMO_QUESTION, SECOND_QUESTION], PROJECT_ID), [[], []])


class TestUnit09Rendering(unittest.TestCase):
    """验收证据用的渲染：条数、来源、项目、距离都要在文本里。"""

    def test_render_shows_count_and_fields(self):
        hit = SearchResult(
            document_id=f"{PROJECT_ID}:DOC-02-01",
            text="# 普通商品退货规则",
            source="模拟数据/knowledge/DOC-02-01-普通商品退货规则.md",
            project_id=PROJECT_ID,
            doc_type="knowledge",
            distance=0.6539,
            similarity=0.6731,
        )
        text = render_results("Top-1", [hit])
        self.assertIn("Top-1：1 条", text)
        self.assertIn("DOC-02-01", text)
        self.assertIn("0.6539", text)
        self.assertIn(PROJECT_ID, text)

    def test_render_shows_hint_when_nothing_hit(self):
        self.assertIn("未检索到符合过滤条件的资料", render_results("其他项目", []))


if __name__ == "__main__":
    unittest.main()
