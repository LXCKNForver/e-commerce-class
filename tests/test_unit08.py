"""单元08 验收测试：资料清单、批量写入、持久化、项目内检索、重复运行不增加。

需要本地向量模型与 chromadb；缺任一项时相关用例跳过并说明原因。
写入一律发生在临时目录，不碰单元07 的 `.chroma_unit7/` 验收现场。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from app.embedding import MODEL_NAME, model_ready
from app.ingest import KnowledgeDocument, load_markdown, render_inventory
from app.vector_store import TENANT_ID

MODEL_AVAILABLE, MODEL_DETAIL = model_ready()

try:
    import chromadb  # noqa: F401

    from app.chroma_store import COLLECTION_NAME, DEMO_QUESTION, PROJECT_ID, ChromaVectorStore

    CHROMA_AVAILABLE = True
    CHROMA_DETAIL = ""
except ImportError as exc:  # pragma: no cover - 依赖缺失时才会走这里
    CHROMA_AVAILABLE = False
    CHROMA_DETAIL = str(exc)


def require_model(test_case: unittest.TestCase) -> None:
    if not MODEL_AVAILABLE:
        test_case.skipTest(f"本地向量模型不可用：{MODEL_DETAIL}")


def require_chroma(test_case: unittest.TestCase) -> None:
    if not CHROMA_AVAILABLE:
        test_case.skipTest(f"缺少 chromadb：{CHROMA_DETAIL}；请执行 uv sync --extra chroma")


def temp_dir(test_case: unittest.TestCase) -> Path:
    """给每个用例一个独立数据库目录；Windows 上句柄未及时释放时忽略清理失败。"""
    path = Path(tempfile.mkdtemp(prefix="chroma-unit8-"))
    test_case.addCleanup(shutil.rmtree, path, True)
    return path


class TestUnit08Inventory(unittest.TestCase):
    """第1课：输入资料清单——真实目录、稳定 ID、来源、项目标识。"""

    def test_project_documents_are_ecommerce_after_sale(self):
        records = load_markdown()
        self.assertGreaterEqual(len(records), 5, "任务卡要求至少 5 篇本组资料")
        ids = [record.document_id for record in records]
        self.assertEqual(len(ids), len(set(ids)), "稳定 ID 不能重复")
        for record in records:
            self.assertTrue(record.document_id.startswith(TENANT_ID + ":"))
            self.assertIn("DOC-02-", record.document_id)
            self.assertIn("模拟数据/knowledge/", record.source)
            self.assertTrue(record.text.strip())
            self.assertEqual(record.project_id, TENANT_ID)
            self.assertEqual(record.doc_type, "knowledge")

    def test_metadata_carries_source_project_and_type(self):
        record = load_markdown()[0]
        self.assertEqual(record.metadata["project_id"], TENANT_ID)
        self.assertEqual(record.metadata["doc_type"], "knowledge")
        self.assertTrue(record.metadata["source"].endswith(".md"))

    def test_inventory_lists_sources(self):
        text = render_inventory(load_markdown(), limit=5)
        self.assertIn("模拟数据/knowledge/", text)
        self.assertIn("DOC-02-01", text)
        self.assertIn("其余 10 篇", text)

    def test_missing_folder_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                load_markdown(Path(temp) / "不存在的目录")

    def test_empty_folder_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                load_markdown(Path(temp))

    def test_too_few_documents_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "DOC-02-01-只有一篇.md").write_text("# 只有一篇\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_markdown(folder)

    def test_blank_project_id_is_rejected(self):
        with self.assertRaises(ValueError):
            load_markdown(project_id="   ")


class TestUnit08ChromaWrite(unittest.TestCase):
    """第2、4课：批量 upsert、重新连接仍有记录、重复运行数量不增加。"""

    def setUp(self) -> None:
        require_model(self)
        require_chroma(self)
        self.database_path = temp_dir(self) / "chroma"
        self.records = load_markdown()
        self.store = ChromaVectorStore(self.database_path)

    def test_collection_reuses_unit07_definition(self):
        self.assertEqual(self.store.collection.name, COLLECTION_NAME)
        self.assertEqual(self.store.collection.metadata["embedding_model"], MODEL_NAME)

    def test_upsert_fills_the_collection(self):
        self.assertEqual(self.store.count(), 0, "新建集合应当是空的")
        written = self.store.upsert_documents(self.records)
        self.assertEqual(written, len(self.records))
        self.assertEqual(self.store.count(), len(self.records))

    def test_reopen_keeps_records_and_collection_id(self):
        self.store.upsert_documents(self.records)
        reopened = ChromaVectorStore(self.database_path)
        self.assertEqual(reopened.collection.id, self.store.collection.id)
        self.assertEqual(reopened.count(), len(self.records))

    def test_repeated_upsert_does_not_grow(self):
        """稳定 ID 让第二次导入变成更新：数量必须保持不变。"""
        self.store.upsert_documents(self.records)
        first = self.store.count()
        reopened = ChromaVectorStore(self.database_path)
        reopened.upsert_documents(self.records)
        self.assertEqual(reopened.count(), first)
        self.assertEqual(ChromaVectorStore(self.database_path).count(), first)

    def test_empty_record_list_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.upsert_documents([])


class TestUnit08ChromaQuery(unittest.TestCase):
    """第2、4课：项目内 Top-K 检索，返回来源与距离。"""

    def setUp(self) -> None:
        require_model(self)
        require_chroma(self)
        self.store = ChromaVectorStore(temp_dir(self) / "chroma")
        self.store.upsert_documents(load_markdown())

    def test_query_returns_top_k_with_source_and_distance(self):
        hits = self.store.query(DEMO_QUESTION, PROJECT_ID, top_k=3)
        self.assertEqual(len(hits), 3)
        for hit in hits:
            self.assertIn("模拟数据/knowledge/", hit["source"])
            self.assertTrue(hit["id"].startswith(PROJECT_ID + ":"))
            self.assertIsInstance(hit["distance"], float)
            self.assertGreaterEqual(hit["distance"], 0.0)
            self.assertGreater(hit["similarity"], 0.0)

    def test_hits_are_sorted_by_distance(self):
        hits = self.store.query(DEMO_QUESTION, PROJECT_ID, top_k=3)
        distances = [hit["distance"] for hit in hits]
        self.assertEqual(distances, sorted(distances), "距离应从小到大返回")

    def test_project_filter_blocks_other_projects(self):
        self.assertEqual(self.store.query(DEMO_QUESTION, "other-project"), [])

    def test_query_on_empty_collection_returns_nothing(self):
        empty = ChromaVectorStore(temp_dir(self) / "empty")
        self.assertEqual(empty.query(DEMO_QUESTION, PROJECT_ID), [])

    def test_bad_query_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.query("   ", PROJECT_ID)
        with self.assertRaises(ValueError):
            self.store.query(DEMO_QUESTION, "   ")
        with self.assertRaises(ValueError):
            self.store.query(DEMO_QUESTION, PROJECT_ID, top_k=0)

    def test_single_document_collection_accepts_smaller_top_k(self):
        """请求条数超过库内条数时不应报错，而是返回现有条数。"""
        small = ChromaVectorStore(temp_dir(self) / "single")
        small.upsert_documents(
            [
                KnowledgeDocument(
                    document_id=f"{PROJECT_ID}:DOC-99-01",
                    text="仅用于测试的售后说明：签收后 7 天内可申请退货。",
                    source="测试/UNIT08.md",
                    project_id=PROJECT_ID,
                )
            ]
        )
        self.assertEqual(len(small.query(DEMO_QUESTION, PROJECT_ID, top_k=3)), 1)


if __name__ == "__main__":
    unittest.main()
