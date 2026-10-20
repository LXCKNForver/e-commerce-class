"""单元07 验收测试：记录字段、精确检索基线、元数据过滤、项目隔离，以及 Chroma 空集合。

需要本地向量模型的用例在模型缺失时跳过；Chroma 用例在 chromadb 缺失时跳过。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.embedding import EXPECTED_DIM, model_ready
from app.vector_store import (
    TENANT_ID,
    InMemoryVectorStore,
    build_demo_records,
    load_knowledge_texts,
    load_question_texts,
)

MODEL_AVAILABLE, MODEL_DETAIL = model_ready()

try:
    import chromadb  # noqa: F401

    from setup_chroma import COLLECTION_NAME, create_first_collection

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


class TestUnit07RecordFields(unittest.TestCase):
    """第1课：一条记录的字段图——身份、原文、来源、元数据、向量、项目隔离。"""

    def test_knowledge_texts_are_ecommerce_after_sale(self):
        items = load_knowledge_texts()
        self.assertGreaterEqual(len(items), 5)
        for item in items:
            self.assertTrue(item["document_id"].startswith("DOC-02-"))
            self.assertTrue(item["text"].strip())
            self.assertEqual(item["doc_type"], "knowledge")
            self.assertIn("knowledge", item["source"])

    def test_question_texts_are_ecommerce_after_sale(self):
        items = load_question_texts()
        self.assertGreaterEqual(len(items), 5)
        for item in items:
            self.assertTrue(item["document_id"].startswith("Q-02-"))
            self.assertEqual(item["doc_type"], "question")

    def test_demo_corpus_has_both_types(self):
        items = build_demo_records()
        types = {item["doc_type"] for item in items}
        self.assertEqual(types, {"knowledge", "question"})
        ids = [item["document_id"] for item in items]
        self.assertEqual(len(ids), len(set(ids)), "演示语料里不能有重复 ID")


class TestUnit07StoreValidation(unittest.TestCase):
    """集合自身的校验：名称、维度、必填字段、重复 ID、top_k。"""

    def test_blank_collection_name_is_rejected(self):
        with self.assertRaises(ValueError):
            InMemoryVectorStore("   ")

    def test_non_positive_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            InMemoryVectorStore("after_sale", dimension=0)

    def test_blank_fields_are_rejected(self):
        store = InMemoryVectorStore("after_sale")
        with self.assertRaises(ValueError):
            store.add("DOC-02-05", "   ", "source", "knowledge", TENANT_ID)

    def test_bad_top_k_is_rejected(self):
        store = InMemoryVectorStore("after_sale")
        with self.assertRaises(ValueError):
            store.query("随便问一句", TENANT_ID, top_k=0)
        with self.assertRaises(ValueError):
            store.query("随便问一句", "   ")

    def test_blank_tenant_id_is_rejected(self):
        store = InMemoryVectorStore("after_sale")
        with self.assertRaises(ValueError):
            store.query("随便问一句", "  ")


class TestUnit07ExactSearch(unittest.TestCase):
    """精确遍历基线：这是后面谈 HNSW/IVF 时的对照标准。"""

    def setUp(self) -> None:
        require_model(self)
        self.store = InMemoryVectorStore("after_sale_knowledge_base_demo")
        self.store.add_many(build_demo_records())

    def test_records_carry_expected_dimension(self):
        self.assertEqual(self.store.count(), 21)
        for record in self.store.records:
            self.assertEqual(len(record.vector), EXPECTED_DIM)

    def test_count_can_filter_by_type(self):
        self.assertEqual(self.store.count(doc_type="knowledge"), 15)
        self.assertEqual(self.store.count(doc_type="question"), 6)

    def test_query_is_sorted_and_capped(self):
        hits = self.store.query("商品有质量问题可以退款吗？", TENANT_ID, top_k=3)
        self.assertEqual(len(hits), 3)
        scores = [hit.score for hit in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_metadata_filter_narrows_candidates(self):
        all_hits = self.store.query("能退货吗", TENANT_ID, top_k=5)
        typed = self.store.query("能退货吗", TENANT_ID, top_k=5, doc_type="knowledge")
        self.assertLessEqual(len(typed), len(all_hits))
        self.assertTrue(all(hit.record.doc_type == "knowledge" for hit in typed))

    def test_tenant_isolation(self):
        self.assertEqual(self.store.query("能退货吗", "other-project"), [])
        self.assertEqual(self.store.count("other-project"), 0)

    def test_duplicate_document_id_is_rejected(self):
        first = self.store.records[0]
        with self.assertRaises(ValueError):
            self.store.add(
                first.document_id, first.text, first.source, first.doc_type, TENANT_ID
            )


class TestUnit07ChromaCollection(unittest.TestCase):
    """第3、4课：真实 Chroma 建空集合、只读查看、重复运行集合 ID 不变。"""

    def setUp(self) -> None:
        require_chroma(self)

    def test_collection_is_created_once_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "chroma"
            # 两个独立进程分别建库，避免 Windows 上 SQLite 句柄妨碍临时目录清理。
            code = (
                "import sys; sys.path.insert(0, r'"
                + str(Path(__file__).resolve().parents[1])
                + "'); "
                "from pathlib import Path; "
                "from setup_chroma import create_first_collection; "
                "_, c = create_first_collection(Path(sys.argv[1])); "
                "print(c.name, c.id, c.count())"
            )
            runs = [
                subprocess.run(
                    [sys.executable, "-c", code, str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                for _ in range(2)
            ]
            self.assertEqual(runs[0], runs[1], "重复运行必须复用同一个集合 ID")
            self.assertTrue(runs[0].startswith(COLLECTION_NAME + " "))
            self.assertTrue(runs[0].endswith(" 0"), "本单元集合必须是空的")

    def test_inspect_reads_existing_collection(self):
        from inspect_chroma import inspect_collection
        from setup_chroma import CHROMA_PATH

        create_first_collection()
        info = inspect_collection()
        self.assertEqual(info["name"], COLLECTION_NAME)
        self.assertEqual(info["count"], 0)
        self.assertEqual(info["metadata"]["embedding_model"], "BAAI/bge-small-zh-v1.5")
        self.assertEqual(info["metadata"]["dimension"], 512)
        self.assertIn("after_sale", info["metadata"]["purpose"])
        self.assertTrue(Path(info["database_path"]) == CHROMA_PATH)

    def test_inspect_requires_setup_first(self):
        from inspect_chroma import inspect_collection

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                inspect_collection(Path(temp))


if __name__ == "__main__":
    unittest.main()
