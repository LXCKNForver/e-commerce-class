"""单元11 验收测试：重叠分块 → 分块召回 → BGE 重排 → 两版回答对照。

分块是纯字符处理，**所有切分用例永远运行**，不依赖任何模型。
需要本地 BGE 权重的用例先探测 `rerank_ready()`，模型不在就跳过并说明原因（不判失败）。
涉及网络的 DeepSeek 调用一律用替身模型：本单元要验证的是候选怎么变、顺序怎么变，
真实回答由 `scripts/generate_evidence_unit11.py` 单独留证据。

缺本地重排模型时，先按课程材料说明下载 bge-reranker-base，
再用 `RERANK_MODEL_DIR` 指向它（脚本与测试都会读项目根的 .env）。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 只为拿到 RERANK_MODEL_DIR 指向的本地模型目录；密钥不会被读取到任何输出里。
load_dotenv(PROJECT_ROOT / ".env")

from app.chunking import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEMO_SOURCE,
    SECOND_CHUNK_START,
    TextChunk,
    chunks_of,
    overlap_between,
    render_chunk_positions,
    render_chunk_texts,
    split_documents,
    split_whole_documents,
)
from app.ingest import load_markdown  # noqa: E402
from app.rag_opt import (  # noqa: E402
    CHUNK_TOP_K,
    DEMO_QUESTION,
    DIRECT_EVIDENCE_KEYWORD,
    DIRECT_EVIDENCE_SOURCE,
    RERANK_KEEP,
    WHOLE_TOP_K,
    OptimizedRun,
    RAGOptimizer,
    render_answers,
    render_evidence_check,
    render_recall,
)
from app.rerank import (  # noqa: E402
    RERANKER_NAME,
    rerank_hits,
    rerank_ready,
)
from app.retriever import SearchResult  # noqa: E402

RERANK_AVAILABLE, RERANK_DETAIL = rerank_ready()

try:
    import chromadb  # noqa: F401

    from app.chroma_store import (
        DEFAULT_UNIT11_DB_PATH,
        UNIT11_CHUNK_COLLECTION,
        UNIT11_WHOLE_COLLECTION,
        ChromaVectorStore,
        resolve_unit11_db_path,
    )
    from app.embedding import model_ready

    CHROMA_AVAILABLE = True
    CHROMA_DETAIL = ""
except ImportError as exc:  # pragma: no cover - 依赖缺失时才会走这里
    CHROMA_AVAILABLE = False
    CHROMA_DETAIL = str(exc)
    MODEL_AVAILABLE = False
else:
    MODEL_AVAILABLE, MODEL_DETAIL = model_ready()

if CHROMA_AVAILABLE and not MODEL_AVAILABLE:
    CHROMA_DETAIL = MODEL_DETAIL


def make_chunk(text: str, start: int = 0, source: str = "s.md") -> TextChunk:
    return TextChunk(
        document_id=f"e-commerce-after-sales:DOC-02-11:{start}:{start + len(text)}",
        text=text,
        source=source,
        project_id="e-commerce-after-sales",
        doc_type="knowledge",
        start=start,
        end=start + len(text),
    )


class TestUnit11Chunking(unittest.TestCase):
    """0–8 分钟：相邻块的起止位置与重叠内容（纯字符处理，永远运行）。"""

    def setUp(self) -> None:
        self.records = load_markdown()
        self.chunks = split_documents(self.records)

    def test_second_chunk_starts_at_size_minus_overlap(self):
        """第二块从第 60 个字符开始：80 - 20 = 60，这是可以直接数出来的位置。"""
        target = chunks_of(self.chunks, DEMO_SOURCE)
        self.assertGreaterEqual(len(target), 2)
        self.assertEqual(target[0].start, 0)
        self.assertEqual(target[0].end, CHUNK_SIZE)
        self.assertEqual(target[1].start, SECOND_CHUNK_START)
        self.assertEqual(SECOND_CHUNK_START, CHUNK_SIZE - CHUNK_OVERLAP)

    def test_adjacent_chunks_share_exactly_overlap_characters(self):
        """相邻两块共享 20 个字符，且这段内容在两块里字面相同。"""
        target = chunks_of(self.chunks, DEMO_SOURCE)
        for previous, current in zip(target, target[1:]):
            with self.subTest(pair=(previous.position, current.position)):
                shared = overlap_between(previous, current)
                self.assertEqual(len(shared), CHUNK_OVERLAP)
                self.assertTrue(previous.text.endswith(shared))
                self.assertTrue(current.text.startswith(shared))

    def test_every_chunk_is_within_window_and_covers_the_document(self):
        """每块不超过窗口大小，最后一块落在文末，几块合起来覆盖整篇。"""
        for record in self.records:
            parts = chunks_of(self.chunks, record.source)
            with self.subTest(source=record.source):
                self.assertLessEqual(max(len(part.text) for part in parts), CHUNK_SIZE)
                self.assertEqual(parts[0].start, 0)
                self.assertEqual(parts[-1].end, len(record.text))
                covered = set()
                for part in parts:
                    covered.update(range(part.start, part.end))
                self.assertEqual(len(covered), len(record.text))

    def test_chunk_ids_are_unique_and_carry_positions(self):
        ids = [chunk.document_id for chunk in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        sample = self.chunks[0]
        self.assertTrue(sample.document_id.endswith(f":{sample.start}:{sample.end}"))
        self.assertEqual(sample.position, f"[{sample.start},{sample.end})")

    def test_whole_document_mode_keeps_one_record_per_document(self):
        """对照组：整篇一条，与单元08—10 的入库方式完全一致。"""
        whole = split_whole_documents(self.records)
        self.assertEqual(len(whole), len(self.records))
        for chunk in whole:
            self.assertEqual(chunk.start, 0)
            self.assertEqual(chunk.end, len(chunk.text))

    def test_metadata_keeps_source_project_and_position(self):
        meta = self.chunks[0].metadata
        self.assertEqual(meta["source"], self.chunks[0].source)
        self.assertEqual(meta["project_id"], "e-commerce-after-sales")
        self.assertEqual(meta["doc_type"], "knowledge")
        self.assertEqual(meta["start"], self.chunks[0].start)
        self.assertEqual(meta["end"], self.chunks[0].end)

    def test_invalid_window_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            split_documents(self.records, chunk_size=0)
        with self.assertRaises(ValueError):
            split_documents(self.records, chunk_size=80, overlap=80)
        with self.assertRaises(ValueError):
            split_documents(self.records, chunk_size=80, overlap=-1)
        with self.assertRaises(ValueError):
            split_documents(self.records, chunk_size=None, overlap=20)
        with self.assertRaises(ValueError):
            split_documents([])

    def test_demo_source_is_the_direct_evidence_document(self):
        """演示题的直接证据就在被观察切分的那份资料里。"""
        self.assertIn("DOC-02-11", DEMO_SOURCE)
        record = next(r for r in self.records if r.source == DEMO_SOURCE)
        self.assertIn(DIRECT_EVIDENCE_KEYWORD, record.text)

    def test_render_outputs_positions_and_overlap(self):
        text = render_chunk_positions(chunks_of(self.chunks, DEMO_SOURCE))
        self.assertIn(f"窗口 {CHUNK_SIZE} 字符", text)
        self.assertIn(f"第 {SECOND_CHUNK_START} 个字符开始", text)
        self.assertIn(f"{CHUNK_OVERLAP} 字符：", text)
        self.assertIn("（首块）", text)
        self.assertTrue(render_chunk_texts(chunks_of(self.chunks, DEMO_SOURCE)).startswith("片段正文"))

    def test_render_handles_empty_input(self):
        self.assertIn("没有可展示的片段", render_chunk_positions([]))


@dataclass(frozen=True)
class FakeHit:
    """替身命中：字段与单元09 的 SearchResult 一致，便于脱离 Chroma 测试。"""

    document_id: str
    text: str
    source: str
    project_id: str = "e-commerce-after-sales"
    doc_type: str = "knowledge"
    distance: float = 0.5
    similarity: float | None = None


class FakeRetriever:
    """替身检索器：返回预先排好的候选，不做任何向量计算。"""

    def __init__(self, hits: list[FakeHit], top_k: int):
        self.hits = hits
        self.top_k = top_k
        self.calls: list[tuple[str, int]] = []

    def search(self, question, project_id, top_k, source=None, doc_type=None):
        self.calls.append((question, top_k))
        return list(self.hits[:top_k])


class ScriptedLLM:
    """替身生成模型：记录收到的上下文，按脚本回答。"""

    def __init__(self, reply: str = "依据资料回答 [1]。"):
        self.reply = reply
        self.prompts: list[list[dict]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.prompts.append(messages)
        return self.reply


class ScriptedOptimizer(RAGOptimizer):
    """把重排换成固定分数，让"顺序变化"这条规则脱离 1.1GB 权重也能验证。"""

    def __init__(self, *args, scores: dict[str, float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = scores or {}
        self.rerank_calls: list[str] = []

    def rerank(self, question, candidates):
        self.rerank_calls.append(question)
        ranked = [
            (hit, self.scores.get(hit.document_id, 0.0)) for hit in candidates
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked


def demo_hits() -> tuple[list[FakeHit], list[FakeHit]]:
    """构造与真实运行同构的两组候选：整篇三条不含直接证据，分块五条含。"""
    baseline = [
        FakeHit("e-commerce-after-sales:DOC-02-01:0:160", "普通商品签收后7天内且未使用可申请退货评估。",
                "模拟数据/knowledge/DOC-02-01-普通商品退货规则.md", distance=0.91),
        FakeHit("e-commerce-after-sales:DOC-02-03:0:150", "已发货未签收的订单先查询物流。",
                "模拟数据/knowledge/DOC-02-03-已发货未签收处理.md", distance=0.96),
        FakeHit("e-commerce-after-sales:DOC-02-05:0:144", "定制商品不适用普通7天退货规则。",
                "模拟数据/knowledge/DOC-02-05-定制商品例外.md", distance=0.96),
    ]
    candidates = [
        FakeHit("e-commerce-after-sales:DOC-02-06:60:140", "已拆封易耗品不按普通无理由退货处理。",
                "模拟数据/knowledge/DOC-02-06-易耗品例外.md", distance=0.80),
        FakeHit("e-commerce-after-sales:DOC-02-11:60:140",
                f"漏发需{DIRECT_EVIDENCE_KEYWORD}。缺少证据时请求补充信息。",
                DIRECT_EVIDENCE_SOURCE, distance=0.82),
        FakeHit("e-commerce-after-sales:DOC-02-10:60:140", "签收破损应提交订单号、包装与商品照片说明。",
                "模拟数据/knowledge/DOC-02-10-破损件处理.md", distance=0.87),
        FakeHit("e-commerce-after-sales:DOC-02-03:60:140", "已发货未签收的订单先查询物流。",
                "模拟数据/knowledge/DOC-02-03-已发货未签收处理.md", distance=0.88),
        FakeHit("e-commerce-after-sales:DOC-02-05:60:140", "定制商品不适用课程样例的普通7天退货规则。",
                "模拟数据/knowledge/DOC-02-05-定制商品例外.md", distance=0.89),
    ]
    return baseline, candidates


class TestUnit11Pipeline(unittest.TestCase):
    """8–35 分钟：两版候选、重排顺序、两版回答（用替身，不碰网络）。"""

    def build(self, scores: dict[str, float] | None = None):
        baseline, candidates = demo_hits()
        llm = ScriptedLLM("收到商品少一件属于漏发情形，需核对差异 [1]。")
        optimizer = ScriptedOptimizer(
            FakeRetriever(baseline, WHOLE_TOP_K),  # type: ignore[arg-type]
            FakeRetriever(candidates, CHUNK_TOP_K),  # type: ignore[arg-type]
            llm,
            scores=scores,
        )
        return optimizer, llm, baseline, candidates

    def test_recall_uses_different_top_k_for_two_versions(self):
        """整篇取 Top3、分块取 Top5，这是任务卡给定的两条召回路径。"""
        optimizer, _, baseline, candidates = self.build()
        got_baseline, got_candidates = optimizer.recall(DEMO_QUESTION)
        self.assertEqual(len(got_baseline), len(baseline))
        self.assertEqual(len(got_candidates), len(candidates))
        self.assertEqual(optimizer.whole_retriever.top_k, WHOLE_TOP_K)
        self.assertEqual(optimizer.chunk_retriever.top_k, CHUNK_TOP_K)
        # 两版召回的是同一个问题，只是候选来源不同。
        self.assertEqual(
            optimizer.whole_retriever.calls[0][0], optimizer.chunk_retriever.calls[0][0]
        )

    def test_rerank_keeps_candidates_and_only_reorders(self):
        """重排只改顺序，不增删候选——这是它能力的上限。"""
        optimizer, _, _, candidates = self.build(
            scores={"e-commerce-after-sales:DOC-02-11:60:140": 3.0}
        )
        ranked = optimizer.rerank(DEMO_QUESTION, candidates)
        self.assertEqual(len(ranked), len(candidates))
        self.assertEqual(ranked[0][0].document_id, "e-commerce-after-sales:DOC-02-11:60:140")
        self.assertEqual(
            {hit.document_id for hit, _ in ranked},
            {hit.document_id for hit in candidates},
        )
        scores = [score for _, score in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_optimized_hits_are_top_three_after_rerank(self):
        optimizer, _, _, _ = self.build(
            scores={"e-commerce-after-sales:DOC-02-11:60:140": 3.0}
        )
        run = optimizer.compare(DEMO_QUESTION, document_total=15, chunk_total=45)
        self.assertEqual(len(run.optimized_hits), RERANK_KEEP)
        self.assertEqual(len(run.candidates), CHUNK_TOP_K)
        self.assertTrue(run.order_changed)

    def test_direct_evidence_only_enters_candidates_by_chunking(self):
        """任务卡要说明的那一条：直接证据在整篇候选里没有，靠分块才进来。"""
        optimizer, _, _, _ = self.build()
        run = optimizer.compare(DEMO_QUESTION)
        self.assertFalse(run.direct_evidence_in_baseline)
        self.assertTrue(run.direct_evidence_in_candidates)
        self.assertIn("靠分块才进入候选池", render_evidence_check(run))

    def test_two_versions_share_one_llm_and_prompt(self):
        """两版唯一差别是候选：生成模型、提示词、核验规则必须完全相同。"""
        optimizer, llm, _, _ = self.build()
        run = optimizer.compare(DEMO_QUESTION)
        self.assertEqual(len(llm.prompts), 2)
        self.assertEqual(llm.prompts[0][0]["content"], llm.prompts[1][0]["content"])
        self.assertIn("【检索资料】", llm.prompts[0][1]["content"])
        self.assertIn("【检索资料】", llm.prompts[1][1]["content"])
        self.assertNotEqual(
            llm.prompts[0][1]["content"], llm.prompts[1][1]["content"]
        )
        self.assertTrue(run.before.llm_called)
        self.assertTrue(run.after.llm_called)

    def test_baseline_answer_falls_back_when_evidence_is_missing(self):
        """基础版拿到的三条都不是本题依据，替身模型按提示词要求输出拒答话术。"""
        optimizer, llm, _, _ = self.build()
        llm.reply = "未检索到足够资料，暂不回答。"
        run = optimizer.compare(DEMO_QUESTION)
        self.assertEqual(run.before.sources, [])
        self.assertEqual(run.after.sources, [])

    def test_rerank_does_not_change_when_scores_keep_order(self):
        optimizer, _, _, candidates = self.build()
        run = optimizer.compare(DEMO_QUESTION)
        self.assertFalse(run.order_changed)  # 分数全为 0，稳定排序保持原顺序
        self.assertEqual(
            [hit.document_id for hit in run.optimized_hits],
            [hit.document_id for hit in candidates[:RERANK_KEEP]],
        )

    def test_render_outputs_carry_both_versions(self):
        optimizer, _, _, _ = self.build(
            scores={"e-commerce-after-sales:DOC-02-11:60:140": 3.0}
        )
        run = optimizer.compare(DEMO_QUESTION, document_total=15, chunk_total=45)
        text = render_recall(run)
        self.assertIn(f"基础版（整篇检索 Top{WHOLE_TOP_K}", text)
        self.assertIn(f"优化版（重叠分块召回 Top{CHUNK_TOP_K}", text)
        answers = render_answers(run)
        self.assertIn("基础版回答：", answers)
        self.assertIn("优化版回答：", answers)


@unittest.skipUnless(CHROMA_AVAILABLE, f"需要 chromadb：{CHROMA_DETAIL}")
class TestUnit11Stores(unittest.TestCase):
    """单元11 的两个集合：整篇与子块分开存，库目录与其他单元隔离。"""

    @classmethod
    def setUpClass(cls) -> None:
        if not MODEL_AVAILABLE:
            raise unittest.SkipTest(f"本地向量模型不可用：{CHROMA_DETAIL}")
        cls._dir = Path(tempfile.mkdtemp(prefix="chroma-unit11-cls-"))
        cls.records = load_markdown()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._dir, True)

    def test_two_collections_hold_their_own_counts(self):
        whole = ChromaVectorStore(self._dir, UNIT11_WHOLE_COLLECTION)
        chunk = ChromaVectorStore(self._dir, UNIT11_CHUNK_COLLECTION)
        whole.upsert_documents(split_whole_documents(self.records))
        chunk.upsert_documents(split_documents(self.records))
        self.assertEqual(whole.count(), len(self.records))
        self.assertEqual(chunk.count(), len(split_documents(self.records)))
        self.assertNotEqual(whole.collection.name, chunk.collection.name)

    def test_reimport_does_not_grow_chunk_collection(self):
        """稳定 ID（来源 + 起止位置）让重复导入只是更新。"""
        chunk = ChromaVectorStore(self._dir, UNIT11_CHUNK_COLLECTION)
        total = len(split_documents(self.records))
        chunk.upsert_documents(split_documents(self.records))
        chunk.upsert_documents(split_documents(self.records))
        self.assertEqual(chunk.count(), total)

    def test_unit11_has_own_database_path(self):
        self.assertEqual(DEFAULT_UNIT11_DB_PATH.name, ".chroma_unit11")
        self.assertEqual(resolve_unit11_db_path(), DEFAULT_UNIT11_DB_PATH)
        self.assertNotEqual(DEFAULT_UNIT11_DB_PATH.name, ".chroma_unit10")

    def test_real_search_result_fields_are_kept(self):
        """真实检索返回的字段与单元09 的 SearchResult 完全一致。"""
        whole = ChromaVectorStore(self._dir, UNIT11_WHOLE_COLLECTION)
        whole.upsert_documents(split_whole_documents(self.records))
        from app.retriever import RetrievalService

        hits = RetrievalService(whole).search(DEMO_QUESTION, top_k=3)
        self.assertEqual(len(hits), 3)
        for hit in hits:
            with self.subTest(hit=hit.document_id):
                self.assertIsInstance(hit, SearchResult)
                self.assertTrue(hit.source.startswith("模拟数据/knowledge/"))
                self.assertEqual(hit.project_id, "e-commerce-after-sales")


@unittest.skipUnless(RERANK_AVAILABLE, f"需要本地 BGE 重排模型：{RERANK_DETAIL}")
class TestUnit11Reranker(unittest.TestCase):
    """16–28 分钟：真实 BGE Cross-Encoder 的分数与配对。"""

    def test_related_candidate_scores_higher(self):
        hits = [
            FakeHit("a", "漏发需核对订单商品与收到商品的差异。缺少证据时请求补充信息。", "s1"),
            FakeHit("b", "样例发票状态有未申请、处理中、已开具。", "s2"),
        ]
        ranked = rerank_hits("收到的商品少了一件怎么办？", hits)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0][0].document_id, "a")
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_scores_are_paired_with_input_order(self):
        """分数必须与输入候选一一对应，不能先排序再配对。"""
        hits = [
            FakeHit("a", "签收破损应提交订单号、包装与商品照片说明。", "s1"),
            FakeHit("b", "优惠券是否可退回以样例订单中的优惠金额为准。", "s2"),
            FakeHit("c", "投诉包含人身攻击时应停止自动承诺并转人工处理。", "s3"),
        ]
        ranked = rerank_hits("签收时包装破损需要提交什么材料？", hits)
        self.assertEqual([hit.document_id for hit, _ in ranked][0], "a")
        self.assertEqual(len({id(hit) for hit, _ in ranked}), 3)

    def test_empty_candidates_return_empty(self):
        self.assertEqual(rerank_hits("问题", []), [])

    def test_blank_question_is_rejected(self):
        with self.assertRaises(ValueError):
            rerank_hits("   ", [FakeHit("a", "内容", "s1")])

    def test_model_name_is_reported(self):
        ready, detail = rerank_ready()
        self.assertTrue(ready)
        self.assertEqual(detail, RERANKER_NAME)


if __name__ == "__main__":
    unittest.main()
