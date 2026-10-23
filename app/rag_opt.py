"""单元11：把"重叠分块 + BGE 重排"接进单元10 的 RAG，并保留基础版做对照。

单元10 的链路是：整篇检索 Top3 → 编号上下文 → 生成 → 核验。
本单元只改链路最前面的"候选是怎么来的"，后面四步一个字都不动：

    基础版  整篇检索 Top3                    → 编号 → 生成 → 核验
    优化版  分块召回 Top5 → BGE 重排取前3    → 编号 → 生成 → 核验

两版共用**同一个 llm 对象、同一份 rag_prompts、同一个 RAGService 编排**，
所以两版答案若有差异，只能来自候选集合不同，不会被提示词差异污染。

课堂上要能回答三个问题：
1. 直接证据（支持答案的那一句原文）是怎么进入候选的？——本组这道题里，
   它在整篇 Top3 中根本不出现，是靠分块才进来的。
2. 重排前后顺序变了吗？——本组这道题里变了，而且升到第一位的是真正相关那条。
3. 重排能找回候选池之外的证据吗？——不能。召回定上限，重排只改顺序。

    uv run --extra embedding --extra chroma python -c "from app.rag_opt import DEMO_QUESTION; print(DEMO_QUESTION)"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.chroma_store import (
    PROJECT_ID,
    UNIT11_CHUNK_COLLECTION,
    UNIT11_WHOLE_COLLECTION,
    ChromaVectorStore,
    resolve_unit11_db_path,
)
from app.rag import RAGResult, RAGService
from app.rerank import render_order_change, render_ranked, rerank_hits
from app.retriever import SearchResult, RetrievalService

# 本组售后场景里真实会问的一句话（单元02 的工具场景与单元10 的问答都覆盖过）。
# 选它的原因不是顺口，而是实跑结果：它的直接证据（DOC-02-11 错发漏发）在整篇
# Top3 里不出现，只能靠分块进入候选池——这正好是任务卡要观察的现象。
DEMO_QUESTION = "收到的商品少了一件怎么办？"
# 直接支持回答的那一句原文所在的资料与片段位置；验收时据此核对有没有进候选。
DIRECT_EVIDENCE_SOURCE = "模拟数据/knowledge/DOC-02-11-错发漏发处理.md"
DIRECT_EVIDENCE_KEYWORD = "核对订单商品与收到商品的差异"

# 两组 K 值分别对应任务卡里的"整篇 Top3 / 分块召回 Top5、重排取前三"。
WHOLE_TOP_K = 3
CHUNK_TOP_K = 5
RERANK_KEEP = 3


@dataclass(frozen=True)
class OptimizedRun:
    """一次对照实验的全部可观察产物，字段足够写满验收证据。"""

    question: str
    document_total: int
    chunk_total: int
    baseline_hits: list[SearchResult]
    candidates: list[SearchResult]
    ranked: list[tuple]
    optimized_hits: list[SearchResult]
    before: RAGResult
    after: RAGResult

    @property
    def direct_evidence_in_baseline(self) -> bool:
        """直接证据有没有进入整篇检索的候选。"""
        return any(
            hit.source == DIRECT_EVIDENCE_SOURCE and DIRECT_EVIDENCE_KEYWORD in hit.text
            for hit in self.baseline_hits
        )

    @property
    def direct_evidence_in_candidates(self) -> bool:
        """直接证据有没有进入分块召回的候选池。"""
        return any(
            hit.source == DIRECT_EVIDENCE_SOURCE and DIRECT_EVIDENCE_KEYWORD in hit.text
            for hit in self.candidates
        )

    @property
    def order_changed(self) -> bool:
        """重排前后候选顺序是否变化（只比较留下的前 RERANK_KEEP 条）。"""
        before = [hit.document_id for hit in self.candidates[:RERANK_KEEP]]
        after = [hit.document_id for hit in self.optimized_hits]
        return before != after


class RAGOptimizer:
    """两版流水线：整篇检索 vs 分块 + 重排。"""

    def __init__(
        self,
        whole_retriever: RetrievalService,
        chunk_retriever: RetrievalService,
        llm,
        whole_top_k: int = WHOLE_TOP_K,
        chunk_top_k: int = CHUNK_TOP_K,
        rerank_keep: int = RERANK_KEEP,
    ):
        self.whole_retriever = whole_retriever
        self.chunk_retriever = chunk_retriever
        self.llm = llm
        self.whole_top_k = whole_top_k
        self.chunk_top_k = chunk_top_k
        self.rerank_keep = rerank_keep
        # 两版共用同一个 llm 对象，top_k 只影响各自默认的检索条数，不影响生成。
        self.baseline_service = RAGService(whole_retriever, llm, top_k=whole_top_k)
        self.optimized_service = RAGService(chunk_retriever, llm, top_k=chunk_top_k)

    def recall(
        self, question: str, project_id: str = PROJECT_ID
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """两版各自召回候选：整篇 Top3 与分块 Top5，问题完全相同。"""
        baseline = self.whole_retriever.search(question, project_id, top_k=self.whole_top_k)
        candidates = self.chunk_retriever.search(question, project_id, top_k=self.chunk_top_k)
        return baseline, candidates

    def rerank(self, question: str, candidates: list[SearchResult]) -> list[tuple]:
        """对候选逐对打分；返回 [(片段, 分数)]，按分数降序。"""
        return rerank_hits(question, candidates)

    def compare(
        self,
        question: str,
        project_id: str = PROJECT_ID,
        document_total: int = 0,
        chunk_total: int = 0,
    ) -> OptimizedRun:
        """跑完一整轮对照：召回 → 重排 → 取前三 → 两版各自回答。"""
        baseline_hits, candidates = self.recall(question, project_id)
        ranked = self.rerank(question, candidates)
        optimized_hits = [hit for hit, _ in ranked[: self.rerank_keep]]
        return OptimizedRun(
            question=question,
            document_total=document_total,
            chunk_total=chunk_total,
            baseline_hits=baseline_hits,
            candidates=candidates,
            ranked=ranked,
            optimized_hits=optimized_hits,
            # 两版都走 answer_from_hits：候选之外的一切（提示词、模型、核验）完全相同。
            before=self.baseline_service.answer_from_hits(question, baseline_hits),
            after=self.optimized_service.answer_from_hits(question, optimized_hits),
        )


def build_rag_optimizer(
    database_path: Path | None = None,
    records: list | None = None,
    llm=None,
) -> RAGOptimizer:
    """在单元11 自己的库里建两个集合并导入，返回可直接对照的流水线。

    整篇与子块各占一个集合：混在一个集合里，命中时说不清它是整篇还是片段。
    稳定 ID（来源 + 起止位置）保证重复导入只是更新，记录数不会增长。
    """
    from app.chunking import split_documents, split_whole_documents
    from app.ingest import load_markdown

    if llm is None:
        from app.llm import build_llm

        llm = build_llm()

    base = database_path or resolve_unit11_db_path()
    records = records if records is not None else load_markdown()
    whole_store = ChromaVectorStore(base, UNIT11_WHOLE_COLLECTION)
    chunk_store = ChromaVectorStore(base, UNIT11_CHUNK_COLLECTION)
    whole_store.upsert_documents(split_whole_documents(records))
    chunk_store.upsert_documents(split_documents(records))
    return RAGOptimizer(
        RetrievalService(whole_store), RetrievalService(chunk_store), llm
    )


def render_recall(run: OptimizedRun, preview: int = 52) -> str:
    """第2课要交的"候选来源与向量距离"：整篇 Top3 与分块 Top5 并列。"""
    lines = [f"问题：{run.question}"]
    lines.append(f"基础版（整篇检索 Top{WHOLE_TOP_K}，距离越小越近）：")
    for rank, hit in enumerate(run.baseline_hits, start=1):
        lines.append(f"  {rank}. 距离={hit.distance:.4f}  ID={hit.document_id}")
        lines.append(f"     来源={hit.source}")
        lines.append(f"     原文：{_preview(hit.text, preview)}")
    lines.append(f"优化版（重叠分块召回 Top{CHUNK_TOP_K}，距离越小越近）：")
    for rank, hit in enumerate(run.candidates, start=1):
        position = ":".join(hit.document_id.split(":")[-2:])
        lines.append(f"  {rank}. 距离={hit.distance:.4f}  ID={hit.document_id}  位置=[{position})")
        lines.append(f"     来源={hit.source}")
        lines.append(f"     原文：{_preview(hit.text, preview)}")
    return "\n".join(lines)


def render_evidence_check(run: OptimizedRun) -> str:
    """直接证据进入候选的路径：这是任务卡要求说明的那一条。"""
    lines = [
        f"直接证据定位：{DIRECT_EVIDENCE_SOURCE}",
        f"关键句包含：{DIRECT_EVIDENCE_KEYWORD}",
        f"整篇检索 Top{WHOLE_TOP_K} 中是否出现：{'是' if run.direct_evidence_in_baseline else '否'}",
        f"分块召回 Top{CHUNK_TOP_K} 中是否出现：{'是' if run.direct_evidence_in_candidates else '否'}",
    ]
    if run.direct_evidence_in_candidates and not run.direct_evidence_in_baseline:
        lines.append("结论：直接证据是靠分块才进入候选池的，整篇检索这一题根本没召回它。")
    elif run.direct_evidence_in_baseline:
        lines.append("结论：直接证据在整篇检索里就能召回，这一题观察不到分块带来的增量。")
    return "\n".join(lines)


def render_answers(run: OptimizedRun) -> str:
    """第4课要交的"两版回答及引用"：并排展示，便于核对哪一段直接支持回答。"""
    lines = [f"基础版回答：{run.before.answer}"]
    lines.append(f"  引用编号：{run.before.cited if run.before.cited else '无'}")
    lines.append(f"  来源：{'、'.join(run.before.sources) if run.before.sources else '无'}")
    if run.before.error:
        lines.append(f"  异常：{run.before.error}")
    lines.append(f"优化版回答：{run.after.answer}")
    lines.append(f"  引用编号：{run.after.cited if run.after.cited else '无'}")
    lines.append(f"  来源：{'、'.join(run.after.sources) if run.after.sources else '无'}")
    if run.after.error:
        lines.append(f"  异常：{run.after.error}")
    return "\n".join(lines)


def render_run(run: OptimizedRun) -> str:
    """把一次对照实验渲染成可贴进验收证据的完整文本。"""
    blocks = [
        f"知识文档：{run.document_total} 篇；重叠分块：{run.chunk_total} 块",
        render_recall(run),
        "",
        render_ranked(run.ranked),
        "",
        render_order_change(run.candidates[: RERANK_KEEP], run.optimized_hits),
        "",
        render_evidence_check(run),
        "",
        render_answers(run),
    ]
    return "\n".join(blocks)


def _preview(text: str, preview: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= preview else flat[:preview] + "……"
