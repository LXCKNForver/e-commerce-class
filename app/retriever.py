"""单元09：把单元08 的 Chroma 查询封装成可复用的检索服务。

单元08 只演示"写进去、查一次"；本单元把它收敛成两个接口：

    service.search(question, project_id, top_k)          -> 单问题
    service.search_many([q1, q2], project_id, top_k)     -> 批量

两者返回同一种稳定结构 `SearchResult`，排序与过滤规则完全一致（单条接口直接
委托批量接口实现），后续单元（RAG）可以直接拿它当"取证据"的入口。

三个必须记住的点：
1. **project_id 是必选的**。它是项目隔离的边界，漏掉就会拿到别的项目的数据。
2. **distance 是距离，不是相似度、也不是正确率**。距离越小越接近。
3. **本单元只返回证据，不生成回答**。命中不等于答对，生成留给后续单元。

    uv run --extra embedding --extra chroma python -c "from app.retriever import build_retrieval_service as b; print(b().search('商品有质量问题可以退款吗？','e-commerce-after-sales')[0].source)"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.chroma_store import (
    DEFAULT_UNIT9_DB_PATH,
    PROJECT_ID,
    ChromaVectorStore,
    resolve_unit9_db_path,
)

# 本组售后场景里真实会问的两句话（第二句来自 模拟数据/sample_questions.json 的 Q-02-04）。
DEMO_QUESTION = "商品有质量问题可以退款吗？"
SECOND_QUESTION = "我收到的商品破损了"
# 来源过滤演示：指定这一份资料时，只允许返回它。
DEMO_SOURCE = "模拟数据/knowledge/DOC-02-05-定制商品例外.md"


@dataclass(frozen=True)
class SearchResult:
    """一次命中的稳定返回结构。

    distance 由 Chroma 按集合默认指标（l2，平方欧氏距离）给出，**越小越接近**；
    similarity 是单元06 的余弦相似度，**越大越接近**，两者都不是正确率。
    """

    document_id: str
    text: str
    source: str
    project_id: str
    doc_type: str
    distance: float
    similarity: float | None = None


class RetrievalService:
    """只读检索服务：对外只暴露 search / search_many 两个入口。"""

    def __init__(self, store: ChromaVectorStore):
        self.store = store

    def search(
        self,
        question: str,
        project_id: str = PROJECT_ID,
        top_k: int = 3,
        source: str | None = None,
        doc_type: str | None = None,
    ) -> list[SearchResult]:
        """单问题检索；直接走批量接口，避免两处规则各写一遍。"""
        return self.search_many([question], project_id, top_k, source, doc_type)[0]

    def search_many(
        self,
        questions: list[str],
        project_id: str = PROJECT_ID,
        top_k: int = 3,
        source: str | None = None,
        doc_type: str | None = None,
    ) -> list[list[SearchResult]]:
        """批量检索；返回列表与输入问题一一对应，顺序不变。"""
        batches = self.store.query_many(questions, project_id, top_k, doc_type, source)
        return [
            [
                SearchResult(
                    document_id=hit["id"],
                    text=hit["text"],
                    source=hit["source"],
                    project_id=hit["project_id"],
                    doc_type=hit["doc_type"],
                    distance=hit["distance"],
                    similarity=hit.get("similarity"),
                )
                for hit in batch
            ]
            for batch in batches
        ]


def build_retrieval_service(
    database_path: Path | None = None,
    records: list | None = None,
) -> RetrievalService:
    """建好单元09 自己的库并完成首次导入，返回可直接查询的服务。

    任务卡要求"数据库保存在本单元代码目录，不复制第八单元数据库"，所以库目录是
    `.chroma_unit9/`；稳定 ID 保证重复导入只是更新，记录数不会增长。
    """
    from app.ingest import load_markdown

    store = ChromaVectorStore(database_path or resolve_unit9_db_path())
    store.upsert_documents(records if records is not None else load_markdown())
    return RetrievalService(store)


def render_results(label: str, results: list[SearchResult], preview: int = 48) -> str:
    """把一组命中渲染成可贴进验收证据的文本；原文过长时只截断展示。"""
    lines = [f"{label}：{len(results)} 条"]
    if not results:
        lines.append("  未检索到符合过滤条件的资料")
    for rank, hit in enumerate(results, start=1):
        lines.append(f"  {rank}. 距离={hit.distance:.4f}  ID={hit.document_id}")
        lines.append(f"     来源={hit.source}")
        lines.append(f"     项目={hit.project_id}  类型={hit.doc_type}")
        snippet = " ".join(hit.text.split())[:preview]
        lines.append(f"     原文片段：{snippet}")
    return "\n".join(lines)
