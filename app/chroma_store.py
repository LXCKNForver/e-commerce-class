"""单元08：用真实的 Chroma 集合写入并检索本组售后资料。

承接单元07 建好的空集合（同名、同元数据、同向量模型），把 `模拟数据/knowledge/`
里的售后文档批量 upsert 进去，再重新连接数据库做一次项目内 Top-K 检索。

两个必须记住的点：
1. **写入和查询必须用同一个向量模型**。换模型等于换向量空间，历史向量全部作废，
   所以连接时会核对集合上记录的 `embedding_model`。
2. **检索返回的字段是"距离"，不是相似度，也不是正确率**。距离越小越接近；
   本单元只做检索，不生成任何业务答案。

    uv run --extra embedding --extra chroma python -c "from app.chroma_store import ChromaVectorStore; print(ChromaVectorStore().count())"
"""

from __future__ import annotations

import os
from pathlib import Path

# 与单元07 完全一致：沿用那个空集合，不另起一个。
COLLECTION_NAME = "after_sale_knowledge_base"
COLLECTION_METADATA = {
    "purpose": "ecommerce_after_sale_knowledge_base",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "dimension": 512,
    "content_source": "模拟数据/knowledge/ + 模拟数据/sample_questions.json",
    "write_policy": "read_only_sample_data",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 单元07 的 .chroma_unit7/ 保持 0 条记录的验收现场，本单元写到自己的目录；
# 想直接沿用单元07 那个库文件时，设 CHROMA_DB_DIR=.chroma_unit7 即可。
DEFAULT_DB_PATH = PROJECT_ROOT / ".chroma_unit8"

# 项目隔离标识沿用单元07；查询必须带上它，否则会拿到别的项目的数据。
PROJECT_ID = "e-commerce-after-sales"
# 本组售后场景里真实会问的一句话（与单元07 内存基线同一个问题，便于对照）。
DEMO_QUESTION = "商品有质量问题可以退款吗？"

RECORD_FIELDS = [
    ("document_id", "稳定 ID", f"{PROJECT_ID}:DOC-02-05", "重复运行靠它更新，不追加"),
    ("text", "原文", "# 定制商品例外\\n\\n定制商品不适用……", "整篇 Markdown，送进向量模型"),
    ("source", "来源", "模拟数据/knowledge/DOC-02-05-定制商品例外.md", "命中后据此核对出处"),
    ("doc_type", "资料类型", "knowledge", "可与项目标识一起过滤"),
    ("project_id", "项目隔离", PROJECT_ID, "查询必须指定，防止串项目"),
    ("vector", "向量", "512 个浮点数", "单元06 同一模型产出，检索靠它"),
]


def resolve_db_path() -> Path:
    """数据库目录：默认 `.chroma_unit8/`，可用 CHROMA_DB_DIR 覆盖。"""
    override = os.environ.get("CHROMA_DB_DIR", "").strip()
    return Path(override) if override else DEFAULT_DB_PATH


def render_record_fields() -> str:
    """第1课要交的"记录字段图"：一条写进 Chroma 的记录由哪些部分组成。"""
    lines = ["写入 Chroma 的一条记录由这些部分组成：", "字段\t含义\t示例\t作用"]
    for name, meaning, example, purpose in RECORD_FIELDS:
        lines.append(f"{name}\t{meaning}\t{example}\t{purpose}")
    return "\n".join(lines)


class ChromaVectorStore:
    """真实 Chroma 集合的薄封装：只暴露写入、计数、检索三件事。"""

    def __init__(
        self,
        database_path: Path | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - 依赖缺失时才会走这里
            raise RuntimeError(
                "缺少 chromadb。请执行：uv sync --extra chroma"
            ) from exc

        from app.embedding import MODEL_NAME

        self.database_path = Path(database_path) if database_path else resolve_db_path()
        # PersistentClient 把数据落到指定目录；重新连接只是再打开同一个目录。
        self.client = chromadb.PersistentClient(path=str(self.database_path))
        # 同名集合已存在就复用，不存在才创建；不传 embedding_function，
        # 向量一律由单元06 的本地中文模型算好再送进来。
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata=COLLECTION_METADATA,
        )
        # 复用已有集合时仍要核对模型标识，避免新旧向量混在一个集合里。
        recorded = (self.collection.metadata or {}).get("embedding_model")
        if recorded != MODEL_NAME:
            raise ValueError(
                f"集合记录的向量模型是 {recorded}，不是 {MODEL_NAME}，无法复用该集合"
            )

    def count(self) -> int:
        return self.collection.count()

    def upsert_documents(self, records: list) -> int:
        """批量写入；同一 ID 再次写入会更新原记录，不会新增一条。"""
        from app.embedding import embed_many

        if not records:
            raise ValueError("不能导入空记录列表")

        texts = [record.text for record in records]
        # 真正的写入就这一行：ID、原文、向量、元数据一起交给 Chroma。
        # 向量来自单元06 的同一个模型——这就是"同模型写入"的含义。
        self.collection.upsert(
            ids=[record.document_id for record in records],
            documents=texts,
            embeddings=embed_many(texts),
            metadatas=[record.metadata for record in records],
        )
        return len(records)

    def query(
        self,
        question: str,
        project_id: str = PROJECT_ID,
        top_k: int = 3,
        doc_type: str | None = None,
    ) -> list[dict]:
        """用同一模型把问题变成向量，只在指定项目的记录里找最接近的 top_k 条。"""
        from app.embedding import cosine_similarity, embed

        if not isinstance(question, str) or not question.strip():
            raise ValueError("查询问题不能为空")
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("查询必须指定 project_id，避免跨项目返回记录")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        if doc_type is not None and (not isinstance(doc_type, str) or not doc_type.strip()):
            raise ValueError("doc_type 过滤条件不能为空字符串")
        if self.count() == 0:
            return []

        where: dict = {"project_id": project_id}
        if doc_type:
            where = {"$and": [{"project_id": project_id}, {"doc_type": doc_type}]}

        # 问题也用单元06 同一个模型编码——这就是"同模型查询"的含义。
        query_vector = embed(question)
        # 距离由 Chroma 在库内按集合的距离指标算好返回，这里不再自己写距离公式。
        # 集合新建时未指定指标，沿用 Chroma 默认的 l2（平方欧氏距离），越小越接近。
        result = self.collection.query(
            query_embeddings=[query_vector],
            where=where,
            # 请求条数不能超过库里已有的条数，否则 Chroma 会直接报错。
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        hits: list[dict] = []
        for document_id, text, metadata, distance, vector in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["embeddings"][0],
        ):
            hits.append(
                {
                    "id": document_id,
                    "text": text,
                    "source": (metadata or {}).get("source", ""),
                    "doc_type": (metadata or {}).get("doc_type", ""),
                    "distance": float(distance),
                    # 单元06 的余弦相似度一并给出，方便和距离对照着看。
                    "similarity": cosine_similarity(query_vector, list(vector)),
                }
            )
        return hits
