"""单元07：用可读的内存实现讲清向量库的数据模型与精确检索。

这是**教学基线**，不是真正的向量数据库：全量遍历、精确排序，
没有磁盘持久化、没有 HNSW 索引、没有并发与权限。
真正的 Chroma 集合由 `setup_chroma.py` 创建，本单元只建空集合、不写入数据。

一条记录由四部分组成，正好对应任务卡里的字段图：
    document_id  身份      —— 这条记录是谁（DOC-02-05 / Q-02-01）
    text         原文      —— 送进向量模型的那段话
    source       来源      —— 出自哪个文件，可用于过滤
    doc_type     元数据    —— 知识文档还是样例问题，用于过滤
    vector       向量      —— 单元06 那个模型算出的 512 维数字
    tenant_id    项目隔离  —— 防止跨项目返回记录
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.data_store import KNOWLEDGE_DIR
from app.embedding import EXPECTED_DIM, cosine_similarity, embed
from app.vector_lab import SAMPLE_FILE

# 本组项目的隔离标识；查询必须带上它，避免把别的项目的数据混进来。
TENANT_ID = "e-commerce-after-sales"
# 自拟查询：售后场景里真实会问的一句话。
DEMO_QUERY = "商品有质量问题可以退款吗？"

FIELD_DIAGRAM = [
    ("document_id", "身份", "DOC-02-05 / Q-02-01", "主键，唯一"),
    ("text", "原文", "定制商品不适用普通 7 天退货规则……", "送进 Embedding 的那段文字"),
    ("source", "来源", "模拟数据/knowledge/DOC-02-05-定制商品例外.md", "可按来源过滤"),
    ("doc_type", "元数据", "knowledge / question", "可按类型过滤"),
    ("vector", "向量", f"{EXPECTED_DIM} 个浮点数", "单元06 同一模型产出，检索靠它"),
    ("tenant_id", "项目隔离", TENANT_ID, "查询必须指定，防止串项目"),
]


@dataclass(frozen=True)
class VectorRecord:
    """一条向量库记录：身份 + 原文 + 元数据 + 向量。"""

    document_id: str
    text: str
    source: str
    doc_type: str
    tenant_id: str
    vector: list[float]


@dataclass(frozen=True)
class SearchHit:
    """查询结果 = 原记录 + 本次相似度。"""

    record: VectorRecord
    score: float


def render_field_diagram() -> str:
    """把一条记录的字段图渲染成可直接贴进课堂材料的文本。"""
    lines = ["一条向量记录的字段图：", "字段\t含义\t示例\t作用"]
    for name, meaning, example, purpose in FIELD_DIAGRAM:
        lines.append(f"{name}\t{meaning}\t{example}\t{purpose}")
    return "\n".join(lines)


def _first_paragraph(markdown: str) -> str:
    """取出 `## 规则与说明` 后的第一段，作为这条文档的可检索正文。"""
    body = markdown.split("## 规则与说明", 1)
    text = body[1] if len(body) > 1 else markdown
    for chunk in text.split("## "):
        if chunk.strip():
            return chunk.strip().splitlines()[0].strip()
    return ""


def load_knowledge_texts(directory: Path | None = None) -> list[dict[str, str]]:
    """读取售后知识文档标题与规则首句，作为知识库候选文本。"""
    root = directory or KNOWLEDGE_DIR
    if not root.is_dir():
        raise FileNotFoundError(f"缺少知识文档目录：{root}")

    items: list[dict[str, str]] = []
    for path in sorted(root.glob("DOC-*.md")):
        doc_id = "-".join(path.stem.split("-")[:3])
        markdown = path.read_text(encoding="utf-8")
        title = markdown.lstrip().lstrip("#").strip().splitlines()[0].strip()
        rule = _first_paragraph(markdown)
        if not rule:
            continue
        items.append(
            {
                "document_id": doc_id,
                "text": f"{title}（{doc_id}）：{rule}",
                "source": f"模拟数据/knowledge/{path.name}",
                "doc_type": "knowledge",
            }
        )
    if len(items) < 5:
        raise ValueError(f"知识文档数量不足（{len(items)} 份），无法说明数据规模")
    return items


def load_question_texts(path: Path | None = None) -> list[dict[str, str]]:
    """读取单元06 用过的售后样例问题，标为 question 类型。"""
    from app.vector_lab import load_sample_texts

    return [
        {
            "document_id": item.source_id,
            "text": item.text,
            "source": item.source,
            "doc_type": "question",
        }
        for item in load_sample_texts(path)
    ]


def build_demo_records(
    tenant_id: str = TENANT_ID,
    question_path: Path | None = None,
) -> list[dict[str, str]]:
    """本组选题的全部候选文本：15 份知识文档 + 6 条样例问题。"""
    return load_knowledge_texts() + load_question_texts(question_path)


class InMemoryVectorStore:
    """教学用集合：全量遍历、精确排序，不提供持久化或 ANN 索引。"""

    def __init__(self, collection_name: str, dimension: int = EXPECTED_DIM):
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("集合名称不能为空")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("向量维度必须是正整数")
        self.collection_name = collection_name
        self.dimension = dimension
        self.records: list[VectorRecord] = []

    def add(
        self,
        document_id: str,
        text: str,
        source: str,
        doc_type: str,
        tenant_id: str,
    ) -> VectorRecord:
        """写入一条记录；同一项目内的 document_id 不允许重复。"""
        fields = (document_id, text, source, doc_type, tenant_id)
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise ValueError("document_id、text、source、doc_type、tenant_id 都不能为空")
        if any(r.tenant_id == tenant_id and r.document_id == document_id for r in self.records):
            raise ValueError(f"项目 {tenant_id} 中的文档 ID 已存在：{document_id}")

        # 先算向量再入库：维度必须与集合声明一致，否则后面没法两两比较。
        vector = embed(text)
        if len(vector) != self.dimension:
            raise ValueError(f"集合要求 {self.dimension} 维，模型输出 {len(vector)} 维")

        record = VectorRecord(document_id, text, source, doc_type, tenant_id, vector)
        self.records.append(record)
        return record

    def add_many(self, items: list[dict[str, str]], tenant_id: str = TENANT_ID) -> int:
        """批量写入，返回成功条数；任一条不合法就整体报错，不写半套数据。"""
        for item in items:
            self.add(
                item["document_id"],
                item["text"],
                item["source"],
                item["doc_type"],
                tenant_id,
            )
        return len(items)

    def count(self, tenant_id: str | None = None, doc_type: str | None = None) -> int:
        """统计记录数，可按项目与类型过滤。"""
        return sum(
            (tenant_id is None or r.tenant_id == tenant_id)
            and (doc_type is None or r.doc_type == doc_type)
            for r in self.records
        )

    def query(
        self,
        text: str,
        tenant_id: str,
        top_k: int = 3,
        source: str | None = None,
        doc_type: str | None = None,
    ) -> list[SearchHit]:
        """先按项目/来源/类型过滤，再逐条算余弦相似度并返回前 k 条。"""
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("查询必须指定 tenant_id，避免跨项目返回记录")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("查询文本不能为空")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        for name, value in (("source", source), ("doc_type", doc_type)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} 过滤条件不能为空字符串")

        # 查询用单元06 同一个模型；换了模型或维度，历史向量就不可比了。
        query_vector = embed(text)
        if len(query_vector) != self.dimension:
            raise ValueError("查询向量维度与集合不一致")

        candidates = [
            r
            for r in self.records
            if r.tenant_id == tenant_id
            and (source is None or r.source == source)
            and (doc_type is None or r.doc_type == doc_type)
        ]
        # 这里是精确遍历：每个候选都算一遍分数。数据量一大就会变慢，
        # 这正是下一层要用 HNSW/IVF 近似索引换速度的原因。
        hits = [SearchHit(r, cosine_similarity(query_vector, r.vector)) for r in candidates]
        # 分数降序；同分时按 document_id 排序，保证结果可复现。
        return sorted(hits, key=lambda hit: (-hit.score, hit.record.document_id))[:top_k]
