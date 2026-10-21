"""单元08：把本组真实售后资料整理成可写入 Chroma 的记录。

沿用课程已有的 `模拟数据/knowledge/`，不另建数据目录；一篇 Markdown 一条记录。
关键在**稳定 ID**：重复运行时它让写入变成"更新同一条"，而不是"又追加一条"。

    uv run --extra embedding python -c "from app.ingest import load_markdown; print(len(load_markdown()))"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.data_store import KNOWLEDGE_DIR
from app.vector_store import TENANT_ID

# 只收本组售后知识文档，避免把别的说明文件也误当成资料写进去。
DOC_GLOB = "DOC-*.md"
# 任务卡要求"至少 5 篇本组资料"；本组现有 15 篇，低于这个数说明数据目录被改动过。
MIN_DOCUMENTS = 5


@dataclass(frozen=True)
class KnowledgeDocument:
    """一条待写入的记录：身份 + 原文 + 来源 + 项目标识 + 资料类型。"""

    document_id: str
    text: str
    source: str
    project_id: str
    doc_type: str = "knowledge"

    @property
    def metadata(self) -> dict[str, str]:
        """写入 Chroma 的元数据：检索时可按来源、项目、类型过滤。"""
        return {
            "source": self.source,
            "project_id": self.project_id,
            "doc_type": self.doc_type,
        }


def load_markdown(
    folder: Path | None = None,
    project_id: str = TENANT_ID,
) -> list[KnowledgeDocument]:
    """读取目录里的售后知识文档，一篇一条，ID 稳定且带项目前缀。"""
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project_id 不能为空")

    root = folder if folder is not None else KNOWLEDGE_DIR
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"知识资料目录不存在：{root}")

    files = sorted(root.glob(DOC_GLOB))
    if not files:
        raise ValueError(f"目录中没有 Markdown 资料：{root}")

    records: list[KnowledgeDocument] = []
    for file in files:
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"知识资料不能为空：{file}")
        # 文件名形如 DOC-02-05-定制商品例外.md，取前三段得到稳定编号 DOC-02-05。
        doc_id = "-".join(file.stem.split("-")[:3])
        records.append(
            KnowledgeDocument(
                # 带上项目前缀：换项目导入时 ID 不会撞名。
                document_id=f"{project_id}:{doc_id}",
                text=text,
                source=f"模拟数据/knowledge/{file.name}",
                project_id=project_id,
            )
        )

    if len(records) < MIN_DOCUMENTS:
        raise ValueError(
            f"本组资料只有 {len(records)} 篇，少于要求的 {MIN_DOCUMENTS} 篇"
        )
    return records


def render_inventory(records: list[KnowledgeDocument], limit: int = 5) -> str:
    """第1课要交的"输入资料清单"：来源、稳定 ID、正文字数。"""
    lines = [
        f"输入资料清单（共 {len(records)} 篇，来源：模拟数据/knowledge/）：",
        "序号\t稳定ID\t来源\t正文字数",
    ]
    for index, record in enumerate(records[:limit], start=1):
        lines.append(
            f"{index}\t{record.document_id}\t{record.source}\t{len(record.text)}"
        )
    if len(records) > limit:
        lines.append(f"…… 其余 {len(records) - limit} 篇同上，ID 与来源一一对应。")
    return "\n".join(lines)
