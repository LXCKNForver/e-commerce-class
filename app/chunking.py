"""单元11：把整篇资料切成有重叠的子块，并保留它在原文里的位置。

前面几个单元都是"一篇资料 = 一条向量"。整篇入库有两个实际问题：
1. 一条向量要代表 200 多个字符里的全部内容，主题被平均掉，细节问法往往召回不到；
2. 句子被切在窗口边界上时，边界附近的信息两边都不完整。

本单元的处理办法是把整篇按固定字符窗口切开，相邻两块留一段重叠，
让边界附近的信息有机会同时出现在两个候选里。位置（start/end）随记录一起保存，
课堂上要能直接指出"第 2 块从原文第 60 个字符开始"。

窗口参数是**教学参数**：80 字符只是方便在屏幕上数清楚，不等于 token 长度，
真实业务里要按 token 切并按段落边界对齐。

    uv run --extra embedding python -c "from app.chunking import split_documents, CHUNK_SIZE; from app.ingest import load_markdown; print(len(split_documents(load_markdown(), CHUNK_SIZE)))"
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingest import KnowledgeDocument

# 任务卡给定的课堂教学参数：80 字符窗口、20 字符重叠。
CHUNK_SIZE = 80
CHUNK_OVERLAP = 20
# 第二块的起点 = 80 - 20 = 60，课堂上直接用来核对"重叠从哪开始"。
SECOND_CHUNK_START = CHUNK_SIZE - CHUNK_OVERLAP
# 分块观察默认看这一份资料：它就是单元11 演示题的直接证据所在文档，
# 切分后还能顺带看出「完整句子被切成两块」——字符窗口不保证句子完整。
DEMO_SOURCE = "模拟数据/knowledge/DOC-02-11-错发漏发处理.md"


@dataclass(frozen=True)
class TextChunk:
    """一个可检索的子块：身份 + 片段正文 + 来源 + 项目标识 + 原文位置。"""

    document_id: str
    text: str
    source: str
    project_id: str
    doc_type: str
    start: int
    end: int

    @property
    def metadata(self) -> dict[str, object]:
        """写入 Chroma 的元数据；起止位置跟着走，命中后能指出它来自原文哪一段。"""
        return {
            "source": self.source,
            "project_id": self.project_id,
            "doc_type": self.doc_type,
            "start": self.start,
            "end": self.end,
        }

    @property
    def position(self) -> str:
        """原文位置，左闭右开：[0,80) 表示第 0 到第 79 个字符。"""
        return f"[{self.start},{self.end})"


def split_documents(
    records: list[KnowledgeDocument],
    chunk_size: int | None = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """把整篇资料切成子块；chunk_size=None 表示整篇一条，用于对照组。

    start/end 是原文的字符下标，左闭右开，检索命中后据此定位片段。
    """
    if chunk_size is None:
        if overlap:
            raise ValueError("整篇入库时不设置重叠，overlap 只能是 0")
    elif not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数")
    elif not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < chunk_size:
        raise ValueError("需要满足 0 <= overlap < chunk_size，否则窗口不会向前推进")
    if not records:
        raise ValueError("待切分的资料列表不能为空")

    chunks: list[TextChunk] = []
    for record in records:
        if not isinstance(record.text, str) or not record.text.strip():
            raise ValueError(f"资料正文为空，无法切分：{record.source}")
        size = chunk_size or len(record.text)
        start = 0
        while start < len(record.text):
            end = min(start + size, len(record.text))
            # ID 里带上起止位置：同参数重复切分得到同一批 ID，重复入库只是更新。
            chunks.append(
                TextChunk(
                    document_id=f"{record.document_id}:{start}:{end}",
                    text=record.text[start:end],
                    source=record.source,
                    project_id=record.project_id,
                    doc_type=record.doc_type,
                    start=start,
                    end=end,
                )
            )
            if end == len(record.text):
                break
            # 关键一步：下一块回退 overlap 个字符再开始，于是相邻两块共享一段内容。
            start = end - overlap
    return chunks


def split_whole_documents(records: list[KnowledgeDocument]) -> list[TextChunk]:
    """对照组：整篇一条，与单元08—10 的入库方式完全一致。"""
    return split_documents(records, chunk_size=None, overlap=0)


def chunks_of(chunks: list[TextChunk], source: str) -> list[TextChunk]:
    """取出某一篇资料的全部子块，保持原文顺序。"""
    return [chunk for chunk in chunks if chunk.source == source]


def overlap_between(previous: TextChunk, current: TextChunk) -> str:
    """相邻两块字面重复的那一段内容；没有重叠时返回空字符串。"""
    length = previous.end - current.start
    if length <= 0:
        return ""
    return current.text[:length]


def render_chunk_positions(chunks: list[TextChunk], limit: int = 6) -> str:
    """第1课要交的"相邻块起止位置"：第几块、覆盖原文哪些字符、与上一块重叠什么。"""
    if not chunks:
        return "分块观察：没有可展示的片段"

    source = chunks[0].source
    lines = [
        f"分块观察（资料：{source}）",
        f"窗口 {CHUNK_SIZE} 字符，重叠 {CHUNK_OVERLAP} 字符；第二块从第 {SECOND_CHUNK_START} 个字符开始",
        "序号\t片段ID\t位置\t字符数\t与上一块重叠",
    ]
    for index, chunk in enumerate(chunks[:limit], start=1):
        if index == 1:
            overlap_text = "—（首块）"
        else:
            shared = overlap_between(chunks[index - 2], chunk)
            overlap_text = f"{len(shared)} 字符：{' '.join(shared.split())}"
        lines.append(
            f"{index}\t{chunk.document_id}\t{chunk.position}\t{len(chunk.text)}\t{overlap_text}"
        )
    if len(chunks) > limit:
        lines.append(f"…… 其余 {len(chunks) - limit} 块同上，位置连续且相邻两块共享 {CHUNK_OVERLAP} 字符。")
    return "\n".join(lines)


def render_chunk_texts(chunks: list[TextChunk], limit: int = 4, preview: int = 60) -> str:
    """片段正文：核对切出来的是否还能独立支撑一次回答。"""
    lines = [f"片段正文（前 {min(limit, len(chunks))} 块）："]
    for index, chunk in enumerate(chunks[:limit], start=1):
        lines.append(f"  {index}. {chunk.document_id} {chunk.position}")
        lines.append(f"     正文：{_preview(chunk.text, preview)}")
    return "\n".join(lines)


def _preview(text: str, preview: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= preview else flat[:preview] + "……"
