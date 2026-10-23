"""单元11：用本地 BGE Cross-Encoder 对召回候选重新排序。

单元06—10 的排序只有一个依据：向量距离。它是"问题向量"和"资料向量"各自算出来
再比对的（双塔），两条文本在变成向量之前从没见过面。

本单元加的是 Cross-Encoder：把"原问题"和"候选片段"拼成一对一起送进模型，
模型在同一个注意力过程里判断它们是否相关，直接输出相关性 logit。

三个必须讲清楚的边界：
1. **BGE 输出的是相关性 logit，越大越相关**；Chroma 给的是平方 L2 距离，越小越近。
   两者方向相反、量纲不同，**不能混算，也不能拿它当概率**。
2. **重排只能改变已经召回的候选顺序，不能找回候选池之外的证据**。
   召回（向量）定上限，重排（BGE）只负责把上限内的顺序排对。
3. **必须先配对再排序**：rerank 返回的分数顺序与输入文档相同，
   拿到分数就丢下原文去排序，会把片段和分数配错。

    uv run --extra embedding python -c "from app.rerank import rerank_ready; print(rerank_ready())"
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

# 与单元06 的向量模型配套的中文重排模型，随课程材料发放，本单元全部本地加载。
RERANKER_NAME = "BAAI/bge-reranker-base"
# 模型权重 1.1GB 左右，不进仓库；用 RERANK_MODEL_DIR 指向课程材料里的目录。
MODEL_FILE = "onnx/model.onnx"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RERANK_MODEL_DIR = PROJECT_ROOT / "models"
# 一次送进模型的候选对数；小课堂上 4 对一批足够，太大只会白等。
RERANK_BATCH_SIZE = 4


def resolve_rerank_model_dir() -> Path:
    """确定重排模型目录：优先读环境变量，其次用项目内的 models/。

    模型体积大，不随仓库分发；换电脑时用 RERANK_MODEL_DIR 指向课程材料目录即可。
    """
    override = os.environ.get("RERANK_MODEL_DIR", "").strip()
    return Path(override) if override else DEFAULT_RERANK_MODEL_DIR


def _local_model_path() -> Path:
    return resolve_rerank_model_dir() / "bge-reranker-base"


@lru_cache(maxsize=1)
def _model():
    """加载本地 Cross-Encoder；缺权重直接报错，不联网下载、不退回字符串打分。"""
    local_path = _local_model_path()
    if not (local_path / MODEL_FILE).is_file():
        raise FileNotFoundError(
            f"缺少本地重排模型：{local_path / MODEL_FILE}。"
            "请将课程材料里的 bge-reranker-base 目录放到 models/ 下，"
            "或在 .env 里用 RERANK_MODEL_DIR 指向它。"
        )
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:  # pragma: no cover - 依赖缺失时才会走这里
        raise RuntimeError("缺少 fastembed。请执行：uv sync --extra embedding") from exc

    return TextCrossEncoder(
        RERANKER_NAME,
        cache_dir=str(resolve_rerank_model_dir()),
        local_files_only=True,
        threads=4,
        specific_model_path=str(local_path),
    )


def load_reranker():
    """提前加载模型：讲到一半才报"模型缺失"最影响课堂节奏。"""
    return _model()


def rerank_hits(question: str, hits, batch_size: int = RERANK_BATCH_SIZE) -> list[tuple]:
    """给候选片段重新打分，返回 [(片段, 分数)]，按分数降序排列。

    片段与分数必须成对出现：先 zip 回原始候选，再排序，顺序不能反过来。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空")
    if not hits:
        return []

    scores = [
        float(score)
        for score in _model().rerank(question, [hit.text for hit in hits], batch_size=batch_size)
    ]
    if len(scores) != len(hits):
        raise RuntimeError(f"重排分数数量（{len(scores)}）与候选数量（{len(hits)}）不一致")
    for index, score in enumerate(scores):
        if not math.isfinite(score):
            raise RuntimeError(f"第 {index + 1} 个候选的重排分数异常：{score}")

    ranked = list(zip(hits, scores))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def top_hits(question: str, hits, keep: int = 3) -> list:
    """重排后取前 keep 个片段；只丢弃片段，分数由调用方另行展示。"""
    return [hit for hit, _ in rerank_hits(question, hits)[:keep]]


def rerank_ready() -> tuple[bool, str]:
    """模型是否可用（供主程序与测试给出友好提示，而不是抛一长串堆栈）。"""
    try:
        _model()
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        return False, str(exc)
    return True, RERANKER_NAME


def render_ranked(ranked: list[tuple], preview: int = 44) -> str:
    """把重排结果渲染成可贴进验收证据的文本；分数与片段并排，方便核对配对。"""
    lines = [f"BGE重排（{RERANKER_NAME}，logit 越大越相关，共 {len(ranked)} 个候选）："]
    if not ranked:
        lines.append("  没有候选可重排")
    for rank, (hit, score) in enumerate(ranked, start=1):
        lines.append(f"  {rank}. 分数={score:+.4f}  ID={hit.document_id}  距离={hit.distance:.4f}")
        lines.append(f"     来源={hit.source}")
        lines.append(f"     片段：{' '.join(hit.text.split())[:preview]}")
    return "\n".join(lines)


def render_order_change(before: list, after: list) -> str:
    """重排前后顺序是否变化：用片段 ID 序列对比，避免只看前几条得出错觉。"""
    before_ids = [hit.document_id for hit in before]
    after_ids = [hit.document_id for hit in after]
    changed = before_ids != after_ids
    lines = [
        f"重排前顺序：{' -> '.join(before_ids) if before_ids else '（无）'}",
        f"重排后顺序：{' -> '.join(after_ids) if after_ids else '（无）'}",
        f"顺序是否变化：{'是' if changed else '否'}",
        f"候选集合是否变化：{'否（重排只改顺序，不增删候选）' if set(before_ids) == set(after_ids) else '是'}",
    ]
    return "\n".join(lines)
