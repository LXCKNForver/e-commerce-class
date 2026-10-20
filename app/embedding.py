"""单元06：用本地中文模型把真实文本转换成可比较的向量。

本文件只做三件事：把文本变成向量、算两个向量的余弦相似度、把异常说清楚。
它不读知识库、不做问答、不做业务判断——第4节课要讨论的"向量存哪里"留给下一单元。

模型是随课发放的 `BAAI/bge-small-zh-v1.5`（Qdrant 发布的 ONNX 版），
放在 `models/` 目录里离线加载，课堂上不联网、不调用 DeepSeek。
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

# 全班统一模型，换模型就等于换向量空间，历史向量全部作废，所以写死在这里。
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# 该模型输出 512 维；用它做一次自检，维度变了说明模型被换过。
EXPECTED_DIM = 512

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


def resolve_model_dir() -> Path:
    """确定模型目录：优先读环境变量，其次用项目内的 models/。

    模型有 180MB 左右，不适合提交进 Git，所以不随仓库分发。
    换电脑时用 `EMBEDDING_MODEL_DIR` 指向课程材料里的 models/ 即可。
    """
    override = os.environ.get("EMBEDDING_MODEL_DIR", "").strip()
    if override:
        return Path(override)
    return DEFAULT_MODEL_DIR


@lru_cache(maxsize=1)
def _model():
    """延迟加载模型，并在同一进程里复用同一个实例（加载一次要几秒）。"""
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover - 依赖缺失时才会走这里
        raise RuntimeError(
            "缺少 fastembed。请执行：uv sync --extra embedding"
        ) from exc

    cache_dir = resolve_model_dir()
    try:
        # local_files_only=True：只用本地文件，缺文件直接报错，不偷偷联网下载。
        return TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            f"中文向量模型未能从 {cache_dir} 加载。"
            "请确认 models/ 目录完整，或在 .env 里设置 EMBEDDING_MODEL_DIR 指向模型目录。"
        ) from exc


def embed_many(texts: list[str]) -> list[list[float]]:
    """批量生成句向量：第 i 条输入对应第 i 条输出。

    批量传入而不是逐条调用，一是省掉重复的加载开销，
    二是保证这批向量出自同一次模型调用，彼此可直接比较。
    """
    # 空文本交给模型会得到一条看似正常、实际没有意义的向量，
    # 后面排序时它还会悄悄占掉一个名次，所以在入口就拦掉。
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError("texts 必须是非空列表")
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"第 {index + 1} 条文本为空，无法生成向量")

    # 分词、加 [CLS]/[SEP]、转 token ID、模型推理、取 [CLS] 位置向量并归一化，
    # 都由 fastembed 在 embed() 内部完成，不在这里手写。
    vectors = [vector.tolist() for vector in _model().embed(list(texts))]

    # 兜底自检：条数对不上就没法和原文一一对应，维度不一致就没法两两比较。
    if len(vectors) != len(texts):
        raise RuntimeError(f"向量数量（{len(vectors)}）与文本数量（{len(texts)}）不一致")
    dims = {len(vector) for vector in vectors}
    if len(dims) != 1:
        raise RuntimeError(f"同一批向量的维度不一致：{sorted(dims)}")
    return vectors


def embed(text: str) -> list[float]:
    """单条文本向量化，内部仍走同一套批量逻辑，保证与批量结果同源。"""
    return embed_many([text])[0]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """余弦相似度 = 点积 / (两条向量的长度之积)，取值在 -1 到 1。"""
    # 维度不同无法逐项相乘；空向量没有方向可比。
    if not left or not right:
        raise ValueError("两条向量都不能为空")
    if len(left) != len(right):
        raise ValueError(f"两条向量维度不一致：{len(left)} 与 {len(right)}，无法计算相似度")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("零向量没有方向，无法计算余弦相似度")

    dot_product = sum(a * b for a, b in zip(left, right))
    return dot_product / (left_norm * right_norm)


def model_ready() -> tuple[bool, str]:
    """模型是否可用（供主程序给出友好提示，而不是抛一长串堆栈）。"""
    try:
        _model()
    except RuntimeError as exc:
        return False, str(exc)
    return True, MODEL_NAME
