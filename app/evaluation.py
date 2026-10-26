"""单元13：人工标注的检索指标与 RAGAS 自动指标，分开定义。

两类指标量的是**两件不同的事**，课堂上必须分开讲：

- `retrieval_scores`（Precision@3 / Recall@3）：**纯规则、零模型调用**。
  逐条检索结果按"来源文件是否在人工标注里"判相关；Precision 按条数计，
  Recall 按命中的**不同**相关来源数计（同一文件的三行 CSV 全命中也只算一次，
  防止重复片段虚增覆盖率）。
- `ragas_scores`（Context Precision / Context Recall / Faithfulness）：**调用真实模型当评审**。
  前两者拿参考答案逐条判断检索文本的相关性；Faithfulness 判断回答里的
  每个事实是否能从检索文本中得到支持——它**不是回答正确率**。

一个容易踩的坑：`expected_sources` 标注的是**文件名**（DOC-02-01-普通商品退货规则.md），
而库里的 `source` 是**相对项目根的路径**。比较前先取最后一段做归一化，
否则每一条都会被判不相关，Precision 直接归零。

    uv run python -c "from app.evaluation import retrieval_scores as r; print(r(['模拟数据/knowledge/DOC-02-01-普通商品退货规则.md'], ['DOC-02-01-普通商品退货规则.md']))"
"""

from __future__ import annotations

import os

# 任务卡点名要看 Top-3 的来源，K 与检索服务的默认值保持一致。
TOP_K = 3


def normalize_source(source: str) -> str:
    """把库里的相对路径归一化成文件名，与 expected_sources 的标注口径对齐。"""
    text = str(source).replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def retrieval_scores(
    retrieved_sources: list[str], expected_sources: list[str]
) -> tuple[float, float]:
    """用人工标注的相关来源计算检索质量，不调用任何模型。

    Precision：Top-K 中每条占一个位置，来源在标注里就算相关，按条数计；
    Recall：命中的相关**来源**去重后计数——同一文件多个片段（如 CSV 三行）
    都命中也只算一次，避免重复片段虚增覆盖率。
    """
    expected = {normalize_source(source) for source in expected_sources}
    if not expected:
        raise ValueError("每道评估题至少标注一个相关来源")
    retrieved = [normalize_source(source) for source in retrieved_sources]

    relevant_results = sum(source in expected for source in retrieved)
    relevant_sources_found = len(set(retrieved) & expected)
    precision = relevant_results / len(retrieved) if retrieved else 0.0
    recall = relevant_sources_found / len(expected)
    return precision, recall


def ragas_scores(
    question: str, answer: str, contexts: list[str], reference: str
) -> dict[str, float]:
    """调用 DeepSeek 作为 RAGAS 评审模型，评估检索上下文与生成答案。

    返回 context_precision / context_recall / faithfulness 三个分数。
    这一步会发起真实 API 请求，评审费用随题数增加；密钥缺失、ragas 未安装
    都在**发起请求之前**拦截并给出可执行的下一步，不会等到一半才报错。
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法运行 RAGAS 评审；请在 .env 里填写")

    try:
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:  # pragma: no cover - 依赖缺失时才会走这里
        raise RuntimeError(
            f"缺少 ragas 依赖（{exc}）。请执行：uv sync --extra evaluation"
        ) from exc

    from openai import AsyncOpenAI

    judge = llm_factory(
        os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        client=AsyncOpenAI(
            api_key=api_key, base_url="https://api.deepseek.com", timeout=60
        ),
    )
    # RAGAS 的 context precision/recall 是参考答案驱动的 LLM 判断，
    # 与上面按人工标注来源算的 Precision@3 / Recall@3 不是同一个数。
    return {
        "context_precision": float(
            ContextPrecision(llm=judge)
            .score(user_input=question, reference=reference, retrieved_contexts=contexts)
            .value
        ),
        "context_recall": float(
            ContextRecall(llm=judge)
            .score(user_input=question, reference=reference, retrieved_contexts=contexts)
            .value
        ),
        "faithfulness": float(
            Faithfulness(llm=judge)
            .score(user_input=question, response=answer, retrieved_contexts=contexts)
            .value
        ),
    }
