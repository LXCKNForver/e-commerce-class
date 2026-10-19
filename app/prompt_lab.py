"""单元03：同一份订单数据 + 同一个问题，对比差/好两套 Prompt 的回答。

课堂上要证明的核心是：输入完全相同、模型完全相同，
只有提示词不同，回答在"是否核对字段、是否引用来源、是否能拒答"上出现差异。
"""

from __future__ import annotations

from app.llm import extract_order_id
from app.prompts import build_messages, format_order_context
from app.tools import AS_OF, query_order


def resolve_context(question: str) -> str:
    """从用户问题里取订单号，查询模拟订单并渲染成 facts 片段。"""
    order_id = extract_order_id(question)
    if not order_id:
        return "（未提供订单号，无法查询模拟订单数据）"

    result = query_order(order_id)
    if not result["ok"]:
        return f"（{result['message']}）"
    return format_order_context(result["order"])


def compare_prompts(llm, question: str, *, as_of: str = AS_OF) -> dict[str, str]:
    """同一上下文、同一问题，分别用两套提示词取一次回答。"""
    context = resolve_context(question)
    weak_answer = llm.chat(build_messages(question, context, quality="weak", as_of=as_of))
    strong_answer = llm.chat(build_messages(question, context, quality="strong", as_of=as_of))
    return {
        "question": question,
        "context": context,
        "weak": weak_answer,
        "strong": strong_answer,
    }


def render_comparison(result: dict[str, str]) -> str:
    """把对比结果渲染成便于截图的文本。"""
    lines = [
        f"顾客问题：{result['question']}",
        "",
        "【输入数据（两次调用完全相同）】",
        result["context"],
        "",
        "【差提示词回答】",
        result["weak"],
        "",
        "【好提示词回答】",
        result["strong"],
    ]
    return "\n".join(lines)
