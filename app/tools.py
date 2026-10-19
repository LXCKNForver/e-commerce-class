"""单元04/05：受控工具实现、Schema 定义与调用前校验。

受控工具（只读，不写任何业务系统）：
- query_order            只读查询模拟订单；
- check_return_eligibility按样例政策判断退货资格，争议场景转人工；
- draft_manual_ticket    只生成工单草稿，不落真实库、不触发真实流程；
- lookup_policy_doc      单元05 新增：按来源编号读取样例政策原文，只摘录不作判断。

工具实现与执行边界分离：本模块只提供"能做什么"的函数，
"允许做什么"由 app/registry.py 的注册表登记决定。
TOOL_SCHEMAS 也从注册表派生，保证模型可见清单与程序可执行清单同源。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from app.data_store import KNOWLEDGE_DIR, OrderRepository

_repo = OrderRepository()

# 数据基准日，用于计算签收天数；与 OrderRepository.as_of 保持一致。
AS_OF = _repo.as_of

CATEGORY_POLICY_DOCS = {
    "定制": "DOC-02-05",
    "易耗品": "DOC-02-06",
}

DOC_ID_PATTERN = re.compile(r"DOC-\d{2}-\d{2}", re.IGNORECASE)


class ToolError(RuntimeError):
    """参数不合法或工具执行失败时抛出，错误信息会回填给模型。"""


# ---------------------------------------------------------------- 工具实现


def query_order(order_id: str) -> dict:
    """只读查询订单。"""
    order = _repo.get_order(order_id)
    if not order:
        return {"ok": False, "message": f"未找到订单 {order_id}，请说明资料不足并建议人工核实。"}

    return {
        "ok": True,
        "order": {key: order[key] for key in order if key in {
            "order_id", "customer_id", "item_category", "item_name",
            "order_status", "logistics_status", "delivered_on",
            "used_or_opened", "invoice_status", "amount_yuan", "after_sale_status",
        }},
        "source": "模拟数据/records.json（课程虚构样例，只读）",
    }


def check_return_eligibility(order_id: str, reason: str) -> dict:
    """按样例政策判断退货资格；争议一律转人工，不替客服下结论。"""
    result = query_order(order_id)
    if not result["ok"]:
        return {"ok": False, "message": result["message"], "manual": True}

    order = result["order"]
    category = order["item_category"]
    used = bool(order["used_or_opened"])
    delivered_on = order.get("delivered_on")

    base = {
        "ok": True,
        "order_id": order_id,
        "as_of": AS_OF,
        "item_category": category,
        "delivered_on": delivered_on,
        "used_or_opened": used,
    }

    # 例外条款优先于普通 7 天退货政策。
    if category == "定制":
        return {
            **base,
            "eligible": False,
            "reason": "定制商品不适用普通 7 天退货（DOC-02-05）。",
            "policy_docs": ["DOC-02-05"],
            "manual": True,
        }
    if category == "易耗品" and used:
        return {
            **base,
            "eligible": False,
            "reason": "已拆封易耗品不按普通无理由退货（DOC-02-06）。",
            "policy_docs": ["DOC-02-06"],
            "manual": True,
        }

    # 质量问题、投诉类原因必须转人工复核。
    dispute_keywords = ["质量", "破损", "错发", "漏发", "投诉", "欺诈", "法律", "假货"]
    if any(keyword in reason for keyword in dispute_keywords):
        return {
            **base,
            "eligible": None,
            "reason": f"原因含争议要素（{reason}），样例政策无法直接判定，转人工复核（DOC-02-04、DOC-02-12）。",
            "policy_docs": ["DOC-02-04", "DOC-02-12"],
            "manual": True,
        }

    # 未签收订单不涉及"已签收才能申请的普通退货"判断。
    if not delivered_on:
        return {
            **base,
            "eligible": None,
            "reason": "订单未签收；如需拦截按 DOC-02-03 生成待人工确认工单，不直接改派。",
            "policy_docs": ["DOC-02-03"],
            "manual": True,
        }

    days = (date.fromisoformat(AS_OF) - date.fromisoformat(delivered_on)).days
    within = days <= 7
    docs = ["DOC-02-01"]
    if used:
        docs.append("DOC-02-06")
    return {
        **base,
        "eligible": bool(within and not used),
        "days_since_delivered": days,
        "calculation": f"{AS_OF} - {delivered_on} = {days} 天（普通 7 天退货样例政策）",
        "reason": (
            "在样例 7 天期限内且未拆封，符合普通退货样例条件。" if within and not used
            else f"超样例期限（{days} 天）或已拆封，样例政策下不符合普通退货条件，建议人工复核。"
        ),
        "policy_docs": docs,
        "manual": not (within and not used),
    }


def draft_manual_ticket(order_id: str, issue: str) -> dict:
    """只生成工单草稿，不写真实系统、不实际退款或改派。"""
    order = _repo.get_order(order_id)
    exists = order is not None
    return {
        "ok": True,
        "ticket_draft": {
            "order_id": order_id,
            "issue": issue,
            "order_found": exists,
            "item_category": order["item_category"] if exists else None,
            "status": "待人工确认",
        },
        "note": "仅生成草稿，未写入任何真实业务系统，也未发起退款或改派。",
        "policy_docs": ["DOC-02-12", "DOC-02-15"],
    }


def lookup_policy_doc(doc_id: str) -> dict:
    """单元05 新增的第二个业务工具：按来源编号读取样例政策原文。

    只摘录文件中的"规则与说明""使用边界"两节，不判断该政策是否适用于当前订单，
    也不改写文件内容——与订单类工具一样是只读的。
    """
    match = DOC_ID_PATTERN.search(doc_id or "")
    if not match:
        return {"ok": False, "message": f"来源编号格式不正确：{doc_id}；正确格式如 DOC-02-05。"}

    normalized = match.group(0).upper()
    if not KNOWLEDGE_DIR.is_dir():
        return {"ok": False, "message": f"缺少样例政策目录：{KNOWLEDGE_DIR}"}

    candidates = sorted(KNOWLEDGE_DIR.glob(f"{normalized}*.md"))
    if not candidates:
        return {
            "ok": False,
            "message": f"未找到来源 {normalized}，请说明资料不足并建议人工核实。",
        }

    path = candidates[0]
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        # 读取失败属于工具异常，交给注册表/调度器统一转成可读提示，不让主程序崩溃。
        raise ToolError(f"读取样例政策失败：{exc}") from exc

    title = ""
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    return {
        "ok": True,
        "doc_id": normalized,
        "title": title,
        "excerpt": _extract_sections(text),
        "source": f"模拟数据/knowledge/{path.name}（课程虚构样例，只读）",
        "note": "仅摘录原文，未判断是否适用于当前订单；适用判断见 check_return_eligibility。",
    }


def _extract_sections(text: str, *, limit: int = 400) -> str:
    """取"规则与说明""使用边界"两节的正文，避免把整份文件塞进对话。"""
    keep = False
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            keep = any(keyword in line for keyword in ("规则与说明", "使用边界"))
            continue
        if keep and line.strip():
            lines.append(line.strip())
    excerpt = "\n".join(lines)
    return excerpt[:limit] + ("…" if len(excerpt) > limit else "")


# ---------------------------------------------------------------- Schema 与校验


def __getattr__(name: str) -> Any:
    """单元05 起 Schema 由注册表派生，避免"模型能选"和"程序能执行"两套名单。

    这里用模块级 __getattr__ 延迟导入，绕开 tools <-> registry 的循环导入，
    同时保留单元04 的 `from app.tools import TOOL_SCHEMAS` 写法。
    """
    if name in {"TOOL_SCHEMAS", "TOOL_NAMES"}:
        from app.registry import default_registry

        schemas = default_registry().schemas()
        if name == "TOOL_SCHEMAS":
            return schemas
        return [item["function"]["name"] for item in schemas]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def execute_tool_call(call: dict[str, Any]) -> dict:
    """单元04 入口保留：实际校验与执行都交给注册表，保证两处行为一致。"""
    from app.registry import default_registry

    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        raise ToolError(f"{name} 的 arguments 必须是 JSON 对象")
    return default_registry().call(name, arguments)
