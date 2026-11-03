"""单元24：工单台账——结构化数据入口 + 坐席权限过滤。

这里只做三件事：读固定路径的 CSV、按身份过滤、给出可复核的统计口径。

**为什么工单不让模型查**：工单状态是结构化事实，模型只能"转述"，一旦让它自己
组织查询（写 SQL、拼路径），就等于把数据库交出去。所以工单号由正则识别、由 Python
按固定路径读取，模型全程不参与——它最多在解释里复述一次已经取到的字段。

**为什么"查不到"和"没权限"回同一个结果**：客服 A001 问 A002 名下的工单，如果回
"无权限"，就等于确认了这张工单存在。统一回 404，才不泄露工单归属。
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.platform_auth import ADMIN_ROLES, Principal, can_view_all
from app.platform_errors import BadRequest, NotFound

# 工单号形如 AS23001：中文紧邻字母时 `\b` 不成立，所以用显式的"前后不是字母数字"。
TICKET_PATTERN = re.compile(r"(?<![A-Za-z0-9])AS\d{5}(?!\d)", re.IGNORECASE)

# 订单号形如 O230701。用户也常写 SO-2001、NO2024 这类格式，那是**查不到**的写法，
# 但仍属"查订单"，必须先接住再解释；否则会掉进知识问答，检索出一堆提到"订单编号"
# 的政策片段，模型只能答"资料不足"——用户看到的就是"明明检索到 3 条却说没有"。
ORDER_PATTERN = re.compile(r"(?<![A-Za-z0-9])O\d{6}(?![0-9])", re.IGNORECASE)
ORDERISH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:SO|SN|NO|ORDER)[-\s]?\d{3,8}(?![0-9])", re.IGNORECASE
)

LEDGER_FILE = "售后工单台账.csv"
ASSIGN_FILE = "坐席分工.csv"
SEAT_FILE = "坐席.csv"

LEDGER_FIELDS = (
    "case_id", "submitted_at", "order_id", "product_category", "issue_type",
    "order_amount", "refund_amount", "status", "channel", "needs_handoff",
)
ASSIGN_FIELDS = ("case_id", "seat_id", "customer_id", "priority", "summary", "updated_at")
SEAT_FIELDS = ("seat_id", "name", "role", "team")

STATUS_ORDER = ("处理中", "人工复核", "已退款", "已关闭")


def extract_ticket_id(text: str) -> str | None:
    """从"帮我看下工单 AS23004 的状态"里取出工单号；取不到返回 None。"""
    match = TICKET_PATTERN.search(text or "")
    return match.group(0).upper() if match else None


def extract_order_id(text: str) -> str | None:
    """取出台账格式的订单号（O + 六位数字）；取不到返回 None。"""
    match = ORDER_PATTERN.search(text or "")
    return match.group(0).upper() if match else None


def extract_orderish(text: str) -> str | None:
    """宽松取一个"像订单号"的写法，只用来给格式提示，绝不拿去查台账。"""
    match = ORDERISH_PATTERN.search(text or "")
    return match.group(0).upper() if match else None


@dataclass(frozen=True)
class Seat:
    seat_id: str
    name: str
    role: str
    team: str


@dataclass(frozen=True)
class Ticket:
    """一张售后工单：台账字段 + 坐席分工字段拼起来。"""

    case_id: str
    seat_id: str
    customer_id: str
    order_id: str
    product_category: str
    issue_type: str
    order_amount: float
    refund_amount: float
    status: str
    channel: str
    needs_handoff: bool
    priority: str
    summary: str
    submitted_at: str
    updated_at: str
    seat_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "seat_id": self.seat_id,
            "seat_name": self.seat_name,
            "customer_id": self.customer_id,
            "order_id": self.order_id,
            "product_category": self.product_category,
            "issue_type": self.issue_type,
            "order_amount": self.order_amount,
            "refund_amount": self.refund_amount,
            "status": self.status,
            "channel": self.channel,
            "needs_handoff": self.needs_handoff,
            "priority": self.priority,
            "summary": self.summary,
            "submitted_at": self.submitted_at,
            "updated_at": self.updated_at,
        }


def _read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.exists():
        raise NotFound(f"缺少数据文件：{path.name}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        missing = [name for name in required if name not in header]
        if missing:
            raise BadRequest(f"{path.name} 缺少字段：{'、'.join(missing)}")
        return [dict(row) for row in reader]


def _number(raw: str) -> float:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return 0.0


@dataclass
class PlatformData:
    """一次加载、只读使用；写操作只发生在会话库里。"""

    ledger_path: Path
    tickets: list[Ticket] = field(default_factory=list)
    seats: dict[str, Seat] = field(default_factory=dict)
    _index: dict[str, Ticket] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, data_dir: Path) -> "PlatformData":
        ledger_rows = _read_csv(Path(data_dir) / LEDGER_FILE, LEDGER_FIELDS)
        assign_rows = {row["case_id"]: row for row in
                       _read_csv(Path(data_dir) / ASSIGN_FILE, ASSIGN_FIELDS)}
        seat_rows = _read_csv(Path(data_dir) / SEAT_FILE, SEAT_FIELDS)
        seats = {
            row["seat_id"]: Seat(row["seat_id"], row["name"], row["role"], row["team"])
            for row in seat_rows
        }
        tickets: list[Ticket] = []
        for row in ledger_rows:
            extra = assign_rows.get(row["case_id"])
            if extra is None:
                raise BadRequest(f"坐席分工.csv 缺少工单：{row['case_id']}")
            seat = seats.get(extra["seat_id"])
            tickets.append(Ticket(
                case_id=row["case_id"],
                seat_id=extra["seat_id"],
                customer_id=extra["customer_id"],
                order_id=row["order_id"],
                product_category=row["product_category"],
                issue_type=row["issue_type"],
                order_amount=_number(row["order_amount"]),
                refund_amount=_number(row["refund_amount"]),
                status=row["status"],
                channel=row["channel"],
                needs_handoff=row["needs_handoff"].strip() == "是",
                priority=extra["priority"],
                summary=extra["summary"],
                submitted_at=row["submitted_at"],
                updated_at=extra["updated_at"],
                seat_name=seat.name if seat else extra["seat_id"],
            ))
        data = cls(ledger_path=Path(data_dir) / LEDGER_FILE, tickets=tickets, seats=seats)
        data._index = {ticket.case_id.upper(): ticket for ticket in tickets}
        return data

    # ------------------------------------------------------------ 查询（带权限）

    def seat_of(self, principal: Principal) -> Seat | None:
        return self.seats.get(principal.seat_id)

    def visible_tickets(self, principal: Principal) -> list[Ticket]:
        """客服只拿自己名下的；主管与管理员拿全部。"""
        if can_view_all(principal):
            return list(self.tickets)
        return [ticket for ticket in self.tickets if ticket.seat_id == principal.seat_id]

    def read_ticket(self, principal: Principal, case_id: str) -> Ticket:
        """按工单号读一张工单：先看权限，再返回字段。"""
        key = (case_id or "").strip().upper()
        if not TICKET_PATTERN.fullmatch(key):
            raise BadRequest("工单号格式应为 AS 加五位数字，例如 AS23004")
        ticket = self._index.get(key)
        if ticket is None:
            raise NotFound("未找到可访问的工单")
        if not can_view_all(principal) and ticket.seat_id != principal.seat_id:
            # 与"不存在"同一个结果：不泄露别人的工单是否存在。
            raise NotFound("未找到可访问的工单")
        return ticket

    def tickets_of_order(self, principal: Principal, order_id: str) -> list[Ticket]:
        """按订单号读该订单名下的工单；一个订单可能对应多张。

        不在名下与不存在仍回同一个 404——订单归属同样不能泄露。
        """
        key = (order_id or "").strip().upper()
        if not ORDER_PATTERN.fullmatch(key):
            raise BadRequest("订单号格式应为 O 加六位数字，例如 O230701")
        hits = [ticket for ticket in self.tickets if ticket.order_id.upper() == key]
        if not can_view_all(principal):
            hits = [ticket for ticket in hits if ticket.seat_id == principal.seat_id]
        if not hits:
            raise NotFound("未找到可访问的工单")
        return hits

    def list_tickets(
        self,
        principal: Principal,
        *,
        status: str | None = None,
        issue: str | None = None,
        keyword: str | None = None,
        limit: int = 50,
    ) -> list[Ticket]:
        """工单列表：先按权限收窄，再按状态/问题类型/关键词筛。"""
        rows = self.visible_tickets(principal)
        if status:
            rows = [row for row in rows if row.status == status]
        if issue:
            rows = [row for row in rows if row.issue_type == issue]
        if keyword:
            needle = keyword.strip()
            rows = [row for row in rows
                    if needle in row.case_id or needle in row.summary
                    or needle in row.issue_type
                    or needle in row.customer_id or needle in row.order_id]
        rows.sort(key=lambda row: (row.submitted_at, row.case_id), reverse=True)
        return rows[:limit]

    # ------------------------------------------------------------ 可复核的统计

    def overview(self, principal: Principal) -> dict[str, Any]:
        """页面顶部的数字：全部来自可见工单，不猜、不补。"""
        rows = self.visible_tickets(principal)
        status_counter = Counter(row.status for row in rows)
        issue_counter = Counter(row.issue_type for row in rows)
        priority_counter = Counter(row.priority for row in rows)
        refund_total = round(sum(row.refund_amount for row in rows), 2)
        amount_total = round(sum(row.order_amount for row in rows), 2)
        handoff = [row.case_id for row in rows if row.needs_handoff]
        return {
            "scope": "全部坐席" if can_view_all(principal) else f"{principal.seat_id} 名下",
            "total": len(rows),
            "by_status": [{"name": name, "count": status_counter[name]}
                          for name in STATUS_ORDER if status_counter[name]],
            "by_issue": [{"name": name, "count": count}
                         for name, count in issue_counter.most_common()],
            "by_priority": [{"name": name, "count": priority_counter[name]}
                            for name in ("高", "中", "低") if priority_counter[name]],
            "handoff_count": len(handoff),
            "refund_total": refund_total,
            "amount_total": amount_total,
            "avg_amount": round(amount_total / len(rows), 2) if rows else 0.0,
            "seats": [
                {"seat_id": seat.seat_id, "name": seat.name, "role": seat.role,
                 "count": sum(1 for row in self.tickets if row.seat_id == seat.seat_id)}
                for seat in self.seats.values()
            ],
        }

    def seat_names(self) -> dict[str, str]:
        return {seat.seat_id: seat.name for seat in self.seats.values()}


def render_seat_table(data: PlatformData) -> str:
    """坐席与名下工单数；文档与验收都用这一份。

    主管/管理员不接单，名下天然是 0 张。直接印 0 会被读成"主管没有工单可看"，
    所以多一列「可见范围」，把"名下"和"能看"分开写清楚。
    """
    lines = ["坐席\t姓名\t角色\t名下工单\t可见范围"]
    counts = Counter(ticket.seat_id for ticket in data.tickets)
    agents = sum(1 for seat in data.seats.values() if seat.role not in ADMIN_ROLES)
    admins = len(data.seats) - agents
    for seat in data.seats.values():
        if seat.role in ADMIN_ROLES:
            scope = f"全部 {len(data.tickets)} 张（含 {agents} 名客服名下）"
        else:
            scope = f"本席 {counts[seat.seat_id]} 张"
        lines.append(f"{seat.seat_id}\t{seat.name}\t{seat.role}\t{counts[seat.seat_id]}\t{scope}")
    lines.append(f"合计\t\t\t{len(data.tickets)}\t客服 {agents} 人、主管 {admins} 人")
    return "\n".join(lines)
