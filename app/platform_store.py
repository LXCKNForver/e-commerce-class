"""单元24：会话、反馈、转人工与审计——四张表，一个 SQLite 文件。

四张表分开，是因为它们回答四个不同的问题：

    turns     谁在哪个会话里问了什么、答复是什么     —— 页面恢复对话用
    audit     这次请求走了哪条路由、多久、命中几条   —— "为什么慢/为什么没命中"用
    feedback  这条答复有没有帮上忙                   —— 答复质量复盘用
    handoffs  哪条答复被要求转人工、处理了没有       —— 人工接管队列用

**每一张表的查询都带 `seat_id` 条件**。权限不是在接口层"顺手 if 一下"，而是写进
SQL 的 WHERE：拿别人的 request_id 去点反馈，改不到任何行，直接 404。
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.platform_auth import Principal, require_admin
from app.platform_errors import NotFound

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    request_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
    seat_id TEXT NOT NULL, session_id TEXT NOT NULL,
    question TEXT NOT NULL, answer TEXT NOT NULL,
    route TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    request_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
    seat_id TEXT NOT NULL, route TEXT NOT NULL,
    status TEXT NOT NULL, duration_ms INTEGER NOT NULL,
    source_count INTEGER NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    request_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
    seat_id TEXT NOT NULL, helpful INTEGER NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS handoffs (
    request_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
    seat_id TEXT NOT NULL, status TEXT NOT NULL,
    created_at TEXT NOT NULL, resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (tenant_id, seat_id, session_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit (created_at);
"""


class SessionStore:
    """四张表的读写；构造时建库建表，重复启动不会丢数据。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def _session(self):
        """一次读写：成功提交、无论如何都关连接。

        Windows 上没关掉的 sqlite 连接会一直占着文件句柄，测试里的临时目录就删不掉
        （PermissionError: WinError 32）——所以这里不用 `with connect()`（它只提交不关）。
        """
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    # ------------------------------------------------------------ 写入

    def save_turn(
        self,
        principal: Principal,
        request_id: str,
        session_id: str,
        question: str,
        answer: str,
        route: str,
        status: str,
        duration_ms: int,
        source_count: int,
    ) -> None:
        """答复与审计同一事务提交：只有对话没有追踪记录，排障时等于没有。"""
        now = self._now()
        with self._session() as db:
            db.execute(
                "INSERT INTO turns VALUES (?,?,?,?,?,?,?,?,?)",
                (request_id, principal.tenant, principal.seat_id, session_id,
                 question, answer, route, status, now),
            )
            db.execute(
                "INSERT INTO audit VALUES (?,?,?,?,?,?,?,?)",
                (request_id, principal.tenant, principal.seat_id, route, status,
                 int(duration_ms), int(source_count), now),
            )

    def save_feedback(self, principal: Principal, request_id: str, helpful: bool) -> None:
        """反馈只能绑本人的答复；重复提交覆盖为最新一次（页面允许改主意）。"""
        with self._session() as db:
            found = db.execute(
                "SELECT 1 FROM turns WHERE request_id=? AND tenant_id=? AND seat_id=?",
                (request_id, principal.tenant, principal.seat_id),
            ).fetchone()
            if not found:
                raise NotFound("未找到本人的答复记录")
            db.execute(
                "INSERT INTO feedback VALUES (?,?,?,?,?) ON CONFLICT(request_id) "
                "DO UPDATE SET helpful=excluded.helpful, created_at=excluded.created_at",
                (request_id, principal.tenant, principal.seat_id, int(helpful), self._now()),
            )

    def request_handoff(self, principal: Principal, request_id: str) -> None:
        """客服只能把本人的答复转人工；模型没有这个写入工具。"""
        with self._session() as db:
            found = db.execute(
                "SELECT 1 FROM turns WHERE request_id=? AND tenant_id=? AND seat_id=?",
                (request_id, principal.tenant, principal.seat_id),
            ).fetchone()
            if not found:
                raise NotFound("未找到本人的答复记录")
            db.execute(
                "INSERT INTO handoffs VALUES (?,?,?,?,?,NULL) ON CONFLICT(request_id) DO NOTHING",
                (request_id, principal.tenant, principal.seat_id, "pending", self._now()),
            )

    def resolve_handoff(self, principal: Principal, request_id: str) -> None:
        """只有主管/管理员能关闭队列项，且只能关本租户里 pending 的那条。"""
        require_admin(principal)
        with self._session() as db:
            cursor = db.execute(
                "UPDATE handoffs SET status='resolved', resolved_at=? "
                "WHERE request_id=? AND tenant_id=? AND status='pending'",
                (self._now(), request_id, principal.tenant),
            )
            if cursor.rowcount == 0:
                raise NotFound("未找到待处理的转人工记录")

    # ------------------------------------------------------------ 查询

    def history(self, principal: Principal, session_id: str, limit: int = 6) -> list[dict[str, str]]:
        """最近几轮问答，用于指代与页面恢复；跨坐席读不到。"""
        with self._session() as db:
            rows = db.execute(
                "SELECT question, answer, route, status, created_at FROM turns "
                "WHERE tenant_id=? AND seat_id=? AND session_id=? "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (principal.tenant, principal.seat_id, session_id, int(limit)),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def list_handoffs(self, principal: Principal) -> list[dict[str, Any]]:
        require_admin(principal)
        with self._session() as db:
            rows = db.execute(
                "SELECT h.request_id, h.seat_id, h.status, h.created_at, h.resolved_at, "
                "t.question, t.answer FROM handoffs h "
                "JOIN turns t ON t.request_id=h.request_id "
                "WHERE h.tenant_id=? ORDER BY h.status='resolved', h.created_at DESC LIMIT 50",
                (principal.tenant,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_feedback(self, principal: Principal) -> list[dict[str, Any]]:
        """管理员看反馈与对应问答，按"没帮助优先"排序。"""
        require_admin(principal)
        with self._session() as db:
            rows = db.execute(
                "SELECT f.request_id, f.seat_id, f.helpful, f.created_at, "
                "t.question, t.answer FROM feedback f "
                "JOIN turns t ON t.request_id=f.request_id "
                "WHERE f.tenant_id=? ORDER BY f.helpful ASC, f.created_at DESC LIMIT 50",
                (principal.tenant,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_audit(self, principal: Principal, limit: int = 50) -> list[dict[str, Any]]:
        """调用记录：只给管理台，不返回问题正文（正文在 turns 里）。"""
        require_admin(principal)
        with self._session() as db:
            rows = db.execute(
                "SELECT request_id, seat_id, route, status, duration_ms, source_count, created_at "
                "FROM audit WHERE tenant_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (principal.tenant, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self, principal: Principal) -> dict[str, Any]:
        """管理台顶部的四个数字：请求数、平均耗时、转人工待处理、反馈无帮助率。"""
        require_admin(principal)
        with self._session() as db:
            total = db.execute(
                "SELECT COUNT(*) FROM audit WHERE tenant_id=?", (principal.tenant,)
            ).fetchone()[0]
            avg = db.execute(
                "SELECT AVG(duration_ms) FROM audit WHERE tenant_id=?", (principal.tenant,)
            ).fetchone()[0]
            pending = db.execute(
                "SELECT COUNT(*) FROM handoffs WHERE tenant_id=? AND status='pending'",
                (principal.tenant,),
            ).fetchone()[0]
            feedback_total, feedback_bad = db.execute(
                "SELECT COUNT(*), SUM(CASE WHEN helpful=0 THEN 1 ELSE 0 END) "
                "FROM feedback WHERE tenant_id=?", (principal.tenant,)
            ).fetchone()
        return {
            "requests": int(total or 0),
            "avg_duration_ms": int(round(avg or 0)),
            "pending_handoffs": int(pending or 0),
            "feedback_total": int(feedback_total or 0),
            "feedback_bad": int(feedback_bad or 0),
        }

    def count_turns(self, principal: Principal) -> int:
        with self._session() as db:
            return int(db.execute(
                "SELECT COUNT(*) FROM turns WHERE tenant_id=? AND seat_id=?",
                (principal.tenant, principal.seat_id),
            ).fetchone()[0])
