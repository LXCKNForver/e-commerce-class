"""单元24 验收测试：网页工作台——身份、工单权限、四路由、会话闭环、接口与页面。

任务卡（第八阶段收口）要求"能独立访问的完整产品"，所以测的不只是函数，还有：

1. **身份与权限**：令牌验签、过期、角色；客服只看自己名下工单，主管看全部。
2. **四路由**：工单号 → 结构化查询；FAQ 编号 → SQLite；统计问法 → 固定算子；
   其余 → 检索 + 模型。前三类**不调模型**，这是"不让模型编事实"的落点。
3. **闭环**：答复落库 → 反馈 → 转人工 → 主管处理 → 审计留痕。
4. **接口与页面**：没有 FastAPI 时接口用例跳过；页面文件不许依赖 CDN。

默认全离线：模型用 `MockLLM`，向量库用假注册表，只有工单台账读真实 CSV。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app.platform_auth import (
    PermissionDenied,
    Principal,
    ROLE_ADMIN,
    ROLE_AGENT,
    ROLE_SUPERVISOR,
    TokenError,
    can_view_all,
    issue_token,
    parse_bearer,
    render_roles,
    require_admin,
    verify_token,
)
from app.platform_data import (
    LEDGER_FILE,
    PlatformData,
    extract_order_id,
    extract_orderish,
    extract_ticket_id,
    render_seat_table,
)
from app.platform_errors import BadRequest, NotFound
from app.platform_runtime import PLATFORM_DATA_DIR, WEB_DIR, PlatformContext
from app.platform_service import (
    LOW_EVIDENCE_ANSWER,
    ROUTE_ANALYSIS,
    ROUTE_FAQ,
    ROUTE_KNOWLEDGE,
    ROUTE_TICKET,
    PlatformService,
    STATUS_ANSWERED,
    STATUS_NEEDS_HUMAN,
    render_routes,
    validate_question,
    validate_session_id,
)
from app.platform_store import SessionStore

SECRET = "unit24-test-secret-0123456789abcdef"
SECRET_KW = {"secret": SECRET}


def seat(seat_id: str, role: str = ROLE_AGENT) -> Principal:
    return Principal(seat_id=seat_id, role=role)


class FakeRegistry:
    """假的工具注册表：只记录被调用了什么，不碰向量库与 SQLite。"""

    def __init__(self, hits: list[dict] | None = None, faq: dict | None = None):
        self.hits = hits if hits is not None else [
            {"source": "DOC-02-01-普通商品退货规则.md", "text": "签收后 7 天内且未使用可申请退货评估。"}
        ]
        self.faq = faq
        self.calls: list[tuple[str, dict]] = []

    def names(self) -> list[str]:
        return ["search_knowledge", "lookup_faq", "compute_return_deadline"]

    def call(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, dict(arguments)))
        if name == "search_knowledge":
            return {"storage": "fake", "query": arguments.get("query"), "hits": self.hits}
        if name == "lookup_faq":
            if self.faq is None:
                return {"status": "未找到", "question_id": arguments.get("question_id")}
            return self.faq
        raise AssertionError(f"未预期的工具调用：{name}")


def make_context(tmp: Path, registry: FakeRegistry, model=None) -> PlatformContext:
    """拼一个不依赖 Chroma 的上下文：真实台账 + 假工具表 + 临时会话库。"""
    from app.llm import MockLLM

    return PlatformContext(
        model=model or MockLLM(),
        registry=registry,  # type: ignore[arg-type]
        retriever=None,     # type: ignore[arg-type]
        store=None,         # type: ignore[arg-type]
        data=PlatformData.load(PLATFORM_DATA_DIR),
        sessions=SessionStore(tmp / "platform.db"),
        knowledge_dir=PLATFORM_DATA_DIR,
        faq_file=PLATFORM_DATA_DIR / "常见问题.csv",
        knowledge_count=7,
        faq_rows=3,
    )


# ---------------------------------------------------------------- ① 身份


class AuthTest(unittest.TestCase):
    def test_roundtrip_keeps_seat_and_role(self) -> None:
        token = issue_token("A002", ROLE_AGENT, **SECRET_KW)
        principal = verify_token(token, **SECRET_KW)
        self.assertEqual(principal.seat_id, "A002")
        self.assertEqual(principal.role, ROLE_AGENT)
        self.assertEqual(principal.role_label, "客服坐席")

    def test_tampered_payload_is_rejected(self) -> None:
        token = issue_token("A001", ROLE_AGENT, **SECRET_KW)
        body, signature = token.split(".")
        forged = body[:-2] + ("AA" if body[-2:] != "AA" else "BB") + "." + signature
        with self.assertRaises(TokenError):
            verify_token(forged, **SECRET_KW)

    def test_expired_token_is_rejected(self) -> None:
        token = issue_token("A001", ROLE_AGENT, hours=1, now=1_000_000, **SECRET_KW)
        with self.assertRaises(TokenError) as caught:
            verify_token(token, now=1_000_000 + 3601, **SECRET_KW)
        self.assertIn("过期", str(caught.exception))

    def test_wrong_secret_is_rejected(self) -> None:
        token = issue_token("A001", ROLE_AGENT, **SECRET_KW)
        with self.assertRaises(TokenError):
            verify_token(token, secret="another-secret-0123456789abcdefgh")

    def test_bad_seat_and_role(self) -> None:
        for bad_seat in ("E001", "A01", "a001", ""):
            with self.subTest(seat=bad_seat):
                with self.assertRaises(TokenError):
                    issue_token(bad_seat, ROLE_AGENT, **SECRET_KW)
        with self.assertRaises(TokenError):
            issue_token("A001", "root", **SECRET_KW)

    def test_bearer_header_parsing(self) -> None:
        self.assertEqual(parse_bearer("Bearer abc.def"), "abc.def")
        for bad in (None, "", "abc.def", "Basic abc", "Bearer "):
            with self.subTest(header=bad):
                with self.assertRaises(TokenError):
                    parse_bearer(bad)

    def test_role_scope(self) -> None:
        self.assertFalse(can_view_all(seat("A001")))
        self.assertTrue(can_view_all(seat("M001", ROLE_SUPERVISOR)))
        require_admin(seat("M001", ROLE_SUPERVISOR))
        require_admin(seat("M001", ROLE_ADMIN))
        with self.assertRaises(PermissionDenied):
            require_admin(seat("A001"))

    def test_role_table_lists_three_roles(self) -> None:
        text = render_roles()
        for marker in ("客服坐席", "售后主管", "系统管理员", "只有自己名下"):
            self.assertIn(marker, text)


# ---------------------------------------------------------------- ② 工单与权限


class TicketDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = PlatformData.load(PLATFORM_DATA_DIR)

    def test_ledger_shape(self) -> None:
        self.assertEqual(len(self.data.tickets), 32)
        self.assertEqual(len(self.data.seats), 5)
        self.assertEqual(self.data.ledger_path.name, LEDGER_FILE)

    def test_ticket_id_extraction(self) -> None:
        self.assertEqual(extract_ticket_id("工单AS23004现在什么状态"), "AS23004")
        self.assertEqual(extract_ticket_id("as23031 处理到哪了"), "AS23031")
        self.assertIsNone(extract_ticket_id("订单 O230704 有物流异常"))
        # 六位数字不是本项目的工单号：AS230041 不应该被截成 AS23004 命中
        self.assertIsNone(extract_ticket_id("AS230041"))

    def test_order_id_extraction(self) -> None:
        self.assertEqual(extract_order_id("订单 O230701 退款到哪了"), "O230701")
        self.assertEqual(extract_order_id("o230704"), "O230704")
        # SO-2001 / NO2024 是"像订单号但台账里没有"的写法，严格匹配必须取不到，
        # 宽松匹配要能取出来用于给格式提示——两者分工不能混。
        self.assertIsNone(extract_order_id("我要查订单 SO-2001"))
        self.assertEqual(extract_orderish("我要查订单 SO-2001"), "SO-2001")
        self.assertIsNone(extract_orderish("退款要几天到账"))

    def test_order_lookup_respects_seat(self) -> None:
        mine = next(row for row in self.data.tickets if row.seat_id == "A001")
        found = self.data.tickets_of_order(seat("A001"), mine.order_id)
        self.assertEqual([row.case_id for row in found], [mine.case_id])
        # 同一张订单换别人来查：与非存在同一个结果
        other = next(row for row in self.data.tickets if row.seat_id == "A002")
        with self.assertRaises(NotFound):
            self.data.tickets_of_order(seat("A001"), other.order_id)
        self.assertEqual(len(self.data.tickets_of_order(seat("M001", ROLE_SUPERVISOR), other.order_id)), 1)
        with self.assertRaises(BadRequest):
            self.data.tickets_of_order(seat("A001"), "SO-2001")

    def test_agent_sees_only_own_tickets(self) -> None:
        mine = self.data.visible_tickets(seat("A001"))
        self.assertEqual(len(mine), 8)
        self.assertTrue(all(row.seat_id == "A001" for row in mine))

    def test_supervisor_sees_everything(self) -> None:
        self.assertEqual(len(self.data.visible_tickets(seat("M001", ROLE_SUPERVISOR))), 32)

    def test_other_seat_ticket_looks_missing(self) -> None:
        other = next(row for row in self.data.tickets if row.seat_id == "A002")
        with self.assertRaises(NotFound):
            self.data.read_ticket(seat("A001"), other.case_id)
        # 主管能读到同一张
        self.assertEqual(self.data.read_ticket(seat("M001", ROLE_SUPERVISOR), other.case_id).case_id,
                         other.case_id)

    def test_bad_and_unknown_ids(self) -> None:
        with self.assertRaises(BadRequest):
            self.data.read_ticket(seat("A001"), "T20261001")
        with self.assertRaises(NotFound):
            self.data.read_ticket(seat("M001", ROLE_SUPERVISOR), "AS99999")

    def test_overview_numbers(self) -> None:
        overview = self.data.overview(seat("M001", ROLE_SUPERVISOR))
        self.assertEqual(overview["total"], 32)
        self.assertEqual(overview["handoff_count"], 16)
        self.assertEqual(overview["refund_total"], 5544.0)
        self.assertEqual(sum(item["count"] for item in overview["by_issue"]), 32)
        self.assertEqual(overview["scope"], "全部坐席")
        mine = self.data.overview(seat("A001"))
        self.assertEqual(mine["total"], 8)
        self.assertEqual(mine["scope"], "A001 名下")

    def test_list_filters(self) -> None:
        rows = self.data.list_tickets(seat("M001", ROLE_SUPERVISOR), status="人工复核")
        self.assertTrue(rows)
        self.assertTrue(all(row.status == "人工复核" for row in rows))
        self.assertEqual(
            len(self.data.list_tickets(seat("M001", ROLE_SUPERVISOR), keyword="物流异常")),
            sum(1 for row in self.data.tickets if "物流异常" in row.summary or "物流异常" in row.issue_type),
        )
        # 列表同样按权限收窄
        self.assertTrue(all(row.seat_id == "A001" for row in self.data.list_tickets(seat("A001"))))

    def test_seat_table(self) -> None:
        text = render_seat_table(self.data)
        for name in ("靖瑞厅", "李省春"):
            self.assertIn(name, text)
        # 主管名下 0 张 ≠ 看不到：那一行必须写明"可见全部"
        boss_line = next(line for line in text.splitlines() if line.startswith("M001"))
        self.assertIn(f"全部 {len(self.data.tickets)} 张", boss_line)


# ---------------------------------------------------------------- ③ 会话库


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.tmp.name) / "platform.db")
        self.agent = seat("A001")
        self.boss = seat("M001", ROLE_SUPERVISOR)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def save(self, principal: Principal, request_id: str, session: str = "s1", route: str = "ticket") -> None:
        self.store.save_turn(principal, request_id, session, "工单 AS23001 什么状态", "已退款",
                             route, STATUS_ANSWERED, 12, 1)

    def test_history_is_per_seat(self) -> None:
        self.save(self.agent, "a" * 32)
        self.save(seat("A002"), "b" * 32)
        mine = self.store.history(self.agent, "s1")
        self.assertEqual(len(mine), 1)
        self.assertEqual(mine[0]["question"], "工单 AS23001 什么状态")

    def test_feedback_only_on_own_turn(self) -> None:
        self.save(self.agent, "a" * 32)
        self.store.save_feedback(self.agent, "a" * 32, True)
        self.assertEqual(len(self.store.list_feedback(self.boss)), 1)
        with self.assertRaises(NotFound):
            self.store.save_feedback(seat("A002"), "a" * 32, False)

    def test_handoff_lifecycle(self) -> None:
        self.save(self.agent, "a" * 32)
        self.store.request_handoff(self.agent, "a" * 32)
        items = self.store.list_handoffs(self.boss)
        self.assertEqual(items[0]["status"], "pending")
        stats = self.store.stats(self.boss)
        self.assertEqual(stats["pending_handoffs"], 1)
        self.store.resolve_handoff(self.boss, "a" * 32)
        self.assertEqual(self.store.list_handoffs(self.boss)[0]["status"], "resolved")
        with self.assertRaises(NotFound):
            self.store.resolve_handoff(self.boss, "a" * 32)
        with self.assertRaises(PermissionDenied):
            self.store.list_handoffs(self.agent)

    def test_audit_has_no_question_text(self) -> None:
        self.save(self.agent, "a" * 32)
        events = self.store.recent_audit(self.boss)
        self.assertEqual(len(events), 1)
        self.assertNotIn("question", events[0])
        self.assertEqual(events[0]["route"], "ticket")
        with self.assertRaises(PermissionDenied):
            self.store.recent_audit(self.agent)

    def test_handoff_on_other_turn_is_missing(self) -> None:
        self.save(self.agent, "a" * 32)
        with self.assertRaises(NotFound):
            self.store.request_handoff(seat("A003"), "a" * 32)


# ---------------------------------------------------------------- ④ 四路由


class ServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = FakeRegistry()
        self.context = make_context(Path(self.tmp.name), self.registry)
        self.service = PlatformService(self.context)
        self.agent = seat("A001")
        self.boss = seat("M001", ROLE_SUPERVISOR)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ticket_route_does_not_touch_model(self) -> None:
        result = self.service.ask(self.agent, "工单 AS23001 现在是什么状态？", "s1")
        self.assertEqual(result.route, ROUTE_TICKET)
        self.assertEqual(result.status, STATUS_ANSWERED)
        self.assertIn("已退款", result.answer)
        self.assertEqual(self.registry.calls, [])
        self.assertEqual(result.meta["ticket"]["case_id"], "AS23001")

    def test_ticket_of_other_seat_is_rejected(self) -> None:
        with self.assertRaises(NotFound):
            self.service.ask(self.agent, "工单 AS23002 什么状态？", "s1")

    def test_order_route_reads_ledger_without_model(self) -> None:
        # 用户说的"订单 O230701"必须查台账，不能丢给知识库：
        # 知识库只有政策文档，检索出来只会答"资料不足"。
        result = self.service.ask(self.agent, "订单 O230701 退到哪一步了？", "s1")
        self.assertEqual(result.route, ROUTE_TICKET)
        self.assertEqual(result.status, STATUS_ANSWERED)
        self.assertIn("AS23001", result.answer)
        self.assertEqual(self.registry.calls, [])

    def test_wrong_order_format_explains_the_shape(self) -> None:
        # 截图里那句"SO-2001 → 资料不足"就是缺了这一段：
        # 格式不对也要给"台账里长什么样"，而不是让模型说不知道。
        result = self.service.ask(self.agent, "我要查订单 SO-2001", "s1")
        self.assertEqual(result.route, ROUTE_TICKET)
        self.assertEqual(result.status, STATUS_NEEDS_HUMAN)
        self.assertIn("O 加六位数字", result.answer)
        self.assertIn("AS23004", result.answer)
        self.assertEqual(self.registry.calls, [])

    def test_render_context_tolerates_temp_session_db(self) -> None:
        # 验收脚本用临时会话库；装配摘要不能因为"路径不在仓库里"就崩。
        from app.platform_runtime import render_context

        text = render_context(self.context)
        self.assertIn("会话库", text)
        self.assertIn("（临时）", text)

    def test_analysis_route_counts_with_filters(self) -> None:
        result = self.service.ask(self.boss, "需要转人工的工单有多少？", "s1")
        self.assertEqual(result.route, ROUTE_ANALYSIS)
        self.assertIn("结果 16 单", result.answer)
        self.assertIn("总 32 行", result.answer)
        self.assertEqual(result.meta["analysis"]["rows_total"], 32)

    def test_analysis_refund_total(self) -> None:
        result = self.service.ask(self.boss, "已退款金额合计是多少？", "s1")
        self.assertEqual(result.route, ROUTE_ANALYSIS)
        self.assertIn("5544", result.answer)
        self.assertIn("算式", result.answer)

    def test_faq_route_found(self) -> None:
        registry = FakeRegistry(faq={
            "question_id": "EC-FAQ-01", "question": "普通商品签收后还能退货吗",
            "answer": "签收后 7 天内未使用可申请退货评估", "source_id": "DOC-02-01",
        })
        service = PlatformService(make_context(Path(self.tmp.name), registry))
        result = service.ask(self.boss, "EC-FAQ-01 的标准答复是什么？", "s1")
        self.assertEqual(result.route, ROUTE_FAQ)
        self.assertIn("签收后 7 天内", result.answer)
        self.assertEqual(registry.calls[0][0], "lookup_faq")

    def test_faq_route_not_found_needs_human(self) -> None:
        result = self.service.ask(self.boss, "EC-FAQ-99 怎么答复？", "s1")
        self.assertEqual(result.route, ROUTE_FAQ)
        self.assertEqual(result.status, STATUS_NEEDS_HUMAN)
        self.assertIn("没有 EC-FAQ-99", result.answer)

    def test_knowledge_route_uses_hits(self) -> None:
        result = self.service.ask(self.boss, "商品有质量问题可以退款吗？", "s1")
        self.assertEqual(result.route, ROUTE_KNOWLEDGE)
        self.assertEqual(result.status, STATUS_ANSWERED)
        self.assertTrue(result.answer.strip())
        self.assertEqual(result.sources[0]["source"], "DOC-02-01-普通商品退货规则.md")
        self.assertEqual(self.registry.calls[0][0], "search_knowledge")

    def test_no_hits_says_so(self) -> None:
        registry = FakeRegistry(hits=[])
        service = PlatformService(make_context(Path(self.tmp.name), registry))
        result = service.ask(self.boss, "海外仓的税怎么退？", "s1")
        self.assertEqual(result.route, ROUTE_KNOWLEDGE)
        self.assertEqual(result.status, STATUS_NEEDS_HUMAN)
        self.assertEqual(result.answer, LOW_EVIDENCE_ANSWER)

    def test_answer_is_persisted_with_audit(self) -> None:
        result = self.service.answer(self.agent, "工单 AS23001 现在是什么状态？", "s1")
        history = self.context.sessions.history(self.agent, "s1")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], STATUS_ANSWERED)
        events = self.context.sessions.recent_audit(self.boss)
        self.assertEqual(events[0]["request_id"], result.request_id)
        self.assertEqual(events[0]["route"], ROUTE_TICKET)

    def test_question_and_session_guards(self) -> None:
        for bad in ("", "   ", "查" * 501, 123):
            with self.subTest(question=bad):
                with self.assertRaises(BadRequest):
                    validate_question(bad)
        self.assertEqual(validate_question("  工单   AS23001  "), "工单 AS23001")
        for bad in ("", "a b", "x" * 81, "会话#1"):
            with self.subTest(session=bad):
                with self.assertRaises(BadRequest):
                    validate_session_id(bad)
        self.assertEqual(validate_session_id("s-1_ok"), "s-1_ok")

    def test_dashboard_numbers(self) -> None:
        board = self.service.dashboard(self.boss)
        self.assertEqual(board["tickets"]["total"], 32)
        self.assertEqual(board["analysis"]["rows_total"], 32)
        self.assertEqual(board["analysis"]["refund_total"], 5544.0)
        self.assertEqual(sum(item["count"] for item in board["analysis"]["issue_groups"]), 32)
        self.assertEqual(board["knowledge"]["documents"], 7)
        self.assertIn("search_knowledge", board["knowledge"]["tools"])
        mine = self.service.dashboard(self.agent)
        self.assertEqual(mine["tickets"]["total"], 8)

    def test_route_table(self) -> None:
        text = render_routes()
        for marker in ("ticket", "faq", "analysis", "knowledge", "不调"):
            self.assertIn(marker, text)


# ---------------------------------------------------------------- ⑤ 接口


def _fastapi_ready() -> bool:
    try:
        import fastapi  # noqa: F401
        import httpx  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_fastapi_ready(), "未安装 fastapi/httpx：跳过接口用例（uv sync --extra web）")
class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import create_app

        self.tmp = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"APP_TOKEN_SECRET": SECRET})
        self.env.start()
        self.registry = FakeRegistry()
        self.context = make_context(Path(self.tmp.name), self.registry)
        self.client = TestClient(create_app(self.context))
        self.agent_token = issue_token("A001", ROLE_AGENT)
        self.boss_token = issue_token("M001", ROLE_SUPERVISOR)

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def headers(self, token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    def test_health_needs_no_token(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_page_is_served(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("售后服务工作台", response.text)

    def test_missing_or_bad_token_is_401(self) -> None:
        self.assertEqual(self.client.get("/v1/me").status_code, 401)
        bad = self.client.get("/v1/me", headers={"Authorization": "Bearer broken.token"})
        self.assertEqual(bad.status_code, 401)

    def test_me_reports_role(self) -> None:
        me = self.client.get("/v1/me", headers=self.headers(self.agent_token)).json()
        self.assertEqual(me["seat_id"], "A001")
        self.assertEqual(me["name"], "靖瑞厅")
        self.assertFalse(me["can_view_all"])
        boss = self.client.get("/v1/me", headers=self.headers(self.boss_token)).json()
        self.assertTrue(boss["can_view_all"])

    def test_chat_ticket_then_feedback_then_handoff(self) -> None:
        body = {"question": "工单 AS23001 现在是什么状态？", "session_id": "s-api"}
        response = self.client.post("/v1/chat", json=body, headers=self.headers(self.agent_token))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["route"], "ticket")
        request_id = payload["request_id"]

        feedback = self.client.post("/v1/feedback", json={"request_id": request_id, "helpful": True},
                                    headers=self.headers(self.agent_token))
        self.assertEqual(feedback.status_code, 200)
        handoff = self.client.post("/v1/handoffs", json={"request_id": request_id},
                                   headers=self.headers(self.agent_token))
        self.assertEqual(handoff.status_code, 200)
        queue = self.client.get("/v1/admin/handoffs", headers=self.headers(self.boss_token)).json()
        self.assertEqual(len(queue["items"]), 1)

    def test_chat_other_seat_ticket_is_404(self) -> None:
        body = {"question": "工单 AS23002 现在是什么状态？", "session_id": "s-api"}
        response = self.client.post("/v1/chat", json=body, headers=self.headers(self.agent_token))
        self.assertEqual(response.status_code, 404)

    def test_admin_routes_are_forbidden_for_agent(self) -> None:
        for path in ("/v1/admin/handoffs", "/v1/admin/feedback", "/v1/admin/audit", "/v1/admin/stats"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path, headers=self.headers(self.agent_token)).status_code, 403)
        self.assertEqual(
            self.client.post("/v1/admin/reindex", headers=self.headers(self.agent_token)).status_code, 403
        )

    def test_ticket_scope_and_dashboard(self) -> None:
        mine = self.client.get("/v1/tickets", headers=self.headers(self.agent_token)).json()
        self.assertEqual(mine["total"], 8)
        self.assertIn("A001", mine["scope"])
        all_rows = self.client.get("/v1/tickets", headers=self.headers(self.boss_token)).json()
        self.assertEqual(all_rows["total"], 32)
        board = self.client.get("/v1/dashboard", headers=self.headers(self.boss_token)).json()
        self.assertEqual(board["tickets"]["handoff_count"], 16)

    def test_session_history_returns_own_turns(self) -> None:
        body = {"question": "各类售后问题各多少单？", "session_id": "s-hist"}
        self.client.post("/v1/chat", json=body, headers=self.headers(self.agent_token))
        history = self.client.get("/v1/sessions/s-hist", headers=self.headers(self.agent_token)).json()
        self.assertEqual(len(history["turns"]), 1)
        other = self.client.get("/v1/sessions/s-hist", headers=self.headers(self.boss_token)).json()
        self.assertEqual(other["turns"], [])

    def test_invalid_payload_is_422(self) -> None:
        response = self.client.post("/v1/chat", json={"question": "", "session_id": "s"},
                                    headers=self.headers(self.agent_token))
        self.assertEqual(response.status_code, 422)
        bad_id = self.client.post("/v1/feedback", json={"request_id": "short", "helpful": True},
                                  headers=self.headers(self.agent_token))
        self.assertEqual(bad_id.status_code, 422)


# ---------------------------------------------------------------- ⑥ 页面


class FrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.page = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    def test_page_exists_with_views(self) -> None:
        for marker in ("view-chat", "view-tickets", "view-data", "view-admin"):
            self.assertIn(f'id="{marker}"', self.page)

    def test_page_calls_the_real_endpoints(self) -> None:
        for endpoint in ("/v1/chat", "/v1/me", "/v1/tickets", "/v1/dashboard",
                         "/v1/feedback", "/v1/handoffs", "/v1/admin/handoffs"):
            self.assertIn(endpoint, self.page)

    def test_page_has_no_cdn_dependency(self) -> None:
        # 离线课堂：页面不许依赖任何外部脚本/样式，否则没网就是白屏。
        self.assertNotIn("cdn.", self.page)
        self.assertNotIn("https://", self.page)
        self.assertNotIn("http://", self.page)

    def test_page_keeps_token_in_memory(self) -> None:
        self.assertNotIn("localStorage", self.page)
        self.assertIn("sessionStorage", self.page)


# ---------------------------------------------------------------- ⑦ 真实模型（可选）


@unittest.skipUnless(os.getenv("RUN_REAL_MODEL") == "1", "未设置 RUN_REAL_MODEL=1：跳过真实模型用例")
class RealModelTest(unittest.TestCase):
    def test_knowledge_route_with_real_model(self) -> None:
        from app.platform_runtime import prepare_platform

        with tempfile.TemporaryDirectory() as tmp:
            context = prepare_platform(session_db=Path(tmp) / "p.db")
            service = PlatformService(context)
            result = service.ask(seat("M001", ROLE_SUPERVISOR),
                                 "商品有质量问题可以退款吗？", "real")
            self.assertEqual(result.route, ROUTE_KNOWLEDGE)
            self.assertTrue(result.sources)
            self.assertGreater(len(result.answer), 20)


if __name__ == "__main__":
    unittest.main()
