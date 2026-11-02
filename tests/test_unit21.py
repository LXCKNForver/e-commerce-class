"""单元21 验收测试：五角色 + 共享黑板 + 协调者；评审拒绝只修订一次。

覆盖任务卡的三条完成标准，默认全离线、不调用任何真实 API：

1. **至少 3 个智能体有明确输入、输出和职责**：角色表逐条核对，并确认一次协作里
   五个角色都实际发过言。
2. **共享状态能追溯**：黑板每条消息都有发送者，计划/观察/证据/草稿/评审都在。
3. **评审未通过时有受控修订，仍未通过不得自动交付**：修订上限 1；二次不通过时
   答复留空、状态转人工、草稿与评审意见保留。

白名单与参数格式校验走单元15 的真代码（`ToolRegistry.call` → `authorize_tool`），
规划与执行走单元20 的真代码（`make_plan` / `execute_plan`），测试里只替换"工具执行
函数"和"模型"，这样测出来的行为就是课堂上的行为。
"""

from __future__ import annotations

import json
import os
import unittest

from langchain_core.messages import AIMessage

from app.multi_agent import (
    MAX_REVISIONS,
    ROLE_COORDINATOR,
    ROLE_EVIDENCE,
    ROLE_EXECUTOR,
    ROLE_PLANNER,
    ROLE_RESPONDER,
    ROLE_REVIEWER,
    ROLES,
    STATUS_ABORTED,
    STATUS_DELIVERED,
    STATUS_HUMAN,
    Blackboard,
    CrewProtocolError,
    ScriptedCrewModel,
    analyze_evidence,
    compose_answer,
    coordinate,
    render_concepts,
    render_flow,
    render_handoff,
    render_result,
    render_roles,
    review_answer,
)
from app.plan_solver import plan_step
from app.toolkit import (
    TOOL_DEADLINE,
    TOOL_FAQ,
    TOOL_SEARCH,
    ToolRegistry,
    ToolSpec,
)

GOAL = "买家 2026年12月1日签收的普通商品要退货，请给出处理建议并说明依据"
FAQ_ANSWER = "课程样例中签收后7天内且未使用可申请退货评估，最终由人工复核"

EVIDENCE_OK = {"sufficient": True, "reason": "观察都返回了真实内容", "missing": "无"}
EVIDENCE_GAP = {"sufficient": False, "reason": "观察里有未找到", "missing": "第1步未找到：EC-FAQ-99"}
REVIEW_PASS = {"passed": True, "reason": "答复只在观察范围内作答"}
REVIEW_FAIL = {"passed": False, "reason": "没有引用依据文件，无法核对来源"}
REVIEW_FAIL_AGAIN = {"passed": False, "reason": "修订后仍未引用依据文件"}

DRAFT_OK = "依据 DOC-02-01-普通商品退货规则.md：7 天时限内申请退货，截止 2026-12-08。"
DRAFT_FLAWED = "普通商品可以在签收后 7 天内申请退货，请尽快提交申请。"
DRAFT_FIXED = "依据 DOC-02-01-普通商品退货规则.md：签收 2026-12-01，截止 2026-12-08。"
DRAFT_GAP = "EC-FAQ-99 在本组资料库里没有对应记录，资料不足，建议转人工核实。"


class FakeModel:
    """模型替身：按脚本依次返回内容，并把每次收到的消息留下来供断言。"""

    def __init__(self, *replies: str | dict) -> None:
        self.replies = list(replies)
        self.calls = 0
        self.seen: list[list] = []

    def invoke(self, messages):
        index = min(self.calls, len(self.replies) - 1) if self.replies else 0
        reply = self.replies[index] if self.replies else ""
        self.calls += 1
        self.seen.append(list(messages))
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return AIMessage(content=content)

    def texts(self) -> list[str]:
        return [
            " ".join(str(getattr(item, "content", "")) for item in messages)
            for messages in self.seen
        ]


def plan_json(*steps: dict) -> str:
    return json.dumps({"steps": [dict(step) for step in steps]}, ensure_ascii=False)


def build_fake_registry(found_ids: tuple[str, ...] = ("EC-FAQ-01",)) -> ToolRegistry:
    """用单元15 的工具名登记三个桩处理器：白名单与参数格式仍是真代码。"""

    def lookup(question_id: str) -> dict:
        if question_id not in found_ids:
            return {
                "storage": "SQLite（结构化查询）",
                "source": "常见售后问题.csv",
                "question_id": question_id,
                "status": "未找到",
            }
        return {
            "storage": "SQLite（结构化查询）",
            "source": "常见售后问题.csv",
            "question_id": question_id,
            "question": "普通商品签收后还能退货吗",
            "answer": FAQ_ANSWER,
            "source_id": "DOC-02-01",
        }

    registry = ToolRegistry()
    registry.register(ToolSpec(
        TOOL_SEARCH,
        "检索本组制度或流程资料，返回片段和文件来源",
        {"query": "制度或流程问题，2—120 字"},
        lambda query: {
            "storage": "Chroma（向量检索）",
            "query": query,
            "hits": [{"source": "DOC-02-01-普通商品退货规则.md", "text": "签收后7天内"}],
        },
    ))
    registry.register(ToolSpec(
        TOOL_FAQ,
        "按本组 FAQ 编号从只读 SQLite 查标准答复与来源编号",
        {"question_id": "本组 FAQ 编号，形如 EC-FAQ-01"},
        lookup,
    ))

    def deadline(sign_date: str, window_days: str, basis_doc: str) -> dict:
        if window_days != "7":
            raise ValueError(f"{basis_doc} 里没有“{window_days}天”这个时限")
        return {
            "storage": "Python 计算（日期加法）",
            "sign_date": sign_date,
            "window_days": window_days,
            "deadline": "2026-12-08",
            "basis_doc": basis_doc,
        }

    registry.register(ToolSpec(
        TOOL_DEADLINE,
        "按签收日期与资料写明的退货时限，算出退货评估截止日期",
        {"sign_date": "签收日期", "window_days": "时限天数", "basis_doc": "依据文件名"},
        deadline,
    ))
    return registry


# ---------------------------------------------------------------- ① 角色契约


class RoleContractTest(unittest.TestCase):
    """完成标准一：至少 3 个智能体，各有明确的输入、输出和职责。"""

    def test_at_least_three_agents(self) -> None:
        self.assertGreaterEqual(len(ROLES), 3)

    def test_every_role_has_full_contract(self) -> None:
        for role in ROLES:
            self.assertTrue(role.key)
            self.assertTrue(role.name)
            self.assertTrue(role.duty, role.key)
            self.assertTrue(role.inputs, role.key)
            self.assertTrue(role.outputs, role.key)
            self.assertTrue(role.forbidden, role.key)

    def test_role_keys_are_unique(self) -> None:
        keys = [role.key for role in ROLES]
        self.assertEqual(len(keys), len(set(keys)))

    def test_five_roles_cover_the_chain(self) -> None:
        self.assertEqual(
            [role.key for role in ROLES],
            [ROLE_PLANNER, ROLE_EXECUTOR, ROLE_EVIDENCE, ROLE_RESPONDER, ROLE_REVIEWER],
        )

    def test_coordinator_is_not_a_model_role(self) -> None:
        """协调者不进角色表：它不调模型，只做调度与放行判断。"""
        self.assertNotIn(ROLE_COORDINATOR, [role.key for role in ROLES])

    def test_responder_cannot_release_itself(self) -> None:
        """答复智能体的"不许做什么"里必须写明不能自行放行。"""
        responder = next(role for role in ROLES if role.key == ROLE_RESPONDER)
        self.assertIn("放行", responder.forbidden)

    def test_reviewer_cannot_write_answer(self) -> None:
        reviewer = next(role for role in ROLES if role.key == ROLE_REVIEWER)
        self.assertIn("代写", reviewer.forbidden)

    def test_role_table_rendered(self) -> None:
        text = render_roles()
        for role in ROLES:
            self.assertIn(role.name, text)
        self.assertIn("协调者", text)


# ---------------------------------------------------------------- ② 协议


class ProtocolTest(unittest.TestCase):
    """结构化角色的输出必须遵守协议：不合规就报错，不猜。"""

    def setUp(self) -> None:
        self.registry = build_fake_registry()
        self.steps = [{
            "step": 1, "tool": TOOL_FAQ, "arguments": {"question_id": "EC-FAQ-01"},
            "observation": {"question_id": "EC-FAQ-01", "answer": FAQ_ANSWER},
        }]

    def test_fenced_json_is_parsed(self) -> None:
        result = analyze_evidence(FakeModel(f"```json\n{json.dumps(EVIDENCE_OK)}\n```"),
                                  GOAL, self.steps)
        self.assertTrue(result["sufficient"])

    def test_free_text_is_rejected(self) -> None:
        with self.assertRaises(CrewProtocolError):
            analyze_evidence(FakeModel("看起来够了"), GOAL, self.steps)

    def test_non_object_json_is_rejected(self) -> None:
        with self.assertRaises(CrewProtocolError):
            analyze_evidence(FakeModel("[1, 2, 3]"), GOAL, self.steps)

    def test_evidence_without_required_fields(self) -> None:
        with self.assertRaises(CrewProtocolError):
            analyze_evidence(FakeModel({"sufficient": True}), GOAL, self.steps)

    def test_review_without_required_fields(self) -> None:
        with self.assertRaises(CrewProtocolError):
            review_answer(FakeModel({"reason": "还行"}), GOAL, self.steps, EVIDENCE_OK, DRAFT_OK)

    def test_empty_draft_is_rejected(self) -> None:
        with self.assertRaises(CrewProtocolError):
            compose_answer(FakeModel("   "), GOAL, self.steps, EVIDENCE_OK)

    def test_evidence_sees_only_observations(self) -> None:
        model = FakeModel(EVIDENCE_OK)
        analyze_evidence(model, GOAL, self.steps)
        self.assertIn(TOOL_FAQ, model.texts()[0])
        self.assertIn(FAQ_ANSWER, model.texts()[0])

    def test_review_sees_draft_and_evidence(self) -> None:
        model = FakeModel(REVIEW_PASS)
        review_answer(model, GOAL, self.steps, EVIDENCE_OK, DRAFT_OK)
        text = model.texts()[0]
        self.assertIn(DRAFT_OK, text)
        self.assertIn("evidence_check", text)


# ---------------------------------------------------------------- ③ 协作与评审回路


class CoordinationTest(unittest.TestCase):
    """完成标准二与三：黑板可追溯；拒绝后修订一次，再拒绝就停止交付。"""

    def setUp(self) -> None:
        self.registry = build_fake_registry()

    def _board(self, *replies: str | dict) -> Blackboard:
        model = FakeModel(*replies)
        return coordinate(model, self.registry, GOAL)

    def test_normal_run_delivers(self) -> None:
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, "查标准答复", question_id="EC-FAQ-01")),
            EVIDENCE_OK, DRAFT_OK, REVIEW_PASS,
        )
        self.assertTrue(board.delivered)
        self.assertEqual(board.status, STATUS_DELIVERED)
        self.assertEqual(board.answer, DRAFT_OK)
        self.assertEqual(board.drafts, [DRAFT_OK])
        self.assertEqual(len(board.review_history), 1)

    def test_all_five_roles_speak(self) -> None:
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            EVIDENCE_OK, DRAFT_OK, REVIEW_PASS,
        )
        senders = set(board.senders)
        for role in (ROLE_PLANNER, ROLE_EXECUTOR, ROLE_EVIDENCE, ROLE_RESPONDER, ROLE_REVIEWER):
            self.assertIn(role, senders)
        self.assertIn(ROLE_COORDINATOR, senders)

    def test_every_message_has_sender_and_content(self) -> None:
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            EVIDENCE_OK, DRAFT_OK, REVIEW_PASS,
        )
        self.assertTrue(board.messages)
        for index, message in enumerate(board.messages, start=1):
            self.assertEqual(message.seq, index)
            self.assertTrue(message.sender)
            self.assertTrue(message.content)

    def test_board_keeps_all_artifacts(self) -> None:
        board = self._board(
            plan_json(
                plan_step(TOOL_FAQ, question_id="EC-FAQ-01"),
                plan_step(TOOL_SEARCH, query="普通商品退货需要哪些材料"),
            ),
            EVIDENCE_OK, DRAFT_OK, REVIEW_PASS,
        )
        self.assertEqual(len(board.plan), 2)
        self.assertEqual(len(board.observations), 2)
        self.assertTrue(board.evidence)
        self.assertEqual(board.drafts, [DRAFT_OK])
        self.assertEqual(len(board.review_history), 1)
        # 每条观察都记了工具、参数与返回值，这是"能查看工具调用结果"的证据。
        for row in board.observations:
            self.assertTrue(row["tool"])
            self.assertTrue(row["arguments"])
            self.assertTrue(row["observation"])

    def test_review_rejection_triggers_one_revision(self) -> None:
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            EVIDENCE_OK, DRAFT_FLAWED, REVIEW_FAIL, DRAFT_FIXED, REVIEW_PASS,
        )
        self.assertEqual(board.revisions, MAX_REVISIONS)
        self.assertEqual(board.drafts, [DRAFT_FLAWED, DRAFT_FIXED])
        self.assertEqual(len(board.review_history), 2)
        self.assertTrue(board.delivered)
        self.assertEqual(board.answer, DRAFT_FIXED)

    def test_revision_feedback_is_passed_to_responder(self) -> None:
        model = FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            EVIDENCE_OK, DRAFT_FLAWED, REVIEW_FAIL, DRAFT_FIXED, REVIEW_PASS,
        )
        coordinate(model, self.registry, GOAL)
        # 调用顺序：0 规划 → 1 证据 → 2 草稿 → 3 评审 → 4 按评审意见修订。
        self.assertIn(REVIEW_FAIL["reason"], model.texts()[4])
        self.assertNotIn(REVIEW_FAIL["reason"], model.texts()[2])

    def test_second_rejection_stops_delivery(self) -> None:
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            EVIDENCE_OK, DRAFT_FLAWED, REVIEW_FAIL, DRAFT_FIXED, REVIEW_FAIL_AGAIN,
        )
        self.assertFalse(board.delivered)
        self.assertEqual(board.answer, "")
        self.assertEqual(board.status, STATUS_HUMAN)
        self.assertEqual(len(board.review_history), 2)
        # 草稿与评审意见留给人工复核，不因为停下就丢掉。
        self.assertEqual(len(board.drafts), 2)
        self.assertTrue(board.review["reason"])

    def test_rejected_draft_is_not_the_answer(self) -> None:
        """答复智能体写了草稿，但它自己不能把草稿变成答复。"""
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            EVIDENCE_OK, DRAFT_FLAWED, REVIEW_FAIL, DRAFT_FIXED, REVIEW_FAIL_AGAIN,
        )
        self.assertTrue(board.draft)
        self.assertNotEqual(board.answer, board.draft)

    def test_human_review_note_is_on_the_board(self) -> None:
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            EVIDENCE_OK, DRAFT_FLAWED, REVIEW_FAIL, DRAFT_FIXED, REVIEW_FAIL_AGAIN,
        )
        joined = " ".join(message.content for message in board.messages)
        self.assertIn("人工复核", joined)

    def test_execution_abort_skips_later_roles(self) -> None:
        """执行阶段就失败：连草稿都没有，证据/答复/评审三个角色不启动。"""
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
        )
        self.assertEqual(board.status, STATUS_ABORTED)
        self.assertEqual(board.drafts, [])
        self.assertEqual(board.review_history, [])
        self.assertEqual(board.answer, "")
        self.assertTrue(board.error)

    def test_insufficient_evidence_is_reported(self) -> None:
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-99")),
            EVIDENCE_GAP, DRAFT_GAP, REVIEW_PASS,
        )
        self.assertFalse(board.evidence_sufficient)
        self.assertIn("未找到", board.evidence["missing"])
        # 证据不足不是失败：如实说明后仍可交付。
        self.assertTrue(board.delivered)
        self.assertIn("资料不足", board.answer)

    def test_model_calls_are_countable(self) -> None:
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            EVIDENCE_OK, DRAFT_OK, REVIEW_PASS,
        )
        # 规划 1 + 证据 1 + 答复 1 + 评审 1 = 4，是可数的，不是口头承诺。
        self.assertEqual(board.model_calls, 4)

    def test_model_calls_with_one_revision(self) -> None:
        board = self._board(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            EVIDENCE_OK, DRAFT_FLAWED, REVIEW_FAIL, DRAFT_FIXED, REVIEW_PASS,
        )
        self.assertEqual(board.model_calls, 6)

    def test_invalid_task_is_rejected_before_any_role(self) -> None:
        for bad in ("", "   ", 123, "查", "查" * 400):
            model = FakeModel(plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")))
            with self.assertRaises(ValueError):
                coordinate(model, self.registry, bad)
            self.assertEqual(model.calls, 0, f"{bad!r} 竟然调用了模型")

    def test_zero_revisions_allowed(self) -> None:
        """max_revisions=0 时一次都不修订：不通过就直接转人工。"""
        board = coordinate(
            FakeModel(
                plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
                EVIDENCE_OK, DRAFT_FLAWED, REVIEW_FAIL,
            ),
            self.registry, GOAL, max_revisions=0,
        )
        self.assertEqual(board.revisions, 0)
        self.assertEqual(len(board.review_history), 1)
        self.assertFalse(board.delivered)


# ---------------------------------------------------------------- ④ 离线剧本模型


class ScriptedCrewModelTest(unittest.TestCase):
    """离线剧本模型：按角色分派，评审结论按调用顺序取。"""

    def _invoke(self, model: ScriptedCrewModel, messages: list) -> str:
        return str(model.invoke(messages).content)

    def test_planner_returns_json_plan(self) -> None:
        model = ScriptedCrewModel(steps=[plan_step(TOOL_FAQ, question_id="EC-FAQ-01")])
        from app.multi_agent import _system, _human
        from app.plan_solver import PLANNER_MARK

        raw = self._invoke(model, [_system(f"你是{PLANNER_MARK}"), _human("目标：退货怎么办")])
        payload = json.loads(raw)
        self.assertEqual(payload["steps"][0]["tool"], TOOL_FAQ)

    def test_planner_revision_uses_revised_script(self) -> None:
        from app.multi_agent import _system, _human
        from app.plan_solver import FEEDBACK_MARK, PLANNER_MARK

        model = ScriptedCrewModel(
            steps=[plan_step(TOOL_FAQ, question_id="BAD")],
            revised=[plan_step(TOOL_FAQ, question_id="EC-FAQ-01")],
        )
        first = self._invoke(model, [
            _system(f"你是{PLANNER_MARK}"), _human(f"目标：退货怎么办\n{FEEDBACK_MARK}无"),
        ])
        second = self._invoke(model, [
            _system(f"你是{PLANNER_MARK}"), _human(f"目标：退货怎么办\n{FEEDBACK_MARK}编号错"),
        ])
        self.assertEqual(json.loads(first)["steps"][0]["arguments"]["question_id"], "BAD")
        self.assertEqual(json.loads(second)["steps"][0]["arguments"]["question_id"], "EC-FAQ-01")

    def test_evidence_rule_says_insufficient_on_not_found(self) -> None:
        from app.multi_agent import EVIDENCE_MARK, _payload, _system, _human

        model = ScriptedCrewModel()
        raw = self._invoke(model, [
            _system(f"你是{EVIDENCE_MARK}"),
            _human(_payload(task=GOAL, steps=[{
                "step": 1, "tool": TOOL_FAQ, "arguments": {"question_id": "EC-FAQ-99"},
                "observation": {"status": "未找到", "question_id": "EC-FAQ-99"},
            }])),
        ])
        result = json.loads(raw)
        self.assertFalse(result["sufficient"])
        self.assertIn("未找到", result["missing"])

    def test_evidence_rule_says_insufficient_without_steps(self) -> None:
        from app.multi_agent import EVIDENCE_MARK, _payload, _system, _human

        model = ScriptedCrewModel()
        raw = self._invoke(model, [
            _system(f"你是{EVIDENCE_MARK}"), _human(_payload(task=GOAL, steps=[])),
        ])
        self.assertFalse(json.loads(raw)["sufficient"])

    def test_evidence_rule_says_sufficient_on_real_observations(self) -> None:
        from app.multi_agent import EVIDENCE_MARK, _payload, _system, _human

        model = ScriptedCrewModel()
        raw = self._invoke(model, [
            _system(f"你是{EVIDENCE_MARK}"),
            _human(_payload(task=GOAL, steps=[{
                "step": 1, "tool": TOOL_FAQ, "arguments": {"question_id": "EC-FAQ-01"},
                "observation": {"question_id": "EC-FAQ-01", "answer": FAQ_ANSWER},
            }])),
        ])
        self.assertTrue(json.loads(raw)["sufficient"])

    def test_reviews_are_taken_in_order(self) -> None:
        from app.multi_agent import REVIEWER_MARK, _payload, _system, _human

        model = ScriptedCrewModel(reviews=[REVIEW_FAIL, REVIEW_PASS])
        messages = [
            _system(f"你是{REVIEWER_MARK}"),
            _human(_payload(task=GOAL, steps=[], evidence_check=EVIDENCE_OK, draft=DRAFT_OK)),
        ]
        self.assertFalse(json.loads(self._invoke(model, messages))["passed"])
        self.assertTrue(json.loads(self._invoke(model, messages))["passed"])

    def test_drafts_are_taken_in_order(self) -> None:
        from app.multi_agent import _system, _human

        model = ScriptedCrewModel(drafts=[DRAFT_FLAWED, DRAFT_FIXED])
        messages = [_system("你是答复员"), _human("用户目标：退货怎么办")]
        self.assertEqual(self._invoke(model, messages), DRAFT_FLAWED)
        self.assertEqual(self._invoke(model, messages), DRAFT_FIXED)

    def test_offline_draft_only_extracts_observations(self) -> None:
        from app.multi_agent import _payload, _system, _human

        model = ScriptedCrewModel()
        text = self._invoke(model, [
            _system("你是答复员"),
            _human(_payload(task=GOAL, steps=[{
                "step": 1, "tool": TOOL_FAQ, "arguments": {"question_id": "EC-FAQ-01"},
                "observation": {"question_id": "EC-FAQ-01", "answer": FAQ_ANSWER},
            }], evidence_check=EVIDENCE_OK)),
        ])
        self.assertIn(FAQ_ANSWER, text)
        self.assertIn("离线脚本模式", text)


# ---------------------------------------------------------------- ⑤ 渲染


class RenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.board = coordinate(
            FakeModel(
                plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
                EVIDENCE_OK, DRAFT_FLAWED, REVIEW_FAIL, DRAFT_FIXED, REVIEW_PASS,
            ),
            build_fake_registry(), GOAL,
        )

    def test_result_shows_status_and_answer(self) -> None:
        text = render_result(self.board)
        self.assertIn(STATUS_DELIVERED, text)
        self.assertIn(DRAFT_FIXED, text)

    def test_result_shows_both_drafts(self) -> None:
        text = render_result(self.board)
        self.assertIn("第1版草稿", text)
        self.assertIn("第2版草稿", text)

    def test_result_shows_two_reviews(self) -> None:
        text = render_result(self.board)
        self.assertIn("第1次评审", text)
        self.assertIn("第2次评审", text)

    def test_handoff_lists_senders(self) -> None:
        text = render_handoff(self.board)
        for role in ROLES:
            self.assertIn(role.name, text)

    def test_scenario_header_prints_the_goal_once(self) -> None:
        """标题只印场景名，目标交给 `render_result` 的第一行。

        早期两处都印了一遍，`/agents demo` 的每个场景开头都是
        「【…】目标：X / 空行 / 目标：X」——没有用例盯渲染格式，就一直没发现。
        """
        from app.multi_lab import CrewScenario, render_scenario

        scenario = CrewScenario(
            key="deliver", label="正常协作 → 评审通过 → 交付", goal=GOAL, board=self.board
        )
        text = render_scenario(scenario)
        self.assertEqual(text.count(GOAL), 1)
        self.assertTrue(text.startswith("【正常协作 → 评审通过 → 交付】"))

    def test_flow_mentions_review_loop(self) -> None:
        text = render_flow()
        for marker in ("规划智能体", "评审智能体", "共享黑板", "停止自动交付"):
            self.assertIn(marker, text)

    def test_concepts_table(self) -> None:
        text = render_concepts()
        for marker in ("智能体", "协调者", "共享黑板", "受控修订", "停止自动交付"):
            self.assertIn(marker, text)

    def test_aborted_board_does_not_fake_evidence(self) -> None:
        board = coordinate(
            FakeModel(
                plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
                plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
            ),
            build_fake_registry(), GOAL,
        )
        self.assertIn("未发生", render_result(board))


# ---------------------------------------------------------------- ⑥ 真实资料


class RealMaterialsTest(unittest.TestCase):
    """接上本组真实资料与单元15 的注册表再跑一遍；缺依赖或资料就跳过。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from app.multi_lab import build_stack
        except ImportError as error:  # pragma: no cover - 依赖缺失
            raise unittest.SkipTest(f"缺少依赖：{error}")
        try:
            cls.registry, _model, cls.kb, cls.rows = build_stack()
        except Exception as error:  # noqa: BLE001 - 资料或向量库不可用时跳过
            raise unittest.SkipTest(f"真实资料不可用：{error}")

    def test_materials_are_the_same_batch(self) -> None:
        self.assertEqual(self.kb, 7)
        self.assertEqual(self.rows, 3)

    def test_deliver_scenario_runs_all_roles(self) -> None:
        from app.multi_lab import run_scenario

        scenario = run_scenario("deliver", registry=self.registry, inject=True)
        self.assertIsNotNone(scenario.board)
        board = scenario.board
        self.assertTrue(board.delivered)
        self.assertEqual(board.status, STATUS_DELIVERED)
        self.assertEqual(len(board.observations), 3)
        for role in (ROLE_PLANNER, ROLE_EXECUTOR, ROLE_EVIDENCE, ROLE_RESPONDER, ROLE_REVIEWER):
            self.assertIn(role, board.senders)

    def test_revise_scenario(self) -> None:
        from app.multi_lab import run_scenario

        board = run_scenario("revise", registry=self.registry, inject=True).board
        self.assertEqual(board.revisions, MAX_REVISIONS)
        self.assertEqual(len(board.review_history), 2)
        self.assertTrue(board.delivered)

    def test_reject_scenario_stops_delivery(self) -> None:
        from app.multi_lab import run_scenario

        board = run_scenario("reject", registry=self.registry, inject=True).board
        self.assertFalse(board.delivered)
        self.assertEqual(board.answer, "")
        self.assertEqual(board.status, STATUS_HUMAN)

    def test_gap_scenario_reports_missing_evidence(self) -> None:
        from app.multi_lab import run_scenario

        board = run_scenario("gap", registry=self.registry, inject=True).board
        self.assertFalse(board.evidence_sufficient)
        self.assertIn("转人工", board.answer)

    def test_abort_scenario_skips_review(self) -> None:
        from app.multi_lab import run_scenario

        board = run_scenario("abort", registry=self.registry, inject=True).board
        self.assertEqual(board.status, STATUS_ABORTED)
        self.assertEqual(board.drafts, [])
        self.assertEqual(board.review_history, [])

    def test_all_standards_pass(self) -> None:
        from app.multi_lab import run_class

        run = run_class(offline=True)
        self.assertTrue(run.standards)
        failed = [name for name, _evidence, verdict in run.standards if verdict != "通过"]
        self.assertEqual(failed, [], f"未通过的标准：{failed}")

    def test_invalid_inputs_are_rejected(self) -> None:
        from app.multi_lab import INVALID_CASES, run_invalid

        rows = run_invalid(self.registry)
        self.assertEqual(len(rows), len(INVALID_CASES))
        self.assertTrue(all("已拒绝" in verdict for _label, verdict in rows))
        self.assertTrue(all("模型调用 0 次" in verdict for _label, verdict in rows))


# ---------------------------------------------------------------- ⑦ 真实模型


class RealModelTest(unittest.TestCase):
    """真实 DeepSeek 的用例：要显式开启（RUN_REAL_MODEL=1）且配了密钥才跑。

    开关刻意不用 `LLM_PROVIDER`——那个变量会被别的用例从 `.env` 读进来，整套测试的
    总数就会随执行顺序变，"多少项通过"是要写进证据文档的数字，不能飘。
    """

    @classmethod
    def setUpClass(cls) -> None:
        if (os.getenv("RUN_REAL_MODEL", "") or "").strip() != "1":
            raise unittest.SkipTest("未显式开启真实模型用例（RUN_REAL_MODEL=1）")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise unittest.SkipTest("未配置 DEEPSEEK_API_KEY")
        try:
            from app.multi_lab import build_stack
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"缺少依赖：{error}")
        cls.registry, _model, _kb, _rows = build_stack()

    def test_real_model_runs_the_crew(self) -> None:
        from app.multi_lab import GOAL_DELIVER, build_crew_model
        from app.plan_solver import build_plan_model

        board = coordinate(build_plan_model(provider="deepseek"), self.registry, GOAL_DELIVER)
        self.assertTrue(board.observations)
        self.assertTrue(board.drafts)
        self.assertTrue(board.review_history)
        self.assertIn(board.status, (STATUS_DELIVERED, STATUS_HUMAN, STATUS_ABORTED))
        # 交付与否由评审说了算：交付了就必须有答复，没交付就必须没有答复。
        self.assertEqual(bool(board.answer), board.delivered)


if __name__ == "__main__":
    unittest.main()
