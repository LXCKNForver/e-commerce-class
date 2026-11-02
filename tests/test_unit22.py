"""单元22 验收测试：三角色群聊 + 消息流 + 轮次/成本双上限 + 复核放行。

覆盖任务卡的四条完成标准，默认全离线、不调用任何真实 API：

1. **至少两个角色参与**：角色表逐条核对；一次协作里实际发言的角色不少于两个。
2. **消息内容可查看**：每条消息都有发送者、发言序号与正文；后发言的人看得到前面
   所有人的原话（断言提示词里确实带着前面的发言）。
3. **轮次和成本受控**：`max_round` 是消息条数硬上限（含发起者那条）；估算 token
   超预算就不再开口；模型调用次数等于发言条数（管理器 0 次）。
4. **结果由复核角色确认**：复核通过才把草稿变成答复；不通过、没轮到都是空的。

白名单与参数格式校验走单元15 的真代码（`ToolRegistry.call` → `authorize_tool`），
测试里只替换"工具执行函数"和"模型"，这样测出来的行为就是课堂上的行为。
"""

from __future__ import annotations

import json
import os
import unittest

from langchain_core.messages import AIMessage

from app.chat_crew import (
    AGENTS,
    CHARS_PER_TOKEN,
    DEFAULT_ROUNDS,
    DEFAULT_TOKEN_BUDGET,
    EVIDENCE_START,
    MAX_ROUNDS,
    ROLE_EXECUTOR,
    ROLE_INITIATOR,
    ROLE_ORDER,
    ROLE_PLANNER,
    ROLE_REVIEWER,
    ROLE_TAGS,
    STATUS_BUDGET,
    STATUS_DELIVERED,
    STATUS_MAX_ROUND,
    STATUS_REJECTED,
    STOP_BUDGET,
    STOP_MAX_ROUND,
    STOP_REJECTED,
    STOP_REVIEWED,
    ChatAgent,
    ChatProtocolError,
    ChatScript,
    GroupChat,
    GroupChatManager,
    ScriptedChatModel,
    build_chat_model,
    collect_evidence,
    estimate_tokens,
    extract_faq_ids,
    knowledge_query,
    parse_verdict,
    render_agents,
    render_concepts,
    render_evidence,
    render_flow,
    render_result,
    render_termination,
    render_transcript,
    run_collaboration,
    validate_budget,
    validate_rounds,
)
from app.toolkit import TOOL_DEADLINE, TOOL_FAQ, TOOL_SEARCH, ToolRegistry, ToolSpec

TASK = "买家 2026年12月1日签收的普通商品要退货：查 EC-FAQ-01 的标准答复，并检索退货规则里对材料和时限的要求"
FAQ_ANSWER = "课程样例中签收后7天内且未使用可申请退货评估，最终由人工复核"

PLAN_TEXT = "（离线）规划：1) 引用 EC-FAQ-01；2) 按 DOC-02-01 说明材料与时限；3) 结尾分「有证据」与「转人工」。"
ANSWER_TEXT = "（离线）结论：签收后 7 天内且未使用可申请退货评估，最终由人工复核（来源 DOC-02-01）。"
ANSWER_GAP = "（离线）EC-FAQ-99 未找到，资料不足，建议转人工核实。"
REVIEW_PASS = {"passed": True, "reason": "只用了证据里的事实"}
REVIEW_FAIL = {"passed": False, "reason": "写了证据里没有的退款承诺"}


class FakeModel:
    """模型替身：按 system 里的角色标记分派，每个角色一条队列；并留下收到的消息。"""

    def __init__(
        self,
        planner: tuple[str, ...] = (),
        executor: tuple[str, ...] = (),
        reviewer: tuple[dict, ...] = (),
    ) -> None:
        self.queues: dict[str, list] = {
            ROLE_PLANNER: list(planner),
            ROLE_EXECUTOR: list(executor),
            ROLE_REVIEWER: list(reviewer),
        }
        self.calls = 0
        self.seen: list[list] = []

    def invoke(self, messages):
        self.calls += 1
        self.seen.append(list(messages))
        system = str(getattr(messages[0], "content", ""))
        if ROLE_TAGS[ROLE_PLANNER] in system:
            return AIMessage(content=self._take(ROLE_PLANNER, PLAN_TEXT))
        if ROLE_TAGS[ROLE_EXECUTOR] in system:
            return AIMessage(content=self._take(ROLE_EXECUTOR, ANSWER_TEXT))
        if ROLE_TAGS[ROLE_REVIEWER] in system:
            return AIMessage(content=json.dumps(self._take(ROLE_REVIEWER, REVIEW_PASS), ensure_ascii=False))
        raise AssertionError(f"认不出角色：{system[:40]}")

    def _take(self, role: str, default):
        queue = self.queues[role]
        return queue.pop(0) if queue else default

    def prompts(self) -> list[str]:
        return [
            "\n".join(str(getattr(item, "content", "")) for item in messages)
            for messages in self.seen
        ]


def build_fake_registry(found_ids: tuple[str, ...] = ("EC-FAQ-01",)) -> ToolRegistry:
    """用单元15 的工具名登记桩处理器：白名单与参数格式仍是真代码。"""

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
    registry.register(ToolSpec(
        TOOL_DEADLINE,
        "按签收日期与资料写明的退货时限，算出退货评估截止日期",
        {"sign_date": "签收日期", "window_days": "时限天数", "basis_doc": "依据文件名"},
        lambda sign_date, window_days, basis_doc: {
            "storage": "Python 计算（日期加法）",
            "deadline": "2026-12-08",
        },
    ))
    return registry


# ---------------------------------------------------------------- ① 角色契约


class AgentContractTest(unittest.TestCase):
    """完成标准一：至少两个角色，各有明确的输入、输出和职责。"""

    def test_at_least_two_agents(self) -> None:
        self.assertGreaterEqual(len(AGENTS), 2)

    def test_three_roles_in_speaking_order(self) -> None:
        self.assertEqual([spec.role for spec in AGENTS], list(ROLE_ORDER))

    def test_every_role_has_full_contract(self) -> None:
        for spec in AGENTS:
            self.assertTrue(spec.name, spec.role)
            self.assertTrue(spec.duty, spec.role)
            self.assertTrue(spec.inputs, spec.role)
            self.assertTrue(spec.outputs, spec.role)
            self.assertTrue(spec.forbidden, spec.role)
            self.assertTrue(spec.tag, spec.role)

    def test_executor_cannot_release_itself(self) -> None:
        executor = next(spec for spec in AGENTS if spec.role == ROLE_EXECUTOR)
        self.assertIn("放行", executor.forbidden)
        self.assertIn("草稿", executor.outputs)

    def test_reviewer_cannot_write_the_answer(self) -> None:
        reviewer = next(spec for spec in AGENTS if spec.role == ROLE_REVIEWER)
        self.assertIn("代写", reviewer.forbidden)
        self.assertIn("passed", reviewer.outputs)

    def test_initiator_is_not_an_agent(self) -> None:
        """发起者只提交任务与证据：它不进角色表，也不算"参与角色"。"""
        self.assertNotIn(ROLE_INITIATOR, [spec.role for spec in AGENTS])

    def test_role_tags_are_distinct(self) -> None:
        tags = list(ROLE_TAGS.values())
        self.assertEqual(len(tags), len(set(tags)))

    def test_role_table_rendered(self) -> None:
        text = render_agents()
        for spec in AGENTS:
            self.assertIn(spec.name, text)
        self.assertIn("发起者", text)


# ---------------------------------------------------------------- ② 守门


class GuardTest(unittest.TestCase):
    """任务、轮次、预算三道守门：不合格就在让任何角色开口之前拦下。"""

    def setUp(self) -> None:
        self.registry = build_fake_registry()

    def test_empty_task_rejected_without_any_call(self) -> None:
        model = FakeModel()
        with self.assertRaises(ValueError):
            run_collaboration(model, self.registry, "  ")
        self.assertEqual(model.calls, 0)

    def test_non_string_task_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_collaboration(FakeModel(), self.registry, 123)

    def test_rounds_out_of_range(self) -> None:
        for rounds in (0, -1, MAX_ROUNDS + 1):
            with self.assertRaises(ValueError):
                validate_rounds(rounds)

    def test_rounds_must_be_int(self) -> None:
        with self.assertRaises(ValueError):
            validate_rounds("3")
        with self.assertRaises(ValueError):
            validate_rounds(True)

    def test_rounds_in_range_accepted(self) -> None:
        self.assertEqual(validate_rounds(1), 1)
        self.assertEqual(validate_rounds(MAX_ROUNDS), MAX_ROUNDS)

    def test_budget_must_be_positive(self) -> None:
        for budget in (0, -5):
            with self.assertRaises(ValueError):
                validate_budget(budget)
        with self.assertRaises(ValueError):
            validate_budget("900")

    def test_bad_rounds_stop_before_model_calls(self) -> None:
        model = FakeModel()
        with self.assertRaises(ValueError):
            run_collaboration(model, self.registry, TASK, rounds=0)
        self.assertEqual(model.calls, 0)


# ---------------------------------------------------------------- ③ 成本估算


class CostEstimateTest(unittest.TestCase):
    def test_estimate_scales_with_length(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("a" * 16), 10)
        self.assertGreater(estimate_tokens("a" * 160), estimate_tokens("a" * 16))

    def test_estimate_is_documented(self) -> None:
        """估算口径要写在常量里，课堂上能用计算器复算。"""
        self.assertAlmostEqual(CHARS_PER_TOKEN, 1.6)
        self.assertEqual(estimate_tokens("x" * 8), 5)

    def test_message_tokens_are_counted_from_content(self) -> None:
        chat = GroupChat(agents=[ChatAgent(ROLE_PLANNER, FakeModel())], max_round=3)
        message = chat.append(ROLE_PLANNER, "a" * 32)
        self.assertEqual(message.tokens, 20)


# ---------------------------------------------------------------- ④ 群聊机制


class GroupChatTest(unittest.TestCase):
    def _chat(self, max_round: int = 4, budget: int = DEFAULT_TOKEN_BUDGET) -> GroupChat:
        agents = [ChatAgent(spec.role, FakeModel()) for spec in AGENTS]
        return GroupChat(agents=agents, max_round=max_round, token_budget=budget)

    def test_turn_is_assigned_by_chat(self) -> None:
        chat = self._chat()
        self.assertEqual(chat.append(ROLE_INITIATOR, "任务").turn, 0)
        self.assertEqual(chat.append(ROLE_PLANNER, "规划").turn, 1)
        self.assertEqual(chat.append(ROLE_EXECUTOR, "草稿").turn, 2)

    def test_initiator_message_is_not_an_agent_message(self) -> None:
        chat = self._chat()
        chat.append(ROLE_INITIATOR, "任务")
        self.assertEqual(chat.agent_messages, [])
        self.assertEqual(chat.senders, [])

    def test_round_robin_speaking_order(self) -> None:
        chat = self._chat()
        chat.append(ROLE_INITIATOR, "任务")
        order = []
        for _ in range(3):
            speaker = chat.next_speaker()
            order.append(speaker.role)
            chat.append(speaker.role, "发言")
        self.assertEqual(order, [ROLE_PLANNER, ROLE_EXECUTOR, ROLE_REVIEWER])

    def test_max_round_counts_the_initiator_message(self) -> None:
        """口径与 AutoGen 一致：3 个角色各发言一次，要写 max_round = 4。"""
        chat = self._chat(max_round=DEFAULT_ROUNDS + 1)
        chat.append(ROLE_INITIATOR, "任务")
        for _ in range(3):
            chat.append(chat.next_speaker().role, "发言")
        self.assertTrue(chat.full)
        self.assertEqual(len(chat.messages), DEFAULT_ROUNDS + 1)

    def test_full_stops_one_message_earlier(self) -> None:
        chat = self._chat(max_round=3)
        chat.append(ROLE_INITIATOR, "任务")
        chat.append(ROLE_PLANNER, "规划")
        self.assertFalse(chat.full)
        chat.append(ROLE_EXECUTOR, "草稿")
        self.assertTrue(chat.full)

    def test_budget_flag(self) -> None:
        chat = self._chat(budget=5)
        chat.append(ROLE_INITIATOR, "a" * 16)
        self.assertTrue(chat.over_budget)

    def test_unknown_selection_method_rejected(self) -> None:
        agents = [ChatAgent(spec.role, FakeModel()) for spec in AGENTS]
        chat = GroupChat(agents=agents, max_round=4, speaker_selection_method="random")
        with self.assertRaises(ValueError):
            chat.next_speaker()

    def test_empty_group_rejected(self) -> None:
        chat = GroupChat(agents=[], max_round=2)
        with self.assertRaises(ValueError):
            chat.next_speaker()


# ---------------------------------------------------------------- ⑤ 复核协议


class VerdictTest(unittest.TestCase):
    def test_plain_json(self) -> None:
        verdict = parse_verdict('{"passed": true, "reason": "ok"}')
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["reason"], "ok")

    def test_fenced_json(self) -> None:
        verdict = parse_verdict('```json\n{"passed": false, "reason": "编造时限"}\n```')
        self.assertFalse(verdict["passed"])

    def test_non_json_is_rejected_not_passed(self) -> None:
        """解析失败必须判不通过：绝不能"看不懂就当通过"。"""
        verdict = parse_verdict("我觉得没问题")
        self.assertFalse(verdict["passed"])
        self.assertTrue(verdict.get("protocol_error"))

    def test_missing_bool_is_rejected(self) -> None:
        verdict = parse_verdict('{"reason": "看起来可以"}')
        self.assertFalse(verdict["passed"])
        self.assertTrue(verdict.get("protocol_error"))

    def test_json_array_is_rejected(self) -> None:
        self.assertFalse(parse_verdict("[1, 2]")["passed"])


# ---------------------------------------------------------------- ⑥ 协作行为


class CollaborationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_fake_registry()

    def test_normal_run_delivers(self) -> None:
        collab = run_collaboration(FakeModel(), self.registry, TASK)
        self.assertEqual(collab.stop_reason, STOP_REVIEWED)
        self.assertEqual(collab.status, STATUS_DELIVERED)
        self.assertTrue(collab.delivered)
        self.assertEqual(collab.answer, ANSWER_TEXT)

    def test_at_least_two_roles_spoke(self) -> None:
        collab = run_collaboration(FakeModel(), self.registry, TASK)
        self.assertGreaterEqual(len(collab.participants), 2)
        self.assertEqual(collab.participants, list(ROLE_ORDER))

    def test_messages_are_readable(self) -> None:
        collab = run_collaboration(FakeModel(), self.registry, TASK)
        self.assertTrue(all(message.sender for message in collab.messages))
        self.assertTrue(all(message.content.strip() for message in collab.messages))
        self.assertEqual([message.turn for message in collab.messages], [0, 1, 2, 3])

    def test_later_speakers_see_earlier_speech(self) -> None:
        """消息流的意义：后发言的人在提示词里看得到前面所有人的原话。"""
        model = FakeModel()
        run_collaboration(model, self.registry, TASK)
        prompts = model.prompts()
        self.assertNotIn(PLAN_TEXT, prompts[0])          # 规划员是第一位
        self.assertIn(PLAN_TEXT, prompts[1])             # 执行员看得到规划
        self.assertIn(PLAN_TEXT, prompts[2])             # 评审员看得到规划
        self.assertIn(ANSWER_TEXT, prompts[2])           # 评审员看得到草稿

    def test_evidence_is_fetched_before_the_chat(self) -> None:
        collab = run_collaboration(FakeModel(), self.registry, TASK)
        self.assertEqual(collab.evidence["tool_calls"], 2)
        self.assertIn(EVIDENCE_START, collab.messages[0].content)

    def test_model_calls_equal_speech_count(self) -> None:
        collab = run_collaboration(FakeModel(), self.registry, TASK)
        self.assertEqual(collab.model_calls, len(collab.agent_messages))
        self.assertEqual(collab.model_calls, 3)

    def test_manager_never_calls_the_model(self) -> None:
        collab = run_collaboration(FakeModel(), self.registry, TASK)
        self.assertEqual(collab.manager_model_calls, 0)

    def test_round_limit_leaves_no_review(self) -> None:
        collab = run_collaboration(FakeModel(), self.registry, TASK, rounds=2)
        self.assertEqual(collab.stop_reason, STOP_MAX_ROUND)
        self.assertEqual(collab.status, STATUS_MAX_ROUND)
        self.assertEqual(len(collab.messages), collab.max_round)
        self.assertFalse(collab.reviewed)
        self.assertEqual(collab.answer, "")

    def test_round_limit_is_not_paid_for_with_extra_call(self) -> None:
        model = FakeModel()
        collab = run_collaboration(model, self.registry, TASK, rounds=1)
        self.assertEqual(model.calls, 1)
        self.assertEqual(collab.model_calls, 1)

    def test_budget_limit_stops_before_paying_again(self) -> None:
        collab = run_collaboration(FakeModel(), self.registry, TASK, token_budget=40)
        self.assertEqual(collab.stop_reason, STOP_BUDGET)
        self.assertEqual(collab.status, STATUS_BUDGET)
        self.assertGreater(collab.tokens_used, collab.token_budget)
        self.assertLess(len(collab.messages), collab.max_round)
        self.assertEqual(collab.answer, "")

    def test_review_rejection_blocks_delivery(self) -> None:
        collab = run_collaboration(FakeModel(reviewer=(REVIEW_FAIL,)), self.registry, TASK)
        self.assertEqual(collab.stop_reason, STOP_REJECTED)
        self.assertEqual(collab.status, STATUS_REJECTED)
        self.assertFalse(collab.delivered)
        self.assertEqual(collab.answer, "")
        self.assertFalse(collab.review["passed"])

    def test_rejection_continues_the_chat_until_rounds_run_out(self) -> None:
        """复核不通过不等于立刻收尾：群聊继续，执行员再改一版，直到有人通过。"""
        collab = run_collaboration(
            FakeModel(
                planner=(PLAN_TEXT, PLAN_TEXT),
                executor=("第一版草稿", "第二版草稿"),
                reviewer=(REVIEW_FAIL, REVIEW_PASS),
            ),
            self.registry,
            TASK,
            rounds=6,
        )
        self.assertEqual(collab.stop_reason, STOP_REVIEWED)
        self.assertEqual(len(collab.review_history), 2)
        self.assertFalse(collab.review_history[0]["passed"])
        self.assertTrue(collab.review_history[1]["passed"])
        self.assertEqual(collab.answer, "第二版草稿")

    def test_rejection_loop_is_capped_by_rounds(self) -> None:
        """同一条坏草稿反复提交也不行：轮次到了就停，答复依然是空的。"""
        collab = run_collaboration(
            FakeModel(reviewer=(REVIEW_FAIL, REVIEW_FAIL)),
            self.registry,
            TASK,
            rounds=6,
        )
        self.assertEqual(collab.stop_reason, STOP_REJECTED)
        self.assertEqual(collab.answer, "")
        self.assertEqual(len(collab.messages), collab.max_round)
        self.assertEqual(len(collab.review_history), 2)

    def test_protocol_error_blocks_delivery(self) -> None:
        model = FakeModel()
        model.queues[ROLE_REVIEWER] = ["我看没问题"]
        collab = run_collaboration(model, self.registry, TASK)
        self.assertEqual(collab.stop_reason, STOP_REJECTED)
        self.assertEqual(collab.answer, "")

    def test_empty_reply_raises_protocol_error(self) -> None:
        model = FakeModel()
        model.queues[ROLE_EXECUTOR] = [""]
        collab = run_collaboration(model, self.registry, TASK)
        self.assertIn("空内容", collab.error)
        self.assertEqual(collab.answer, "")

    def test_gap_task_still_delivers_honest_answer(self) -> None:
        registry = build_fake_registry(found_ids=())
        collab = run_collaboration(FakeModel(), registry, "查 EC-FAQ-99 的标准答复")
        self.assertTrue(collab.delivered)
        self.assertIn("未找到", collab.evidence["faq"][0]["status"])

    def test_last_executor_message_wins(self) -> None:
        """多轮时以执行员的**最后**一版为答复，不取第一版。"""
        collab = run_collaboration(
            FakeModel(executor=("第一版草稿", "第二版草稿"), reviewer=(REVIEW_FAIL, REVIEW_PASS)),
            self.registry,
            TASK,
            rounds=6,
        )
        self.assertEqual(collab.answer, "第二版草稿")
        self.assertNotEqual(collab.answer, "第一版草稿")


# ---------------------------------------------------------------- ⑦ 证据收集


class EvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_fake_registry()

    def test_faq_ids_extracted(self) -> None:
        self.assertEqual(extract_faq_ids(TASK), ["EC-FAQ-01"])
        self.assertEqual(extract_faq_ids("EC-FAQ-01 和 EC-FAQ-02 都查一下"), ["EC-FAQ-01", "EC-FAQ-02"])
        self.assertEqual(extract_faq_ids("没有编号的问题"), [])

    def test_id_glued_to_chinese_still_matches(self) -> None:
        """中文紧贴编号时 `\\b` 匹配不到，这里必须能取出来。"""
        self.assertEqual(extract_faq_ids("查EC-FAQ-03的标准答复"), ["EC-FAQ-03"])

    def test_knowledge_query_is_short(self) -> None:
        self.assertEqual(knowledge_query("普通商品退货要准备什么"), "普通商品退货需要哪些材料，时限是多久")
        self.assertLessEqual(len(knowledge_query("随便问一个很长的没有关键词的问题" * 3)), 40)

    def test_tool_calls_count(self) -> None:
        evidence = collect_evidence(self.registry, TASK)
        self.assertEqual(evidence["tool_calls"], 2)
        self.assertEqual(len(evidence["faq"]), 1)
        self.assertEqual(evidence["faq"][0]["question_id"], "EC-FAQ-01")

    def test_without_id_only_search_runs(self) -> None:
        evidence = collect_evidence(self.registry, "普通商品退货要准备什么材料")
        self.assertEqual(evidence["tool_calls"], 1)
        self.assertEqual(evidence["faq"], [])

    def test_missing_id_reports_not_found(self) -> None:
        evidence = collect_evidence(build_fake_registry(found_ids=()), "查 EC-FAQ-99")
        self.assertEqual(evidence["faq"][0]["status"], "未找到")

    def test_malformed_id_is_not_treated_as_an_id(self) -> None:
        """编号不成形时不会被当成编号去查库：只做检索，工具调用 1 次。"""
        evidence = collect_evidence(self.registry, "查 EC-FAQ-1 的标准答复")
        self.assertEqual(evidence["faq"], [])
        self.assertEqual(evidence["tool_calls"], 1)


# ---------------------------------------------------------------- ⑧ 离线剧本模型


class ScriptedModelTest(unittest.TestCase):
    def _invoke(self, model: ScriptedChatModel, tag: str, payload: str = "{}") -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        return str(model.invoke([
            SystemMessage(content=f"{tag}\n你是这个角色"),
            HumanMessage(content=f"{EVIDENCE_START}{payload}"),
        ]).content)

    def test_planner_uses_script(self) -> None:
        model = ScriptedChatModel(planner=[PLAN_TEXT])
        self.assertEqual(self._invoke(model, ROLE_TAGS[ROLE_PLANNER]), PLAN_TEXT)

    def test_executor_uses_script_in_order(self) -> None:
        model = ScriptedChatModel(executor=["一", "二"])
        self.assertEqual(self._invoke(model, ROLE_TAGS[ROLE_EXECUTOR]), "一")
        self.assertEqual(self._invoke(model, ROLE_TAGS[ROLE_EXECUTOR]), "二")

    def test_reviewer_takes_verdicts_in_order(self) -> None:
        model = ScriptedChatModel(reviewer=[REVIEW_FAIL, REVIEW_PASS])
        first = json.loads(self._invoke(model, ROLE_TAGS[ROLE_REVIEWER]))
        second = json.loads(self._invoke(model, ROLE_TAGS[ROLE_REVIEWER]))
        self.assertFalse(first["passed"])
        self.assertTrue(second["passed"])

    def test_reviewer_default_is_pass_with_note(self) -> None:
        model = ScriptedChatModel()
        verdict = json.loads(self._invoke(model, ROLE_TAGS[ROLE_REVIEWER]))
        self.assertTrue(verdict["passed"])
        self.assertIn("离线剧本", verdict["reason"])

    def test_executor_default_extracts_evidence_only(self) -> None:
        model = ScriptedChatModel()
        payload = json.dumps({
            "faq": [{"question_id": "EC-FAQ-01", "answer": FAQ_ANSWER, "source_id": "DOC-02-01"}],
            "knowledge": {"hits": [{"source": "DOC-02-01-普通商品退货规则.md", "text": "签收后7天内"}]},
            "tool_calls": 2,
        }, ensure_ascii=False)
        text = self._invoke(model, ROLE_TAGS[ROLE_EXECUTOR], payload)
        self.assertIn(FAQ_ANSWER, text)
        self.assertIn("离线脚本模式", text)

    def test_planner_default_mentions_facts_count(self) -> None:
        model = ScriptedChatModel()
        payload = json.dumps({"faq": [{"question_id": "EC-FAQ-01", "answer": FAQ_ANSWER}]}, ensure_ascii=False)
        self.assertIn("规划", self._invoke(model, ROLE_TAGS[ROLE_PLANNER], payload))

    def test_unknown_tag_raises(self) -> None:
        """认不出角色就报错，不猜一个角色糊弄过去。"""
        model = ScriptedChatModel()
        with self.assertRaises(ChatProtocolError):
            self._invoke(model, "【角色：路人】")

    def test_factory_returns_scripted_model_in_mock(self) -> None:
        model = build_chat_model(ChatScript([PLAN_TEXT], [ANSWER_TEXT], [REVIEW_PASS]), "mock")
        self.assertIsInstance(model, ScriptedChatModel)
        self.assertEqual(model.planner, [PLAN_TEXT])

    def test_factory_rejects_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            build_chat_model(provider="openai")


# ---------------------------------------------------------------- ⑨ 渲染


class RenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.collab = run_collaboration(FakeModel(), build_fake_registry(), TASK)

    def test_transcript_prints_every_message(self) -> None:
        text = render_transcript(self.collab)
        self.assertIn(PLAN_TEXT, text)
        self.assertIn(ANSWER_TEXT, text)
        self.assertIn("评审员", text)
        self.assertIn("消息流", text)

    def test_transcript_keeps_evidence_once(self) -> None:
        """发起者那条只留摘要，证据全文由 render_evidence 出，避免同一段 JSON 印两遍。"""
        text = render_transcript(self.collab)
        self.assertIn("见上文", text)
        self.assertEqual(text.count('"tool_calls"'), 0)

    def test_evidence_render(self) -> None:
        text = render_evidence(self.collab.evidence)
        self.assertIn("工具调用 2 次", text)
        self.assertIn("EC-FAQ-01", text)

    def test_termination_shows_cost_and_reason(self) -> None:
        text = render_termination(self.collab)
        for marker in ("max_round", "模型调用", "估算 token", "终止原因", "状态"):
            self.assertIn(marker, text)

    def test_result_contains_all_sections(self) -> None:
        text = render_result(self.collab)
        for marker in ("平台工具证据", "消息流", "终止条件与成本"):
            self.assertIn(marker, text)

    def test_flow_and_concepts(self) -> None:
        flow = render_flow()
        for marker in ("GroupChat", "max_round", "复核通过"):
            self.assertIn(marker, flow)
        concepts = render_concepts()
        for marker in ("群聊", "轮次上限", "成本上限", "复核确认"):
            self.assertIn(marker, concepts)


# ---------------------------------------------------------------- ⑩ 管理器


class ManagerTest(unittest.TestCase):
    """管理器只转发与推进轮次：它自己不产生模型调用。"""

    def test_manager_stops_at_max_round(self) -> None:
        agents = [ChatAgent(spec.role, FakeModel()) for spec in AGENTS]
        chat = GroupChat(agents=agents, max_round=3)
        chat.append(ROLE_INITIATOR, "任务")
        manager = GroupChatManager(groupchat=chat)
        reason, verdict = manager.run(TASK, {})
        self.assertEqual(reason, STOP_MAX_ROUND)
        self.assertEqual(verdict, {})
        self.assertEqual(manager.model_calls, 0)
        self.assertEqual(len(chat.messages), 3)

    def test_manager_stops_before_budget_overflow(self) -> None:
        agents = [ChatAgent(spec.role, FakeModel()) for spec in AGENTS]
        chat = GroupChat(agents=agents, max_round=8, token_budget=30)
        chat.append(ROLE_INITIATOR, "a" * 64)
        reason, _verdict = GroupChatManager(groupchat=chat).run(TASK, {})
        self.assertEqual(reason, STOP_BUDGET)
        self.assertEqual(len(chat.messages), 1)


# ---------------------------------------------------------------- ⑪ 真实资料


class RealMaterialsTest(unittest.TestCase):
    """接上本组真实资料与单元15 的注册表再跑一遍；缺依赖或资料就跳过。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from app.chat_lab import build_stack
        except ImportError as error:  # pragma: no cover - 依赖缺失
            raise unittest.SkipTest(f"缺少依赖：{error}")
        try:
            cls.registry, _model, cls.kb, cls.rows = build_stack()
        except Exception as error:  # noqa: BLE001 - 资料或向量库不可用时跳过
            raise unittest.SkipTest(f"真实资料不可用：{error}")

    def test_materials_are_the_same_batch(self) -> None:
        self.assertEqual(self.kb, 7)
        self.assertEqual(self.rows, 3)

    def test_deliver_scenario_uses_real_evidence(self) -> None:
        from app.chat_lab import run_scenario

        collab = run_scenario("deliver", registry=self.registry, inject=True).collaboration
        self.assertIsNotNone(collab)
        self.assertTrue(collab.delivered)
        self.assertEqual(collab.evidence["tool_calls"], 2)
        self.assertEqual(collab.evidence["faq"][0]["source_id"], "DOC-02-01")
        self.assertGreaterEqual(len(collab.evidence["knowledge"]["hits"]), 1)

    def test_tight_scenario_stops_without_review(self) -> None:
        from app.chat_lab import run_scenario

        collab = run_scenario("tight", registry=self.registry, inject=True).collaboration
        self.assertEqual(collab.stop_reason, STOP_MAX_ROUND)
        self.assertFalse(collab.answer)
        self.assertEqual(len(collab.messages), collab.max_round)

    def test_budget_scenario_stops_before_review_but_rounds_remain(self) -> None:
        from app.chat_lab import run_scenario

        collab = run_scenario("budget", registry=self.registry, inject=True).collaboration
        self.assertEqual(collab.stop_reason, STOP_BUDGET)
        self.assertLess(len(collab.messages), collab.max_round)
        self.assertGreater(collab.tokens_used, collab.token_budget)

    def test_reject_scenario_delivers_nothing(self) -> None:
        from app.chat_lab import run_scenario

        collab = run_scenario("reject", registry=self.registry, inject=True).collaboration
        self.assertEqual(collab.stop_reason, STOP_REJECTED)
        self.assertEqual(collab.answer, "")

    def test_gap_scenario_is_delivered_but_honest(self) -> None:
        from app.chat_lab import run_scenario

        collab = run_scenario("gap", registry=self.registry, inject=True).collaboration
        self.assertTrue(collab.delivered)
        self.assertIn("未找到", collab.answer)

    def test_all_standards_pass(self) -> None:
        from app.chat_lab import run_class

        run = run_class(offline=True)
        self.assertTrue(run.standards)
        failed = [name for name, _evidence, verdict in run.standards if verdict != "通过"]
        self.assertEqual(failed, [], f"未通过的标准：{failed}")

    def test_invalid_inputs_are_rejected(self) -> None:
        from app.chat_lab import INVALID_CASES, run_invalid

        rows = run_invalid(self.registry)
        self.assertEqual(len(rows), len(INVALID_CASES))
        self.assertTrue(all("已拒绝" in verdict for _label, verdict in rows))
        self.assertTrue(all("模型调用 0 次" in verdict for _label, verdict in rows))

    def test_guard_cases_are_rejected(self) -> None:
        from app.chat_lab import GUARD_CASES, run_guards

        rows = run_guards(self.registry)
        self.assertEqual(len(rows), len(GUARD_CASES))
        self.assertTrue(all("已拒绝" in verdict for _label, verdict in rows))


# ---------------------------------------------------------------- ⑫ 真实模型


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
            from app.chat_lab import build_stack
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"缺少依赖：{error}")
        cls.registry, _model, _kb, _rows = build_stack()

    def test_real_model_runs_the_group_chat(self) -> None:
        from app.chat_lab import GOAL_GROUP, build_chat_model

        collab = run_collaboration(
            build_chat_model(provider="deepseek"), self.registry, GOAL_GROUP, rounds=3
        )
        self.assertGreaterEqual(len(collab.participants), 2)
        self.assertTrue(collab.evidence["faq"])
        self.assertEqual(collab.model_calls, len(collab.agent_messages))
        self.assertLessEqual(len(collab.messages), collab.max_round)
        # 复核说了通过才允许交付；不通过就必须是空的。
        if collab.review.get("passed"):
            self.assertTrue(collab.answer)
        else:
            self.assertEqual(collab.answer, "")
