"""单元20 验收测试：计划生成、逐步执行、失败后至多重规划一次、未解决就终止报错。

覆盖任务卡的四条完成标准，默认全离线、不调用任何真实 API：

1. **计划可逐步执行**：几步计划就执行几次工具，步号连续、参数与返回值都留痕。
2. **能查看工具调用与观察结果**：每条观察记工具名、参数、真实返回值与耗时。
3. **失败后至多重规划一次**：重规划次数上限 1，改完从第一步重跑，旧观察留作证据。
4. **未解决的失败不会被误报为完成**：终止时状态 aborted、答复为空、不调总结模型。

白名单与参数格式校验都走单元15 的真代码（`ToolRegistry.call` → `authorize_tool`），
测试里只替换"工具执行函数"和"模型"，这样测出来的行为就是课堂上的行为。
"""

from __future__ import annotations

import json
import os
import unittest

from langchain_core.messages import AIMessage

from app.plan_solver import (
    MAX_PLAN_STEPS,
    MAX_REPLANS,
    STATUS_ABORTED,
    STATUS_DONE,
    PlanAborted,
    ScriptedPlanModel,
    build_plan_model,
    execute_plan,
    make_plan,
    plan_step,
    render_concepts,
    render_plan,
    render_result,
    render_trajectory,
    solve,
    validate_goal,
)
from app.toolkit import (
    TOOL_DEADLINE,
    TOOL_FAQ,
    TOOL_SEARCH,
    ToolRejected,
    ToolRegistry,
    ToolSpec,
)

GOAL = "查一下 EC-FAQ-01 的标准答复，并说明处理时要关注什么"
FAQ_ANSWER = "课程样例中签收后7天内且未使用可申请退货评估，最终由人工复核"


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
    """把若干步骤包成模型应输出的 JSON（可带 Markdown 围栏）。"""
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
        # 与真实工具一样：天数不在资料原文里就拒绝，不让模型口算。
        if window_days != "7":
            raise ToolRejected(f"{basis_doc} 里没有“{window_days}天”这个时限")
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


# ---------------------------------------------------------------- ① 计划生成


class PlanningTest(unittest.TestCase):
    """make_plan：模型给的是数据，程序负责把它变成可执行的白名单计划。"""

    def setUp(self) -> None:
        self.registry = build_fake_registry()

    def test_empty_goal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_goal("")

    def test_non_string_goal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_goal(123)

    def test_too_short_goal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_goal("查")

    def test_too_long_goal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_goal("查" * 400)

    def test_goal_is_normalized(self) -> None:
        self.assertEqual(validate_goal("  EC-FAQ-01   的答复  "), "EC-FAQ-01 的答复")

    def test_fenced_json_is_parsed(self) -> None:
        model = FakeModel(
            f"```json\n{plan_json(plan_step(TOOL_FAQ, question_id='EC-FAQ-01'))}\n```"
        )
        plan = make_plan(model, self.registry, GOAL)
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual(plan.steps[0].tool, TOOL_FAQ)

    def test_step_count_upper_bound(self) -> None:
        many = [plan_step(TOOL_FAQ, question_id="EC-FAQ-01") for _ in range(MAX_PLAN_STEPS + 1)]
        with self.assertRaises(ValueError):
            make_plan(FakeModel(plan_json(*many)), self.registry, GOAL)

    def test_non_whitelisted_tool_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            make_plan(
                FakeModel(plan_json(plan_step("delete_collection", name="faq"))),
                self.registry,
                GOAL,
            )
        self.assertIn("非白名单", str(ctx.exception))

    def test_bad_argument_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_plan(
                FakeModel(plan_json({"tool": TOOL_FAQ, "arguments": {"question_id": True}})),
                self.registry,
                GOAL,
            )

    def test_integer_argument_is_accepted(self) -> None:
        plan = make_plan(
            FakeModel(plan_json({
                "tool": TOOL_DEADLINE,
                "arguments": {"sign_date": "2026-12-01", "window_days": 7,
                              "basis_doc": "DOC-02-01-普通商品退货规则.md"},
            })),
            self.registry,
            GOAL,
        )
        # JSON 里 7 和 "7" 是一回事：宽的是类型，不是值。
        self.assertEqual(plan.steps[0].arguments["window_days"], "7")

    def test_reason_is_kept(self) -> None:
        plan = make_plan(
            FakeModel(plan_json(plan_step(TOOL_FAQ, "拿到标准答复", question_id="EC-FAQ-01"))),
            self.registry,
            GOAL,
        )
        self.assertEqual(plan.steps[0].reason, "拿到标准答复")

    def test_model_sees_tool_descriptions(self) -> None:
        model = FakeModel(plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")))
        make_plan(model, self.registry, GOAL)
        self.assertIn(TOOL_FAQ, model.texts()[0])
        self.assertIn("EC-FAQ-01", model.texts()[0])

    def test_plan_carries_no_executable(self) -> None:
        """计划只是数据：没有 execute 字段，模型给不出函数、路径或 SQL。"""
        plan = make_plan(
            FakeModel(plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01"))),
            self.registry,
            GOAL,
        )
        self.assertEqual([field for field in plan.steps[0].__dataclass_fields__],
                         ["tool", "arguments", "reason"])


# ---------------------------------------------------------------- ② 执行与重规划


class ExecutionTest(unittest.TestCase):
    """execute_plan / solve：逐步执行、失败重规划一次、仍失败就终止。"""

    def setUp(self) -> None:
        self.registry = build_fake_registry()

    def test_steps_run_in_order(self) -> None:
        run = solve(GOAL, self.registry, FakeModel(
            plan_json(
                plan_step(TOOL_FAQ, question_id="EC-FAQ-01"),
                plan_step(TOOL_SEARCH, query="退货需要哪些材料"),
            ),
            "答复：7天",
        ))
        self.assertEqual(run.status, STATUS_DONE)
        self.assertEqual([item.step for item in run.steps], [1, 2])
        self.assertEqual([item.tool for item in run.steps], [TOOL_FAQ, TOOL_SEARCH])

    def test_every_step_records_arguments_and_observation(self) -> None:
        run = solve(GOAL, self.registry, FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")), "答复：7天",
        ))
        step = run.steps[0]
        self.assertEqual(step.arguments, {"question_id": "EC-FAQ-01"})
        self.assertIn(FAQ_ANSWER, json.dumps(step.observation, ensure_ascii=False))
        self.assertGreaterEqual(step.elapsed_ms, 0)

    def test_failure_triggers_exactly_one_replan(self) -> None:
        run = solve(GOAL, self.registry, FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),      # ① 格式错 → 被拒
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),  # ② 修订后正确
            "答复：7天",
        ))
        self.assertTrue(run.replanned)
        self.assertEqual(run.replans, MAX_REPLANS)
        self.assertEqual(run.status, STATUS_DONE)
        self.assertEqual(len(run.failures), 1)
        self.assertEqual(len(run.steps), 1)

    def test_replan_feedback_contains_error(self) -> None:
        model = FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            "答复：7天",
        )
        solve(GOAL, self.registry, model)
        self.assertIn("执行失败", model.texts()[1])
        self.assertIn("格式不正确", model.texts()[1])

    def test_second_failure_aborts(self) -> None:
        run = solve(GOAL, self.registry, FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),  # 修订后还是错的
        ))
        self.assertEqual(run.status, STATUS_ABORTED)
        self.assertFalse(run.finished)
        self.assertEqual(run.answer, "")
        self.assertIn("仍失败", run.error)

    def test_aborted_run_skips_summary_call(self) -> None:
        """终止时连总结那一次模型调用都不发生——"没答完"不能伪装成"答完了"。"""
        model = FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
            "不会被用到的答复",
        )
        run = solve(GOAL, self.registry, model)
        self.assertEqual(run.model_calls, 2)  # 1 次规划 + 1 次重规划，没有总结
        self.assertEqual(model.calls, 2)

    def test_valid_but_missing_id_is_not_a_failure(self) -> None:
        """编号格式合法但库里没有：只是"未找到"，不该触发重规划。"""
        run = solve(GOAL, self.registry, FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-99")), "答复：未找到",
        ))
        self.assertEqual(run.status, STATUS_DONE)
        self.assertEqual(run.replans, 0)
        self.assertEqual(run.steps[0].observation["status"], "未找到")

    def test_observations_from_failed_attempt_are_kept(self) -> None:
        """第1轮已成功的步骤留作审计证据，不会被重规划冲掉。"""
        run = solve(GOAL, self.registry, FakeModel(
            plan_json(
                plan_step(TOOL_FAQ, question_id="EC-FAQ-01"),
                plan_step(TOOL_DEADLINE, sign_date="2026-12-01", window_days="30",
                          basis_doc="DOC-02-01-普通商品退货规则.md"),
            ),
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            "答复：7天",
        ))
        self.assertEqual(len(run.steps), 2)   # 第1轮第1步 + 第2轮第1步
        self.assertEqual(len(run.failures), 1)
        self.assertEqual([item.attempt for item in run.steps], [1, 2])

    def test_cursor_resets_after_replan(self) -> None:
        run = solve(GOAL, self.registry, FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            "答复：7天",
        ))
        self.assertEqual(sorted({item.attempt for item in run.steps}), [2])
        self.assertEqual(run.plan.cursor, len(run.plan.steps))

    def test_invalid_goal_never_calls_model(self) -> None:
        model = FakeModel(plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")))
        with self.assertRaises(ValueError):
            solve("", self.registry, model)
        self.assertEqual(model.calls, 0)

    def test_raise_on_aborted_directly(self) -> None:
        """直接调 execute_plan 时，修订后仍失败是抛异常，不是返回半截结果。"""
        model = FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
        )
        plan = make_plan(model, self.registry, GOAL)
        with self.assertRaises(PlanAborted):
            execute_plan(model, self.registry, plan)

    def test_model_call_count_is_explicit(self) -> None:
        run = solve(GOAL, self.registry, FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")), "答复：7天",
        ))
        self.assertEqual(run.model_calls, 2)  # 1 次规划 + 1 次总结

    def test_offline_script_model_outputs_plan_then_answer(self) -> None:
        """离线剧本模型：先出计划 JSON，再给只含工具原文的答复。"""
        model = ScriptedPlanModel(steps=[plan_step(TOOL_FAQ, question_id="EC-FAQ-01")])
        run = solve(GOAL, self.registry, model)
        self.assertEqual(run.status, STATUS_DONE)
        self.assertIn("离线脚本模式", run.answer)
        self.assertIn(FAQ_ANSWER, run.answer)


# ---------------------------------------------------------------- ③ 渲染


class RenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run = solve(GOAL, build_fake_registry(), FakeModel(
            plan_json(plan_step(TOOL_FAQ, question_id="BAD")),
            plan_json(plan_step(TOOL_FAQ, question_id="EC-FAQ-01")),
            "答复：7天",
        ))

    def test_result_covers_key_fields(self) -> None:
        text = render_result(self.run)
        for marker in ("目标", "状态", "是否重规划", "模型调用", "答复"):
            self.assertIn(marker, text)

    def test_plan_table_lists_every_step(self) -> None:
        text = render_plan(self.run.plan)
        self.assertIn(TOOL_FAQ, text)
        self.assertIn("第1轮计划", text)
        self.assertIn("第2轮计划", text)  # 修订前后都要看得见

    def test_trajectory_orders_by_attempt(self) -> None:
        text = render_trajectory(self.run)
        self.assertLess(text.index("第1轮\t1"), text.index("第2轮\t1"))

    def test_trajectory_marks_failure(self) -> None:
        self.assertIn("失败", render_trajectory(self.run))
        self.assertIn("格式不正确", render_trajectory(self.run))

    def test_concepts_table(self) -> None:
        text = render_concepts()
        for marker in ("规划", "求解", "观察", "重规划", "终止"):
            self.assertIn(marker, text)


# ---------------------------------------------------------------- ④ 真实资料


class RealMaterialsTest(unittest.TestCase):
    """接上本组真实资料与单元15 的注册表再跑一遍；缺依赖就跳过。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from app.plan_lab import build_stack
        except ImportError as error:  # pragma: no cover - 依赖缺失
            raise unittest.SkipTest(f"缺少依赖：{error}")
        try:
            cls.registry, _model, cls.kb, cls.rows = build_stack()
        except Exception as error:  # noqa: BLE001 - 资料或向量库不可用时跳过
            raise unittest.SkipTest(f"真实资料不可用：{error}")

    def test_materials_are_the_same_batch(self) -> None:
        self.assertEqual(self.kb, 7)
        self.assertEqual(self.rows, 3)

    def test_minimal_scenario_runs_one_step(self) -> None:
        from app.plan_lab import run_scenario

        scenario = run_scenario("minimal", registry=self.registry, inject=True)
        self.assertIsNotNone(scenario.run)
        self.assertEqual(scenario.run.status, STATUS_DONE)
        self.assertEqual(len(scenario.run.steps), 1)
        self.assertIn("7天", scenario.run.answer)

    def test_multi_scenario_runs_three_steps(self) -> None:
        from app.plan_lab import run_scenario

        scenario = run_scenario("multi", registry=self.registry, inject=True)
        self.assertEqual(len(scenario.run.steps), MAX_PLAN_STEPS)
        self.assertIn("2026-12-08", json.dumps(
            [item.observation for item in scenario.run.steps], ensure_ascii=False
        ))

    def test_replan_scenario_recovers(self) -> None:
        from app.plan_lab import run_scenario

        scenario = run_scenario("replan", registry=self.registry, inject=True)
        self.assertEqual(scenario.run.replans, 1)
        self.assertEqual(scenario.run.status, STATUS_DONE)
        self.assertTrue(scenario.run.answer)

    def test_abort_scenario_reports_not_completes(self) -> None:
        from app.plan_lab import run_scenario

        scenario = run_scenario("abort", registry=self.registry, inject=True)
        self.assertEqual(scenario.run.status, STATUS_ABORTED)
        self.assertEqual(scenario.run.answer, "")
        self.assertFalse(scenario.run.finished)

    def test_missing_id_does_not_replan(self) -> None:
        from app.plan_lab import run_scenario

        scenario = run_scenario("missing", registry=self.registry, inject=True)
        self.assertEqual(scenario.run.replans, 0)
        self.assertEqual(scenario.run.status, STATUS_DONE)
        self.assertIn("未找到", scenario.run.steps[0].preview())

    def test_all_standards_pass(self) -> None:
        from app.plan_lab import run_class

        run = run_class(offline=True)
        self.assertTrue(run.standards)
        failed = [name for name, _evidence, verdict in run.standards if verdict != "通过"]
        self.assertEqual(failed, [], f"未通过的标准：{failed}")

    def test_invalid_inputs_are_rejected(self) -> None:
        from app.plan_lab import INVALID_CASES, run_invalid

        rows = run_invalid(self.registry)
        self.assertEqual(len(rows), len(INVALID_CASES))
        self.assertTrue(all("已拒绝" in verdict for _label, verdict in rows))


# ---------------------------------------------------------------- ⑤ 真实模型


class RealModelTest(unittest.TestCase):
    """真实 DeepSeek 的用例：要显式开启（RUN_REAL_MODEL=1）且配了密钥才跑。

    开关刻意不用 `LLM_PROVIDER`——那个变量会被别的用例从 `.env` 读进来，
    整套测试的总数就会随执行顺序变，"多少项通过"是要写进证据文档的数字，不能飘。
    """

    @classmethod
    def setUpClass(cls) -> None:
        if (os.getenv("RUN_REAL_MODEL", "") or "").strip() != "1":
            raise unittest.SkipTest("未显式开启真实模型用例（RUN_REAL_MODEL=1）")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise unittest.SkipTest("未配置 DEEPSEEK_API_KEY")
        try:
            from app.plan_lab import build_stack
        except ImportError as error:  # pragma: no cover
            raise unittest.SkipTest(f"缺少依赖：{error}")
        cls.registry, _model, _kb, _rows = build_stack(
            model=build_plan_model(provider="deepseek")
        )

    def test_real_model_plans_and_solves(self) -> None:
        from app.plan_lab import GOAL_MINIMAL

        run = solve(GOAL_MINIMAL, self.registry, build_plan_model(provider="deepseek"))
        self.assertEqual(run.status, STATUS_DONE)
        self.assertTrue(run.steps)
        self.assertIn("7天", run.answer)


if __name__ == "__main__":
    unittest.main()
