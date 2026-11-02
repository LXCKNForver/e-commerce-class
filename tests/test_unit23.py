"""单元23 验收测试：安全读 CSV + 字段白名单 + 空值非数字提示 + 结果可复核。

覆盖任务卡的四条完成标准，默认全离线、不调用任何真实 API：

1. **字段白名单有效**：白名单内的字段跑得通；标识列（`case_id`/`order_id`）与表外
   字段在统计之前被拒绝。
2. **空值和非数字有提示**：逐格扫描出 `CellIssue`；数值聚合跳过并报数，分组计数
   归入「（空值）」——既不静默当 0，也不整行丢弃。
3. **统计结果可复核**：行数、算式、抽样明细齐全；解释里每个带单位的数字都要能在
   结果里找到出处。
4. **越权字段被拒绝**：拒绝发生在统计之前，且不再调用解释模型。

另外盯住一条底线：**不执行任意生成代码**——模块源码里不允许出现 `eval(` / `exec(`，
模型只能从 8 个固定算子里挑。
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path

from app.data_agent import (
    ALLOWED_FIELDS,
    ALLOWED_METRICS,
    COLUMNS,
    DATA_FILE,
    DATA_ROOT,
    DIRTY_FILE,
    EMPTY_GROUP,
    FORBIDDEN_FIELDS,
    GROUP_FIELDS,
    METRIC_COUNT,
    METRIC_GROUP_COUNT,
    METRIC_GROUP_MEAN,
    METRIC_GROUP_SUM,
    METRIC_MAX,
    METRIC_MEAN,
    METRIC_MIN,
    METRIC_SUM,
    NUMERIC_FIELDS,
    TAG_CHOOSE,
    TAG_EXPLAIN,
    AnalysisRejected,
    DataProtocolError,
    DataScript,
    ScriptedDataModel,
    allowed_numbers,
    build_data_model,
    check_explanation,
    compute,
    load_table,
    render_dictionary,
    render_flow,
    render_result,
    render_whitelist,
    resolve_source,
    run_analysis,
    validate_request,
    validate_question,
)
from app.data_lab import (
    FILE_CASES,
    REJECT_CASES,
    REQUEST_DIRTY,
    REQUEST_FORBIDDEN,
    REQUEST_HANDOFF,
    REQUEST_MISSING,
    REQUEST_OVERVIEW,
    REQUEST_REFUND,
    SCENARIOS,
    SCENARIO_ORDER,
    run_class,
    run_file_guards,
    run_invalid,
    run_rejects,
    run_scenario,
    source_without_execution,
)

TOTAL_ROWS = 32
QUESTION = "各类售后问题的工单分别有多少单？"


def script_of(request: dict | None = None, explanation: str = "") -> DataScript:
    return DataScript(
        requests=[request] if request is not None else [],
        explanations=[explanation] if explanation else [],
    )


def offline(script: DataScript):
    """离线剧本模型：显式 `provider="mock"`。

    为什么不能省略：`build_data_model` 默认读 `LLM_PROVIDER`，而别的测试模块会
    `load_dotenv()`，把 `.env` 里的 `deepseek` 带进 `os.environ`——全库一起跑时
    剧本场景会偷偷换成真实模型，数字随每次调用而变，用例也就失去意义。
    """
    return build_data_model(script, provider="mock")


def _invoke(model, tag: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    reply = model.invoke([SystemMessage(content=tag), HumanMessage(content="问题")])
    return str(reply.content)


class DictionaryTest(unittest.TestCase):
    """第1节的数据字典：字段、类型、能不能统计。"""

    def test_columns_match_the_dictionary(self) -> None:
        self.assertEqual(len(COLUMNS), 10)
        self.assertIn("case_id", COLUMNS)
        self.assertIn("refund_amount", COLUMNS)

    def test_id_columns_are_not_analyzable(self) -> None:
        self.assertEqual(set(FORBIDDEN_FIELDS), {"case_id", "order_id"})
        for name in FORBIDDEN_FIELDS:
            self.assertNotIn(name, ALLOWED_FIELDS)

    def test_whitelist_has_numeric_and_group_fields(self) -> None:
        self.assertEqual(set(NUMERIC_FIELDS), {"order_amount", "refund_amount"})
        self.assertIn("issue_type", GROUP_FIELDS)
        self.assertIn("month", GROUP_FIELDS)  # 派生字段，只用于分组

    def test_dictionary_renders_both_sides_of_the_whitelist(self) -> None:
        text = render_dictionary()
        self.assertIn("可分析字段（白名单）", text)
        self.assertIn("不在白名单", text)
        for name in ("case_id", "order_id"):
            self.assertIn(name, text)

    def test_whitelist_render_lists_every_metric(self) -> None:
        text = render_whitelist()
        for metric in ALLOWED_METRICS:
            self.assertIn(metric, text)
        self.assertIn("模型给不出路径", text)


class SourceGuardTest(unittest.TestCase):
    """数据目录与权限：模型给不出路径，只能给文件名。"""

    def test_normal_file_resolves_inside_the_data_root(self) -> None:
        path = resolve_source(DATA_FILE)
        self.assertTrue(path.is_file())
        self.assertTrue(Path(path).is_relative_to(DATA_ROOT))

    def test_dirty_sample_resolves(self) -> None:
        self.assertTrue(resolve_source(DIRTY_FILE).is_file())

    def test_every_file_case_is_rejected(self) -> None:
        for label, source in FILE_CASES:
            with self.subTest(case=label):
                with self.assertRaises(AnalysisRejected):
                    resolve_source(source)

    def test_traversal_is_rejected(self) -> None:
        with self.assertRaises(AnalysisRejected) as ctx:
            resolve_source("../../records.json")
        self.assertIn("不能带路径", str(ctx.exception))

    def test_non_csv_is_rejected(self) -> None:
        with self.assertRaises(AnalysisRejected) as ctx:
            resolve_source("电商售后工单.txt")
        self.assertIn("只允许读取 CSV", str(ctx.exception))

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(AnalysisRejected):
            resolve_source("不存在的文件.csv")


class LoadTest(unittest.TestCase):
    """读表：表头必须与数据字典一致，行数与列数要能对上。"""

    def test_clean_table_shape(self) -> None:
        table = load_table(DATA_ROOT / DATA_FILE)
        self.assertEqual(table.columns, COLUMNS)
        self.assertEqual(table.row_count, TOTAL_ROWS)
        self.assertEqual(table.issues, [])

    def test_dirty_table_has_issues(self) -> None:
        table = load_table(DATA_ROOT / DIRTY_FILE)
        self.assertEqual(table.row_count, TOTAL_ROWS)
        self.assertEqual(len(table.issues), 7)

    def test_month_is_derived_for_grouping(self) -> None:
        table = load_table(DATA_ROOT / DATA_FILE)
        self.assertEqual(table.rows[0]["month"], "2026-07")

    def test_header_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("case_id,order_id,refund_amount\nAS1,O1,10\n", encoding="utf-8")
            with self.assertRaises(AnalysisRejected) as ctx:
                load_table(path)
            self.assertIn("表头与数据字典不一致", str(ctx.exception))

    def test_empty_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(AnalysisRejected):
                load_table(path)


class IssueTest(unittest.TestCase):
    """标准二：空值和非数字要有提示，而不是静默当 0。"""

    def test_scan_finds_empty_and_non_numeric(self) -> None:
        table = load_table(DATA_ROOT / DIRTY_FILE)
        reasons = " ".join(issue.reason for issue in table.issues)
        self.assertIn("空值", reasons)
        self.assertIn("非数字", reasons)

    def test_issue_keeps_row_number_and_case_id(self) -> None:
        table = load_table(DATA_ROOT / DIRTY_FILE)
        first = table.issues[0]
        self.assertEqual(first.row_no, 4)  # 第 1 行是表头
        self.assertEqual(first.case_id, "AS23003")
        self.assertIn("refund_amount", first.render())

    def test_empty_value_is_skipped_not_counted_as_zero(self) -> None:
        """坏值被跳过而不是当 0：均值必须比"拿总和除以 32 行"更高。"""
        dirty = load_table(DATA_ROOT / DIRTY_FILE)
        result = compute(dirty, validate_request(REQUEST_DIRTY))
        self.assertEqual(result.rows_skipped, 3)
        self.assertEqual(result.rows_used, 29)
        total = result.rows_used * result.value  # 参与计算的金额总和
        self.assertAlmostEqual(total, 4708, places=4)
        self.assertGreater(result.value, total / TOTAL_ROWS)

    def test_empty_category_goes_to_a_named_bucket(self) -> None:
        """分组字段为空的行不能被丢掉：工单不能凭空少一单。"""
        table = load_table(DATA_ROOT / DIRTY_FILE)
        result = compute(table, validate_request(
            {"metric": METRIC_GROUP_COUNT, "field": "issue_type", "filters": {}}
        ))
        self.assertIn(EMPTY_GROUP, [stat.name for stat in result.groups])
        self.assertEqual(sum(stat.count for stat in result.groups), TOTAL_ROWS)


class RequestGuardTest(unittest.TestCase):
    """标准一与标准四：字段白名单与算子白名单。"""

    def test_every_reject_case_is_rejected(self) -> None:
        for label, payload in REJECT_CASES:
            with self.subTest(case=label):
                with self.assertRaises(AnalysisRejected):
                    validate_request(payload)

    def test_id_column_is_rejected_with_whitelist_wording(self) -> None:
        with self.assertRaises(AnalysisRejected) as ctx:
            validate_request(REQUEST_FORBIDDEN)
        self.assertIn("白名单", str(ctx.exception))

    def test_unknown_metric_is_rejected(self) -> None:
        with self.assertRaises(AnalysisRejected) as ctx:
            validate_request({"metric": "median", "field": "refund_amount"})
        self.assertIn("算子未授权", str(ctx.exception))

    def test_code_like_metric_is_rejected(self) -> None:
        """模型想在算子字段里塞一段表达式：这不是算子，是代码——拒。"""
        with self.assertRaises(AnalysisRejected):
            validate_request({"metric": "sum(x for x in rows)", "field": "refund_amount"})

    def test_allowed_requests_pass(self) -> None:
        for payload in (REQUEST_OVERVIEW, REQUEST_REFUND, REQUEST_HANDOFF, REQUEST_DIRTY):
            with self.subTest(metric=payload["metric"]):
                request = validate_request(payload)
                self.assertEqual(request.metric, payload["metric"])

    def test_each_metric_has_a_render(self) -> None:
        for metric in sorted(ALLOWED_METRICS):
            with self.subTest(metric=metric):
                payload: dict = {"metric": metric}
                if metric in (METRIC_SUM, METRIC_MEAN, METRIC_MAX, METRIC_MIN):
                    payload["field"] = "refund_amount"
                elif metric in (METRIC_GROUP_COUNT, METRIC_GROUP_SUM, METRIC_GROUP_MEAN):
                    payload["field"] = "issue_type"
                    if metric in (METRIC_GROUP_SUM, METRIC_GROUP_MEAN):
                        payload["value_field"] = "refund_amount"
                self.assertTrue(validate_request(payload).render())

    def test_numeric_metric_rejects_category_field(self) -> None:
        with self.assertRaises(AnalysisRejected):
            validate_request({"metric": METRIC_SUM, "field": "issue_type"})

    def test_group_metric_rejects_id_column(self) -> None:
        with self.assertRaises(AnalysisRejected):
            validate_request({"metric": METRIC_GROUP_SUM, "field": "case_id",
                              "value_field": "refund_amount"})

    def test_filter_field_must_be_a_category(self) -> None:
        with self.assertRaises(AnalysisRejected):
            validate_request({"metric": METRIC_COUNT, "filters": {"order_amount": "199"}})


class ComputeTest(unittest.TestCase):
    """统计本身：八个算子各自算得对，跳过与空结果都要如实报。"""

    def setUp(self) -> None:
        self.table = load_table(DATA_ROOT / DATA_FILE)

    def _run(self, payload: dict):
        return compute(self.table, validate_request(payload))

    def test_count(self) -> None:
        result = self._run({"metric": METRIC_COUNT, "filters": {}})
        self.assertEqual(result.value, TOTAL_ROWS)
        self.assertEqual(result.rows_used, TOTAL_ROWS)

    def test_count_with_filter(self) -> None:
        result = self._run(REQUEST_HANDOFF)
        self.assertEqual(result.rows_matched, 16)
        self.assertEqual(result.value, 16)

    def test_sum_with_filter(self) -> None:
        result = self._run(REQUEST_REFUND)
        self.assertEqual(result.rows_matched, 16)
        self.assertEqual(result.value, 5544)

    def test_mean_matches_sum_divided_by_rows(self) -> None:
        summed = self._run(REQUEST_REFUND)
        mean = self._run({"metric": METRIC_MEAN, "field": "refund_amount",
                          "filters": {"status": "已退款"}})
        self.assertAlmostEqual(mean.value, summed.value / summed.rows_used, places=6)
        self.assertIn("÷", mean.formula)

    def test_max_and_min(self) -> None:
        top = self._run({"metric": METRIC_MAX, "field": "order_amount"})
        bottom = self._run({"metric": METRIC_MIN, "field": "order_amount"})
        self.assertGreater(top.value, bottom.value)
        self.assertIn("最大值", top.formula)
        self.assertIn("最小值", bottom.formula)
        self.assertTrue(top.sample)  # 极值所在行要能复核

    def test_group_count(self) -> None:
        result = self._run(REQUEST_OVERVIEW)
        self.assertEqual(len(result.groups), 6)
        self.assertEqual(sum(stat.count for stat in result.groups), TOTAL_ROWS)

    def test_group_sum(self) -> None:
        result = self._run({"metric": METRIC_GROUP_SUM, "field": "issue_type",
                            "value_field": "refund_amount"})
        self.assertTrue(all(stat.value is not None for stat in result.groups))
        total = sum(stat.value for stat in result.groups)
        self.assertAlmostEqual(total, 5544, places=6)

    def test_group_mean(self) -> None:
        result = self._run({"metric": METRIC_GROUP_MEAN, "field": "issue_type",
                            "value_field": "refund_amount"})
        self.assertTrue(all(stat.value is not None for stat in result.groups))

    def test_month_grouping(self) -> None:
        result = self._run({"metric": METRIC_GROUP_COUNT, "field": "month"})
        self.assertTrue(result.groups)
        self.assertEqual(sum(stat.count for stat in result.groups), TOTAL_ROWS)

    def test_zero_rows_is_not_an_error(self) -> None:
        result = self._run(REQUEST_MISSING)
        self.assertEqual(result.rows_matched, 0)
        self.assertIsNone(result.value)
        self.assertTrue(result.warnings)

    def test_group_sum_skips_bad_values(self) -> None:
        dirty = load_table(DATA_ROOT / DIRTY_FILE)
        result = compute(dirty, validate_request(
            {"metric": METRIC_GROUP_SUM, "field": "issue_type", "value_field": "refund_amount"}
        ))
        self.assertEqual(result.rows_skipped, 3)
        self.assertTrue(any("未计入分组聚合" in warning for warning in result.warnings))


class ReviewableTest(unittest.TestCase):
    """标准三：结果要能拿计算器对上，解释里的数字要有出处。"""

    def test_result_carries_rows_formula_and_sample(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_OVERVIEW, "共 {rows_total} 单。")), QUESTION)
        self.assertEqual(result.rows_total, TOTAL_ROWS)
        self.assertTrue(result.formula)
        self.assertEqual(len(result.sample), 5)

    def test_formula_shows_the_division_for_mean(self) -> None:
        result = run_analysis(
            offline(script_of(REQUEST_DIRTY, "均值 {value} {unit}。")),
            "平均每单退款金额是多少元？",
            source=DIRTY_FILE,
        )
        self.assertIn("÷", result.formula)
        self.assertIn("跳过 3 行", result.formula)

    def test_allowed_numbers_contains_the_result(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_OVERVIEW, "共 {rows_total} 单。")), QUESTION)
        numbers = allowed_numbers(result)
        self.assertIn(float(TOTAL_ROWS), numbers)
        self.assertIn(8.0, numbers)  # 最大组的单数

    def test_explanation_with_a_strange_number_is_flagged(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_OVERVIEW, "共 {rows_total} 单。")), QUESTION)
        notes = check_explanation("本月新增 9999 元，增长 87.5%", result)
        self.assertTrue(notes)
        self.assertIn("9999", notes[0])

    def test_explanation_with_only_result_numbers_is_clean(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_OVERVIEW, "共 {rows_total} 单。")), QUESTION)
        self.assertEqual(check_explanation(f"共 {TOTAL_ROWS} 单，其中 8 单，占 25.0%。", result), [])

    def test_dates_are_not_treated_as_statistics(self) -> None:
        """2026 年 7 月这种日期数字不该被判成编造。"""
        result = run_analysis(offline(script_of(REQUEST_OVERVIEW, "共 {rows_total} 单。")), QUESTION)
        self.assertEqual(check_explanation("统计区间是 2026 年 7 月到 10 月。", result), [])

    def test_sample_carries_the_filter_value(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_HANDOFF, "{value} {unit}。")),
                              "需要转人工复核的工单有多少单？")
        self.assertTrue(all("needs_handoff=是" in shown for _case, shown in result.sample))


class ScriptedModelTest(unittest.TestCase):
    """离线剧本模型：按步骤标记分派，队列取完有默认值。"""

    def test_choose_step_returns_the_queued_request(self) -> None:
        model = ScriptedDataModel(requests=[REQUEST_OVERVIEW], explanations=["解释"])
        request = validate_request(json.loads(_invoke(model, TAG_CHOOSE)))
        self.assertEqual(request.metric, METRIC_GROUP_COUNT)
        self.assertEqual(request.field, "issue_type")

    def test_explain_step_returns_the_queued_text(self) -> None:
        model = ScriptedDataModel(requests=[REQUEST_OVERVIEW], explanations=["第一句"])
        self.assertEqual(_invoke(model, TAG_EXPLAIN), "第一句")

    def test_explanation_template_uses_real_numbers(self) -> None:
        model = ScriptedDataModel(requests=[REQUEST_OVERVIEW], explanations=["共 {rows_total} 单"])
        object.__setattr__(model, "context", {"rows_total": 32})
        self.assertEqual(_invoke(model, TAG_EXPLAIN), "共 32 单")

    def test_queue_is_consumed_in_order(self) -> None:
        model = ScriptedDataModel(explanations=["第一句", "第二句"])
        self.assertEqual(_invoke(model, TAG_EXPLAIN), "第一句")
        self.assertEqual(_invoke(model, TAG_EXPLAIN), "第二句")

    def test_unknown_step_raises(self) -> None:
        model = ScriptedDataModel(requests=[REQUEST_OVERVIEW])
        with self.assertRaises(DataProtocolError):
            _invoke(model, "【步骤：别的】")

    def test_calls_are_counted(self) -> None:
        model = ScriptedDataModel(requests=[REQUEST_OVERVIEW], explanations=["解释"])
        _invoke(model, TAG_CHOOSE)
        self.assertEqual(model.calls, 1)


class RunAnalysisTest(unittest.TestCase):
    """端到端：六条路径各自跑一遍，各自留下该有的痕迹。"""

    def test_overview_delivers_six_groups(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_OVERVIEW, "共 {rows_total} 单。")), QUESTION)
        self.assertTrue(result.ok)
        self.assertEqual(len(result.groups), 6)
        self.assertEqual(result.choose_calls, 1)
        self.assertEqual(result.explain_calls, 1)
        self.assertEqual(result.model_calls, 2)

    def test_refund_sum(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_REFUND, "合计 {value} {unit}。")),
                              "已经退款的工单，退款金额一共是多少元？")
        self.assertEqual(result.value, 5544)
        self.assertEqual(result.unit, "元")

    def test_dirty_reports_skipped_rows(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_DIRTY, "跳过 {rows_skipped} 行。")),
                              "平均每单退款金额是多少元？", source=DIRTY_FILE)
        self.assertEqual(result.rows_skipped, 3)
        self.assertEqual(len(result.issues), 7)
        self.assertEqual(result.number_notes, [])

    def test_forbidden_field_is_rejected_before_any_math(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_FORBIDDEN, "")),
                              "按订单号统计一下每个订单出现了多少次？")
        self.assertTrue(result.rejected)
        self.assertIn("白名单", result.error)
        self.assertIsNone(result.value)
        self.assertEqual(result.rows_total, 0)
        self.assertEqual(result.explanation, "")  # 被拒后不再调解释模型

    def test_missing_rows_is_not_a_failure(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_MISSING, "没有匹配到行。")),
                              "已经关闭的工单有多少单？")
        self.assertFalse(result.rejected)
        self.assertEqual(result.rows_matched, 0)
        self.assertTrue(result.warnings)

    def test_injected_request_costs_no_choose_call(self) -> None:
        """故障注入的口径是直接给的：这一步根本不该问模型。"""
        result = run_analysis(offline(script_of(REQUEST_DIRTY, "均值 {value} {unit}。")),
                              "平均每单退款金额是多少元？",
                              source=DIRTY_FILE, request_payload=REQUEST_DIRTY)
        self.assertEqual(result.choose_calls, 0)
        self.assertEqual(result.explain_calls, 1)
        self.assertEqual(result.model_calls, 1)

    def test_bad_question_is_rejected_before_the_model(self) -> None:
        model = offline(script_of(REQUEST_OVERVIEW, "解释"))
        with self.assertRaises(ValueError):
            run_analysis(model, "")
        self.assertEqual(model.calls, 0)

    def test_protocol_error_is_recorded_not_raised(self) -> None:
        model = offline(DataScript(requests=["不是 JSON"], explanations=[]))
        result = run_analysis(model, QUESTION)
        self.assertFalse(result.rejected)
        self.assertIn("不是 JSON", result.error)


class NoCodeExecutionTest(unittest.TestCase):
    """底线：不执行任意生成代码。"""

    def test_module_has_no_eval_or_exec(self) -> None:
        import app.data_agent as module

        text = inspect.getsource(module)
        for forbidden in ("eval(", "exec(", "__import__("):
            self.assertNotIn(forbidden, text)

    def test_source_without_execution_reports_clean(self) -> None:
        clean, note = source_without_execution()
        self.assertTrue(clean)
        self.assertIn("没有", note)

    def test_metric_whitelist_is_closed(self) -> None:
        self.assertEqual(len(ALLOWED_METRICS), 8)
        self.assertNotIn("median", ALLOWED_METRICS)
        self.assertNotIn("quantile", ALLOWED_METRICS)


class RenderTest(unittest.TestCase):
    """渲染出来的东西要能直接拿去上课。"""

    def test_render_result_shows_the_whole_chain(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_OVERVIEW, "共 {rows_total} 单。")), QUESTION)
        text = render_result(result)
        for marker in ("统计口径", "行数", "算式（可手工复核）", "抽样复核", "业务解释", "数字核验"):
            self.assertIn(marker, text)

    def test_render_result_marks_rejection(self) -> None:
        result = run_analysis(offline(script_of(REQUEST_FORBIDDEN, "")), QUESTION)
        text = render_result(result)
        self.assertIn("已拒绝", text)
        self.assertIn("统计：0 次", text)

    def test_render_flow_mentions_no_code(self) -> None:
        text = render_flow()
        self.assertIn("选口径", text)
        self.assertIn("eval", text)

    def test_question_guard(self) -> None:
        with self.assertRaises(ValueError):
            validate_question("查")
        self.assertEqual(validate_question("  " + QUESTION + " "), QUESTION)


class LabTest(unittest.TestCase):
    """编排层：四节课跑完，四条标准逐条通过。"""

    def test_run_class_offline_is_reproducible(self) -> None:
        run = run_class(offline=True)
        self.assertEqual(len(run.scenarios), len(SCENARIO_ORDER))
        self.assertEqual(run.row_count, TOTAL_ROWS)
        self.assertEqual(run.column_count, len(COLUMNS))

    def test_all_standards_pass(self) -> None:
        run = run_class(offline=True)
        self.assertEqual(len(run.standards), 6)
        for name, evidence, verdict in run.standards:
            with self.subTest(standard=name):
                self.assertEqual(verdict, "通过", f"{name}：{evidence}")

    def test_reject_cases_are_all_rejected(self) -> None:
        rows = run_rejects()
        self.assertEqual(len(rows), len(REJECT_CASES))
        self.assertTrue(all(verdict.startswith("已拒绝") for _label, verdict in rows))

    def test_file_cases_are_all_rejected(self) -> None:
        rows = run_file_guards()
        self.assertEqual(len(rows), len(FILE_CASES))
        self.assertTrue(all(verdict.startswith("已拒绝") for _label, verdict in rows))

    def test_invalid_questions_cost_no_model_call(self) -> None:
        rows = run_invalid()
        self.assertTrue(rows)
        self.assertTrue(all("模型调用 0 次" in verdict for _label, verdict in rows))

    def test_every_scenario_runs(self) -> None:
        for key in SCENARIO_ORDER:
            with self.subTest(scenario=key):
                # 显式给剧本模型：不传的话会按 LLM_PROVIDER 走，全库一起跑时会被污染。
                row = run_scenario(key, model=offline(SCENARIOS[key].script))
                self.assertIsNotNone(row.result)
                self.assertEqual(row.error, "")

    def test_scenario_questions_are_different(self) -> None:
        questions = {spec.question for spec in SCENARIOS.values()}
        self.assertEqual(len(questions), len(SCENARIOS))


class RealModelTest(unittest.TestCase):
    """真实 DeepSeek 的用例：要显式开启（RUN_REAL_MODEL=1）且配了密钥才跑。"""

    @classmethod
    def setUpClass(cls) -> None:
        if (os.getenv("RUN_REAL_MODEL", "") or "").strip() != "1":
            raise unittest.SkipTest("未显式开启真实模型用例（RUN_REAL_MODEL=1）")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise unittest.SkipTest("未配置 DEEPSEEK_API_KEY")

    def test_real_model_chooses_and_explains(self) -> None:
        result = run_analysis(build_data_model(provider="deepseek"), QUESTION)
        self.assertTrue(result.ok)
        self.assertEqual(result.choose_calls, 1)
        self.assertEqual(result.explain_calls, 1)
        self.assertTrue(result.explanation)
        self.assertEqual(result.rows_total, TOTAL_ROWS)

    def test_real_model_never_picks_a_forbidden_field(self) -> None:
        for question in (QUESTION, "退款金额一共多少元？", "哪个渠道的工单最多？"):
            with self.subTest(question=question):
                result = run_analysis(build_data_model(provider="deepseek"), question)
                if result.request is not None:
                    self.assertNotIn(result.request.field, FORBIDDEN_FIELDS)
                    self.assertIn(result.request.metric, ALLOWED_METRICS)


if __name__ == "__main__":
    unittest.main()
