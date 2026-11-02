"""单元23 的编排层：装本组 CSV → 按四节课跑六条路径 → 逐条验收。

任务卡的四节课在本文件里串成一条：

    第1节  看懂数据分析 Agent：数据字典、字段白名单、固定算子（`/data schema`）
    第2节  定分析问题与允许字段，先跑最小业务场景（`/data` 不带参数）
    第3节  看统计与解释的完整输入输出（`/data refund`、`/data handoff`）
    第4节  验证空值、坏数据与越权字段（`/data dirty`、`/data forbidden`、`/data bad`）

    数据侧：`模拟数据/单元23实践数据/电商售后服务智能体项目组/`（任务卡给的本组 CSV
            + 一份派生的脏数据样本，只用于第4节）
    实现侧：app/data_agent.py（守门 → 选口径 → 校验 → 统计 → 解释 → 数字核验）
    
全程**不执行任何生成代码**：模型只输出"选了哪个算子、哪个字段"，Python 按固定算子
算；`app/data_agent.py` 里没有 `eval(` / `exec(`（有一条测试盯着这件事）。

    uv run --extra langchain python -m app.data_lab
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field as dc_field
from typing import Any

from app.data_agent import (
    ALLOWED_METRICS,
    DATA_FILE,
    DIRTY_FILE,
    DataScript,
    MAX_FILE_BYTES,
    METRIC_COUNT,
    METRIC_GROUP_COUNT,
    METRIC_MEAN,
    METRIC_SUM,
    AnalysisResult,
    build_data_model,
    render_concepts,
    render_dictionary,
    render_flow,
    render_result,
    render_whitelist,
    resolve_source,
    run_analysis,
    validate_request,
)
from app.plan_lab import _shown

# 守门的坏问题用例：与计划单元同一套规则（空 / 空白 / 非字符串 / 过短 / 超长），
# 但措辞换成"问题"——本单元被守的是分析问题，不是目标。
INVALID_CASES: list[tuple[str, Any]] = [
    ("空问题", ""),
    ("纯空白问题", "   "),
    ("非字符串问题", 123),
    ("过短问题", "查"),
    ("超长问题", "查" * 400),
]

# ---------------------------------------------------------------- ① 本组分析问题

# 第2节的最小场景：只问"各类问题各多少单"，一次分组计数就能答。
QUESTION_OVERVIEW = "各类售后问题的工单分别有多少单？"
# 第3节的两个场景：一个数值聚合、一个带筛选的计数。
QUESTION_REFUND = "已经退款的工单，退款金额一共是多少元？"
QUESTION_HANDOFF = "需要转人工复核的工单有多少单？"
# 第4节：脏数据（空值与非数字）、越权字段、筛选后没有数据。
QUESTION_DIRTY = "平均每单退款金额是多少元？"
QUESTION_FORBIDDEN = "按订单号统计一下每个订单出现了多少次？"
QUESTION_MISSING = "已经关闭的工单有多少单？"

# ---------------------------------------------------------------- ② 离线剧本

# 口径：模型在离线模式下"选"的东西（真实模式下由模型自己输出，不用这些）。
REQUEST_OVERVIEW = {"metric": METRIC_GROUP_COUNT, "field": "issue_type", "value_field": "", "filters": {}}
REQUEST_REFUND = {"metric": METRIC_SUM, "field": "refund_amount", "value_field": "", "filters": {"status": "已退款"}}
REQUEST_HANDOFF = {"metric": METRIC_COUNT, "field": "", "value_field": "", "filters": {"needs_handoff": "是"}}
REQUEST_DIRTY = {"metric": METRIC_MEAN, "field": "refund_amount", "value_field": "", "filters": {}}
# 越权：模型要了订单编号——表里真有这一列，但它不在白名单。
REQUEST_FORBIDDEN = {"metric": METRIC_GROUP_COUNT, "field": "order_id", "value_field": "", "filters": {}}
# 筛选后 0 行：状态取值里没有"已关闭"，结果为空而不是出错。
REQUEST_MISSING = {"metric": METRIC_COUNT, "field": "", "value_field": "", "filters": {"status": "已关闭"}}

# 解释模板：{rows_total} 之类会被真实算出来的数字替换（离线剧本不写死数字）。
EXPLAIN_OVERVIEW = (
    "（离线剧本）共 {rows_total} 单工单，{top_name} 最多：{top_value} 单，占 {top_ratio}；"
    "各组单数相加正好等于 {rows_total} 单，可逐组核对。"
)
EXPLAIN_REFUND = (
    "（离线剧本）已退款工单 {rows_matched} 单，退款金额合计 {value} {unit}；"
    "参与计算的 {rows_used} 行 refund_amount 都是数字，没有跳过任何一行。"
)
EXPLAIN_HANDOFF = (
    "（离线剧本）需要转人工的工单 {value} {unit}，占全部 {rows_total} 单的 {top_ratio}；"
    "筛选条件 needs_handoff=是 逐行匹配得到，可按工单编号复核。"
)
EXPLAIN_DIRTY = (
    "（离线剧本）{rows_total} 行里退款金额有 {rows_skipped} 行为空值或非数字，已跳过；"
    "均值按 {rows_used} 行有效数值算得 {value} {unit}，脏数据没有被当成 0 参与平均。"
)
EXPLAIN_MISSING = (
    "（离线剧本）按 status=已关闭 筛选后没有匹配到任何行：表里没有这个状态取值，"
    "所以结果为空——这不是出错，请核对状态取值（已退款 / 处理中 / 人工复核）。"
)

# ---------------------------------------------------------------- ③ 场景表


@dataclass(frozen=True)
class ScenarioSpec:
    """一个演示场景：问什么、读哪份文件、用什么口径、要不要故障注入。"""

    key: str
    label: str
    question: str
    source: str
    request: dict[str, Any] | None
    explanation: str
    inject: bool
    why: str

    @property
    def script(self) -> DataScript:
        return DataScript(
            requests=[self.request] if self.request is not None else [],
            explanations=[self.explanation] if self.explanation else [],
        )


SCENARIOS: dict[str, ScenarioSpec] = {
    "overview": ScenarioSpec(
        "overview",
        "最小场景：按问题类型分组计数",
        QUESTION_OVERVIEW,
        DATA_FILE,
        REQUEST_OVERVIEW,
        EXPLAIN_OVERVIEW,
        False,
        "最小可跑通的一次分析：只用一个分组计数，就能看出工单都集中在哪几类问题上",
    ),
    "refund": ScenarioSpec(
        "refund",
        "数值统计：已退款工单的退款金额合计",
        QUESTION_REFUND,
        DATA_FILE,
        REQUEST_REFUND,
        EXPLAIN_REFUND,
        False,
        "数值聚合 + 筛选：金额必须由 Python 相加，模型只说「要算哪个字段」",
    ),
    "handoff": ScenarioSpec(
        "handoff",
        "带筛选的计数：需要转人工的工单",
        QUESTION_HANDOFF,
        DATA_FILE,
        REQUEST_HANDOFF,
        EXPLAIN_HANDOFF,
        False,
        "筛选条件也走白名单：只能用分类字段，标识列和数值字段都不行",
    ),
    "dirty": ScenarioSpec(
        "dirty",
        "脏数据：空值与非数字有提示，跳过并如实报数",
        QUESTION_DIRTY,
        DIRTY_FILE,
        REQUEST_DIRTY,
        EXPLAIN_DIRTY,
        True,
        "空值与非数字既不静默当 0 也不整行丢弃：跳过多少行、跳的哪几行，全部列出来",
    ),
    "forbidden": ScenarioSpec(
        "forbidden",
        "越权字段：订单编号被白名单拦下",
        QUESTION_FORBIDDEN,
        DATA_FILE,
        REQUEST_FORBIDDEN,
        "",
        True,
        "表里真有 order_id 这一列，但它不在白名单：统计在发生之前就被拒绝",
    ),
    "missing": ScenarioSpec(
        "missing",
        "筛选后 0 行：如实说没有数据，不报错",
        QUESTION_MISSING,
        DATA_FILE,
        REQUEST_MISSING,
        EXPLAIN_MISSING,
        True,
        "查不到 ≠ 失败：筛选值在表里不存在时，结果是空的，但要说清楚为什么空",
    ),
}
SCENARIO_ORDER = ("overview", "refund", "handoff", "dirty", "forbidden", "missing")

# 口径越权的用例：直接喂给 validate_request（不经模型，结果可复现、统计 0 次）。
REJECT_CASES: list[tuple[str, Any]] = [
    ("表外字段 customer_phone", {"metric": METRIC_GROUP_COUNT, "field": "customer_phone"}),
    ("标识列 case_id", {"metric": METRIC_GROUP_COUNT, "field": "case_id"}),
    ("标识列 order_id（分组求和）", {"metric": "group_mean", "field": "order_id", "value_field": "refund_amount"}),
    ("未知算子 median", {"metric": "median", "field": "refund_amount"}),
    ("未知算子（想自己算）", {"metric": "sum(x for x in rows)", "field": "refund_amount"}),
    ("筛选用了标识列", {"metric": METRIC_COUNT, "filters": {"case_id": "AS23001"}}),
    ("筛选用了数值字段", {"metric": METRIC_COUNT, "filters": {"order_amount": "199"}}),
    ("缺字段（mean 没给字段）", {"metric": METRIC_MEAN}),
    ("group_sum 缺数值字段", {"metric": "group_sum", "field": "issue_type"}),
    ("字段不是字符串", {"metric": METRIC_MEAN, "field": 199}),
    ("筛选值超长", {"metric": METRIC_COUNT, "filters": {"status": "已" * 60}}),
    ("口径不是 JSON 对象", "group_count"),
]

# 数据目录与权限的用例：模型给不出路径，只能给文件名，越界一律拒绝。
FILE_CASES: list[tuple[str, Any]] = [
    ("目录穿越", "../../../records.json"),
    ("绝对路径", "C:/somewhere/records.json"),
    ("带目录的文件名", "子目录/电商售后工单.csv"),
    ("非 CSV 后缀", "电商售后工单.txt"),
    ("文件不存在", "不存在的文件.csv"),
    ("空文件名", ""),
    ("文件名不是字符串", 123),
    ("以点开头的隐藏文件", ".env"),
]

CONCEPTS_EXTRA: list[tuple[str, str, str]] = [
    ("本单元用哪份数据？", f"{DATA_FILE}（任务卡给的本组 CSV，32 行 × 10 列）；脏数据用派生样本", "第1节"),
    ("能统计哪些字段？", "商品品类、问题类型、订单金额、退款金额、处理状态、来源渠道、是否转人工、提交月份", "第1节"),
    ("为什么订单号不能统计？", "标识列不是度量，也没有业务含义，还涉及客户信息", "第4节 forbidden"),
    ("空值怎么处理？", "逐格记录并提示；数值聚合跳过、分组归入「（空值）」", "第4节 dirty"),
    ("解释里的数字从哪来？", "只能来自 Python 算出的结果，每个带单位的数字都要能找到出处", "第3节"),
    ("为什么不让它写代码？", "模型只选算子与字段；执行路径是固定的 8 个算子，没有 eval/exec", "第1节"),
]

QUESTION_LIST = [
    ("本单元分析什么数据？", f"{DATA_FILE}：32 行电商售后工单（第1节数据字典）"),
    ("字段白名单怎么生效？", "只有 8 个可分析字段；标识列与表外字段在统计前被拒（第4节）"),
    ("空值和非数字会怎样？", "逐格提示，数值聚合跳过、分组归入「（空值）」，不静默当 0（第4节）"),
    ("结果怎么复核？", "总行数 / 筛选后行数 / 参与行数 + 算式 + 前 5 行抽样（第2、3节）"),
    ("解释会不会编数字？", "每个带单位的数字都要能在结果里找到出处，否则点名列出（第3节）"),
    ("有没有执行生成的代码？", "没有：模型只输出 JSON 口径，模块里没有 eval/exec（第1节）"),
]


# ---------------------------------------------------------------- ④ 数据结构


@dataclass
class ScenarioResult:
    key: str
    label: str
    question: str
    spec: ScenarioSpec
    result: AnalysisResult | None = None
    error: str = ""


@dataclass
class DataLabRun:
    scenarios: list[ScenarioResult] = dc_field(default_factory=list)
    reject_rows: list[tuple[str, str]] = dc_field(default_factory=list)
    file_rows: list[tuple[str, str]] = dc_field(default_factory=list)
    invalid_rows: list[tuple[str, str]] = dc_field(default_factory=list)
    file_name: str = DATA_FILE
    dirty_name: str = DIRTY_FILE
    row_count: int = 0
    column_count: int = 0
    model_name: str = ""
    provider: str = "mock"
    key_configured: bool = False
    standards: list[tuple[str, str, str]] = dc_field(default_factory=list)


# ---------------------------------------------------------------- ⑤ 运行


def _model_for(spec: ScenarioSpec, model=None):
    """挑模型：故障注入场景一律用剧本；其余按 `LLM_PROVIDER` 走。

    为什么 dirty/forbidden/missing 要注入：真实模型不会主动去读一份脏样本，
    也不会故意要一个越权字段——那三条"异常路径"就演示不到，且每次数字都不一样，
    证据没法复核。overview / refund / handoff 三条留给真实模型自己选口径。
    """
    if spec.inject:
        return build_data_model(spec.script, provider="mock")
    if model is not None:
        return model
    return build_data_model(spec.script)


def run_scenario(
    key: str,
    question: str | None = None,
    model=None,
    inject: bool | None = None,
) -> ScenarioResult:
    spec = SCENARIOS[key]
    if inject is True and not spec.inject:
        spec = ScenarioSpec(
            spec.key, spec.label, spec.question, spec.source, spec.request,
            spec.explanation, True, spec.why,
        )
    result_row = ScenarioResult(key=key, label=spec.label, question=question or spec.question, spec=spec)
    try:
        result_row.result = run_analysis(
            _model_for(spec, model),
            result_row.question,
            source=spec.source,
            request_payload=spec.request if spec.inject else None,
        )
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        result_row.error = f"{type(error).__name__}：{error}"
    return result_row


def run_rejects() -> list[tuple[str, str]]:
    """口径越权：直接喂给 validate_request，一次统计都不会发生。"""
    rows: list[tuple[str, str]] = []
    for label, payload in REJECT_CASES:
        try:
            request = validate_request(payload)
            rows.append((label, f"竟然通过了（异常！）：{request.render()}"))
        except ValueError as error:
            rows.append((label, f"已拒绝：{error}"))
    return rows


def run_file_guards() -> list[tuple[str, str]]:
    """数据目录与权限：模型给不出路径，只能给文件名，越界一律拒绝。"""
    rows: list[tuple[str, str]] = []
    for label, source in FILE_CASES:
        try:
            path = resolve_source(source)
            rows.append((label, f"竟然通过了（异常！）：{path}"))
        except ValueError as error:
            shown = source if isinstance(source, str) else repr(source)
            rows.append((f"{label}（{shown[:40]}）", f"已拒绝：{error}"))
    return rows


def run_invalid() -> list[tuple[str, str]]:
    """问题守门：坏问题在问模型之前就被拦下，模型调用 0 次。"""
    rows: list[tuple[str, str]] = []
    for label, raw in INVALID_CASES:
        model = build_data_model(DataScript(requests=[REQUEST_OVERVIEW]), provider="mock")
        try:
            run_analysis(model, raw)
            rows.append((f"{label}（{_shown(raw)}）", "竟然跑起来了（异常！）"))
        except ValueError as error:
            rows.append((f"{label}（{_shown(raw)}）", f"已拒绝：{error}；模型调用 {model.calls} 次"))
    return rows


def run_class(model=None, offline: bool = False) -> DataLabRun:
    """按任务卡顺序跑完四节课：最小场景 → 数值统计 → 脏数据 → 越权 → 空结果。"""
    run = DataLabRun(
        provider=(os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower(),
        key_configured=bool(os.getenv("DEEPSEEK_API_KEY")),
    )
    from app.data_agent import load_table, DATA_ROOT

    table = load_table(DATA_ROOT / DATA_FILE)
    run.row_count, run.column_count = table.row_count, len(table.columns)
    for key in SCENARIO_ORDER:
        result_row = run_scenario(key, model=model, inject=True if offline else None)
        run.scenarios.append(result_row)
        if result_row.result is not None and not run.model_name:
            run.model_name = result_row.result.model_name
    run.reject_rows = run_rejects()
    run.file_rows = run_file_guards()
    run.invalid_rows = run_invalid()
    run.standards = check_standards(run)
    return run


def source_without_execution() -> tuple[bool, str]:
    """核对"不执行任意生成代码"：模块源码里不允许出现 eval( / exec(。"""
    import app.data_agent as module

    text = inspect.getsource(module)
    # 只查"会执行字符串"的那三个入口；re.compile( 是正则，不算。
    found = [name for name in ("eval(", "exec(", "__import__(") if name in text]
    return (not found), ("、".join(found) if found else "data_agent 源码里没有 eval( / exec( / __import__(")


def check_standards(run: DataLabRun) -> list[tuple[str, str, str]]:
    """四条完成标准逐条给结论；结论只从跑出来的数据里取，不手写数字。"""
    by_key = {row.key: row for row in run.scenarios}
    overview = by_key.get("overview")
    refund = by_key.get("refund")
    dirty = by_key.get("dirty")
    forbidden = by_key.get("forbidden")
    missing = by_key.get("missing")

    rows: list[tuple[str, str, str]] = []

    # 标准一：字段白名单有效——白名单内的字段跑得通，白名单外的字段一次都不算。
    whitelist_ok = bool(
        overview and overview.result and overview.result.ok
        and overview.result.request is not None
        and overview.result.request.field in ("issue_type", "product_category", "status",
                                              "channel", "needs_handoff", "month")
    )
    reject_count = sum(1 for _label, verdict in run.reject_rows if verdict.startswith("已拒绝"))
    rows.append((
        "字段白名单有效",
        (
            f"白名单内：{overview.result.request.field} 跑通（{len(overview.result.groups)} 组、"
            f"{overview.result.rows_total} 行）"
            if overview and overview.result and overview.result.request else "最小场景未跑起来"
        ) + f"；越权用例 {reject_count}/{len(run.reject_rows)} 条被拒（统计 0 次）",
        "通过" if (whitelist_ok and reject_count == len(run.reject_rows)) else "不通过",
    ))

    # 标准二：空值和非数字有提示——不只是"跳过了"，还要点名哪一行哪一列。
    dirty_result = dirty.result if dirty else None
    issue_text = " ".join(issue.render() for issue in (dirty_result.issues if dirty_result else []))
    dirty_ok = bool(
        dirty_result and dirty_result.ok
        and dirty_result.rows_skipped > 0
        and dirty_result.issues
        and "空值" in issue_text and "非数字" in issue_text
    )
    rows.append((
        "空值和非数字有提示",
        (
            f"{dirty_result.source}：跳过 {dirty_result.rows_skipped} 行，提示 {len(dirty_result.issues)} 处"
            f"（含空值与非数字两类）；参与计算 {dirty_result.rows_used} 行"
            if dirty_result else "脏数据场景未跑起来"
        ),
        "通过" if dirty_ok else "不通过",
    ))

    # 标准三：统计结果可复核——行数、算式、抽样齐全，且解释里没有来历不明的数字。
    overview_result = overview.result if overview else None
    refund_result = refund.result if refund else None
    reviewable = bool(
        overview_result and overview_result.ok
        and overview_result.formula and overview_result.sample
        and overview_result.rows_total and overview_result.rows_used
        and not overview_result.number_notes
    )
    rows.append((
        "统计结果可复核",
        (
            f"行数 {overview_result.rows_total} → 筛选后 {overview_result.rows_matched} → "
            f"参与 {overview_result.rows_used}；算式：{overview_result.formula}"
            if overview_result else "最小场景未跑起来"
        ) + "；" + (
            f"抽样 {len(overview_result.sample)} 行（{overview_result.sample[0][0]}="
            f"{overview_result.sample[0][1]}…）"
            if overview_result and overview_result.sample else "无抽样"
        ) + "；" + (
            f"解释核验：{'、'.join(refund_result.number_notes) if refund_result and refund_result.number_notes else '每个带单位的数字都有出处'}"
            if refund_result else "数值场景未跑起来"
        ),
        "通过" if reviewable else "不通过",
    ))

    # 标准四：越权字段被拒绝——拒绝发生在统计之前，且不再调用解释模型。
    forbidden_result = forbidden.result if forbidden else None
    forbidden_ok = bool(
        forbidden_result and forbidden_result.rejected
        and "白名单" in forbidden_result.error
        and forbidden_result.rows_total == 0
        and not forbidden_result.explanation  # 被拒后不再调用解释模型
    )
    rows.append((
        "越权字段被拒绝",
        (
            f"order_id（表里存在、不在白名单）：{forbidden_result.error}"
            if forbidden_result and forbidden_result.error else "越权场景未跑起来"
        ) + "；" + (
            f"统计 0 次、解释未调用（模型调用 {forbidden_result.model_calls} 次）"
            if forbidden_result else ""
        ),
        "通过" if forbidden_ok else "不通过",
    ))

    # 附带一：不执行任意生成代码。
    clean, note = source_without_execution()
    rows.append((
        "不执行任意生成代码",
        f"{note}；模型只输出 JSON 口径（{len(ALLOWED_METRICS)} 个固定算子 + 白名单字段），"
        f"执行路径写死在 compute() 里",
        "通过" if clean else "不通过",
    ))

    # 附带二：筛选后 0 行 ≠ 失败。
    missing_result = missing.result if missing else None
    rows.append((
        "筛选后 0 行 ≠ 失败",
        (
            f"筛选 status=已关闭：共 {missing_result.rows_total} 行、匹配 {missing_result.rows_matched} 行"
            f"（不是被拒绝，是表里没有这个状态）；提醒：{'、'.join(missing_result.warnings)}"
            if missing_result else "空结果场景未跑起来"
        ),
        "通过" if (missing_result and missing_result.rows_matched == 0
                   and not missing_result.rejected and missing_result.warnings) else "不通过",
    ))
    return rows


# ---------------------------------------------------------------- ⑥ 渲染


def render_header(run: DataLabRun) -> str:
    provider = "DeepSeek 真实调用" if run.provider == "deepseek" else "离线脚本模式"
    return "\n".join([
        "单元23 数据分析 Agent（电商售后服务智能体项目组）",
        f"模型：{run.model_name or '—'}（{provider}；密钥已配置：{run.key_configured}）",
        f"数据：{run.file_name}（{run.row_count} 行 × {run.column_count} 列）；"
        f"脏数据样本 {run.dirty_name}（第4节用）",
        f"权限：只在本组数据目录内读取，单个文件上限 {MAX_FILE_BYTES} 字节；"
        f"算子固定 {len(ALLOWED_METRICS)} 个",
    ])


def render_scenario(row: ScenarioResult) -> str:
    title = f"【{row.label}】"
    if row.spec.inject:
        title += "（故障注入：口径与解释用剧本，真实模型不受影响）"
    if row.result is None:
        return f"{title}\n问题：{row.question}\n未运行：{row.error}"
    return "\n".join([
        title,
        f"为什么这么设计：{row.spec.why}",
        "",
        render_result(row.result),
    ])


def render_summary_table(run: DataLabRun) -> str:
    lines = [
        "六条路径横向对照",
        "场景\t数据文件\t口径\t总行\t筛选后\t参与\t跳过\t结果\t统计\t模型调用",
    ]
    for row in run.scenarios:
        result = row.result
        if result is None:
            lines.append(f"{row.label}\t—\t—\t—\t—\t—\t—\t未运行\t—\t{row.error}")
            continue
        if result.rejected:
            lines.append(
                f"{row.label}\t{result.source}\t越权被拒\t—\t—\t—\t—\t拒绝\t0 次\t{result.model_calls}"
            )
            continue
        answer = "—"
        if result.groups:
            answer = f"{len(result.groups)} 组"
        elif result.value is not None:
            answer = f"{result.value:.2f} {result.unit}".replace(".00 ", " ")
        lines.append(
            f"{row.label}\t{result.source}\t"
            f"{result.request.render() if result.request else '—'}\t"
            f"{result.rows_total}\t{result.rows_matched}\t{result.rows_used}\t{result.rows_skipped}\t"
            f"{answer}\t{'是' if result.has_data else '否'}\t{result.model_calls}"
        )
    return "\n".join(lines)


def render_standards(run: DataLabRun) -> str:
    lines = ["完成标准逐条验收", "标准\t实测\t结论"]
    lines.extend(f"{name}\t{evidence}\t{verdict}" for name, evidence, verdict in run.standards)
    return "\n".join(lines)


def render_rejects(rows: list[tuple[str, str]]) -> str:
    lines = ["越权与非法口径（直接喂给 validate_request，统计 0 次）", "输入\t结果"]
    lines.extend(f"{label}\t{verdict}" for label, verdict in rows)
    return "\n".join(lines)


def render_file_guards(rows: list[tuple[str, str]]) -> str:
    lines = ["数据目录与权限（模型给不出路径，只能给文件名）", "输入\t结果"]
    lines.extend(f"{label}\t{verdict}" for label, verdict in rows)
    return "\n".join(lines)


def render_invalid(rows: list[tuple[str, str]]) -> str:
    lines = ["无效问题（守门在问模型之前）", "输入\t结果"]
    lines.extend(f"{label}\t{verdict}" for label, verdict in rows)
    return "\n".join(lines)


def render_question_list() -> str:
    lines = ["第1节要交的项目问题清单", "问题\t在哪一节验证"]
    lines.extend(f"{question}\t{where}" for question, where in QUESTION_LIST)
    return "\n".join(lines)


def render_concept_table() -> str:
    lines = ["概念记录", "概念\t含义\t在本项目里落在哪"]
    lines.extend(f"{question}\t{answer}\t{where}" for question, answer, where in CONCEPTS_EXTRA)
    lines.append("")
    lines.append(render_concepts())
    return "\n".join(lines)


def _main() -> None:
    run = run_class()
    print(render_header(run))
    print()
    print(render_flow())
    print()
    print(render_dictionary())
    print()
    print(render_whitelist())
    print()
    print(render_question_list())
    print()
    for row in run.scenarios:
        print(render_scenario(row))
        print()
    print(render_summary_table(run))
    print()
    print(render_standards(run))
    print()
    print(render_rejects(run.reject_rows))
    print()
    print(render_file_guards(run.file_rows))
    print()
    print(render_invalid(run.invalid_rows))
    print()
    print(render_concept_table())
    print()
    print("模型只输出「选了哪个算子、哪个字段」：执行路径是固定的 8 个算子，没有 eval/exec。")
    print("空值与非数字既不静默当 0，也不整行丢弃：跳过多少行、跳的哪几行，全部列出来。")


if __name__ == "__main__":
    _main()
