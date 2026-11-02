"""单元23：数据分析 Agent —— 安全读取本组 CSV，只做白名单内的统计。

任务卡的原话是"让 Agent 安全读取项目 CSV，完成指定统计并用业务语言解释结果，
**不执行任意生成代码**"。这句话里的重点是"不执行任意生成代码"，所以本单元的做法是：

    业务问题 → ① 模型只**选口径**（从固定算子与白名单字段里挑一个组合）
            → ② Python 校验口径（字段越权、算子未知当场拒绝，一次都不算）
            → ③ Python 读 CSV 并按算子统计（空值与非数字逐格提示，绝不静默当 0）
            → ④ 模型只**解释**结果（提示词里禁止新增数字）
            → ⑤ Python 复核解释里出现的每一个带单位的数字都能在结果里找到出处

模型在这条链上的两次发言都不能碰数据本体：第一次只输出"选了什么"，第二次只把
结果翻译成人话。**代码一步都不生成、也不执行**——本模块没有 `eval` / `exec`，
也没有给模型任何可以拼字符串执行的地方（有一条测试盯着这件事）。

四条完成标准的落点：

1. **字段白名单有效**：`ALLOWED_FIELDS` 只放业务上允许分析的字段；`case_id` /
   `order_id` 这类标识列**表里存在但不在白名单**，表外的字段（如 `customer_phone`）
   更不可能通过。`validate_request()` 在统计之前拦下。
2. **空值和非数字有提示**：`scan_issues()` 逐格扫描，空值与非数字都记成
   `CellIssue`（第几行、哪个字段、原值是什么）；数值聚合**跳过**这些行并如实报数，
   分组计数把它们归到「（空值）」组（工单不能凭空少一单）。
3. **统计结果可复核**：结果里带来源文件、总行数、筛选后行数、实际参与行数、
   算式、抽样明细（前 5 行的工单编号与取值），任何人拿计算器都能对上。
4. **越权字段被拒绝**：见第 1 条；被拒时统计 0 次、也不再调用解释模型。

数据只在本目录内读取：`DATA_ROOT` 之外的路径、绝对路径、含 `..` 的路径、
非 `.csv` 后缀、超过大小上限的文件一律拒绝（"数据目录与权限"这一项由第3位成员负责）。

    uv run --extra langchain python -m app.data_lab
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import dataclass, field as dc_field
from itertools import combinations
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

# ---------------------------------------------------------------- ① 数据目录与文件

PROJECT_ID = "电商售后服务智能体项目组"
# 数据目录写死在本仓库内：模型给不出路径，只能给文件名，且文件名不许带目录分隔符。
DATA_ROOT = Path(__file__).resolve().parent.parent / "模拟数据" / "单元23实践数据" / PROJECT_ID
DATA_FILE = "电商售后工单.csv"
# 第4节要验证"空值和非数字有提示"，但任务卡给的原表是干净的；这份是从原表派生的
# 脏样本（7 处改动：数值空值 2、数值非数字 3、分类空值 2），改动清单写在单元文档里。
DIRTY_FILE = "电商售后工单_脏数据.csv"

MAX_FILE_BYTES = 512 * 1024
MAX_ROWS = 2000
SAMPLE_ROWS = 5
PREVIEW_CHARS = 60

# 问题守门的上下限（与单元20 的目标守门同一套规则，只是被守的对象换成了"问题"）
MIN_QUESTION_CHARS = 4
MAX_QUESTION_CHARS = 200

OFFLINE_MODEL_NAME = "scripted-data-model"

# ---------------------------------------------------------------- ② 数据字典与白名单

KIND_NUMBER = "number"
KIND_CATEGORY = "category"
KIND_DATE = "date"
KIND_ID = "id"


@dataclass(frozen=True)
class FieldSpec:
    """数据字典里的一行：字段叫什么、是什么类型、能不能拿来统计。"""

    name: str
    label: str
    kind: str
    analyzable: bool
    note: str


# 列顺序就是 CSV 的表头顺序，读取时会逐列核对（多一列、少一列都拒绝）。
FIELDS: list[FieldSpec] = [
    FieldSpec("case_id", "工单编号", KIND_ID, False,
              "流水标识，不是度量；拿它做统计没有业务含义（表里存在，但不在白名单）"),
    FieldSpec("submitted_at", "提交日期", KIND_DATE, True,
              "只用于派生「月份」参与分组；不按日期做加减（避免随手算时限）"),
    FieldSpec("order_id", "订单编号", KIND_ID, False,
              "关联到交易的标识，涉及客户信息，不参与统计（表里存在，但不在白名单）"),
    FieldSpec("product_category", "商品品类", KIND_CATEGORY, True, "数码配件/家居用品/服饰鞋包"),
    FieldSpec("issue_type", "问题类型", KIND_CATEGORY, True, "退货退款/质量问题/物流异常/尺码不合适/规格不符/退款进度"),
    FieldSpec("order_amount", "订单金额", KIND_NUMBER, True, "单位：元；用于求和、均值、最大、最小"),
    FieldSpec("refund_amount", "退款金额", KIND_NUMBER, True, "单位：元；0 表示尚未退款"),
    FieldSpec("status", "处理状态", KIND_CATEGORY, True, "已退款/处理中/人工复核"),
    FieldSpec("channel", "来源渠道", KIND_CATEGORY, True, "网页/APP/小程序"),
    FieldSpec("needs_handoff", "是否转人工", KIND_CATEGORY, True, "是/否"),
]

FIELD_BY_NAME = {spec.name: spec for spec in FIELDS}
COLUMNS = [spec.name for spec in FIELDS]

# 派生字段：按提交日期取月份，只用于分组，不在 CSV 里（避免模型把日期当数值算）。
MONTH_FIELD = "month"

# 白名单：只有这些字段能进统计口径。`case_id` / `order_id` 故意不在里面。
NUMERIC_FIELDS = frozenset(spec.name for spec in FIELDS if spec.kind == KIND_NUMBER)
CATEGORY_FIELDS = frozenset(spec.name for spec in FIELDS if spec.kind == KIND_CATEGORY)
GROUP_FIELDS = CATEGORY_FIELDS | {MONTH_FIELD}
ALLOWED_FIELDS = NUMERIC_FIELDS | GROUP_FIELDS
# 表里存在、但明确不许统计的字段（白名单的反面，演示"越权字段被拒绝"）。
FORBIDDEN_FIELDS = frozenset(spec.name for spec in FIELDS if not spec.analyzable)
# 筛选只能用分类字段（按金额精确筛选没有业务意义，也容易被拿来试值）。
FILTER_FIELDS = GROUP_FIELDS

EMPTY_GROUP = "（空值）"

# ---------------------------------------------------------------- ③ 固定算子

METRIC_COUNT = "count"
METRIC_SUM = "sum"
METRIC_MEAN = "mean"
METRIC_MAX = "max"
METRIC_MIN = "min"
METRIC_GROUP_COUNT = "group_count"
METRIC_GROUP_SUM = "group_sum"
METRIC_GROUP_MEAN = "group_mean"

# 模型能"选"的只有这八种；没有中位数、没有百分位、更没有自定义表达式——
# 想加新算子必须改这个文件（改代码是一个能被评审的动作，模型自己加不了）。
ALLOWED_METRICS = frozenset({
    METRIC_COUNT, METRIC_SUM, METRIC_MEAN, METRIC_MAX, METRIC_MIN,
    METRIC_GROUP_COUNT, METRIC_GROUP_SUM, METRIC_GROUP_MEAN,
})
# 需要数值字段的算子：field 是数值字段。
VALUE_METRICS = frozenset({METRIC_SUM, METRIC_MEAN, METRIC_MAX, METRIC_MIN})
# 需要分组字段的算子：field 是分组字段，group_sum/group_mean 还要一个数值字段。
GROUP_METRICS = frozenset({METRIC_GROUP_COUNT, METRIC_GROUP_SUM, METRIC_GROUP_MEAN})
AGGREGATE_METRICS = frozenset({METRIC_GROUP_SUM, METRIC_GROUP_MEAN})

METRIC_LABELS = {
    METRIC_COUNT: "计数（有多少单）",
    METRIC_SUM: "求和",
    METRIC_MEAN: "均值",
    METRIC_MAX: "最大值",
    METRIC_MIN: "最小值",
    METRIC_GROUP_COUNT: "分组计数（每组多少单）",
    METRIC_GROUP_SUM: "分组求和",
    METRIC_GROUP_MEAN: "分组均值",
}

UNIT_TICKET = "单"
UNIT_YUAN = "元"

MAX_FILTERS = 3
MAX_VALUE_CHARS = 40


class AnalysisRejected(ValueError):
    """口径不合法：字段越权、算子未知、筛选字段不对——一律在统计之前拒绝。"""


class DataProtocolError(RuntimeError):
    """模型没按约定回答：不是 JSON、缺字段、或返回了空内容。"""


# ---------------------------------------------------------------- ④ 守门与读取


def validate_question(question: Any) -> str:
    """问题守门：不合格就在问模型之前拦下（不浪费一次调用）。

    规则沿用单元20 的目标守门（非字符串 / 空 / 过短 / 超长），措辞改成"问题"，
    因为本单元被守的是运营随口问的那句话。
    """
    if not isinstance(question, str):
        raise AnalysisRejected(f"问题必须是字符串，实际是：{type(question).__name__}")
    clean = " ".join(question.split())
    if not clean:
        raise AnalysisRejected("问题不能为空")
    if len(clean) < MIN_QUESTION_CHARS:
        raise AnalysisRejected(f"问题至少 {MIN_QUESTION_CHARS} 个字，实际只有 {len(clean)} 个")
    if len(clean) > MAX_QUESTION_CHARS:
        raise AnalysisRejected(
            f"问题最多 {MAX_QUESTION_CHARS} 个字，实际 {len(clean)} 个；过长会撑爆口径窗口"
        )
    return clean


def resolve_source(source: Any) -> Path:
    """把"文件名"解析成数据目录里的真实路径；任何越界企图一律拒绝。

    模型最多只能给一个文件名：带目录分隔符、绝对路径、目录穿越（`..`）、
    非 `.csv` 后缀、文件不存在——全部拒绝，且不让 Python 打开任何别处的文件。
    """
    if not isinstance(source, str):
        raise AnalysisRejected(f"数据文件必须是文件名（字符串），实际是：{type(source).__name__}")
    name = source.strip()
    if not name:
        raise AnalysisRejected("数据文件不能为空")
    if "/" in name or "\\" in name:
        raise AnalysisRejected(f"数据文件只能是文件名，不能带路径：{name!r}")
    if name.startswith(".") or ".." in name:
        raise AnalysisRejected(f"数据文件名不允许以点开头或包含「..」：{name!r}")
    if Path(name).suffix.lower() != ".csv":
        raise AnalysisRejected(f"只允许读取 CSV 文件：{name!r}")
    path = (DATA_ROOT / name).resolve()
    if not path.is_relative_to(DATA_ROOT.resolve()):
        raise AnalysisRejected(f"数据文件必须在本组数据目录内：{name!r}")
    if not path.is_file():
        raise AnalysisRejected(f"本组数据目录里没有这个文件：{name!r}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise AnalysisRejected(f"文件超过 {MAX_FILE_BYTES} 字节上限：{size} 字节")
    return path


@dataclass(frozen=True)
class CellIssue:
    """一个坏单元格：第几行、哪个字段、原值是什么、为什么不参与计算。"""

    row_no: int
    case_id: str
    field: str
    raw: str
    reason: str

    def render(self) -> str:
        shown = self.raw if len(self.raw) <= 12 else self.raw[:12] + "…"
        return (
            f"第 {self.row_no} 行（{self.case_id}）{self.field}="
            f"{shown or '（空）'}：{self.reason}"
        )


@dataclass
class Table:
    """读进来的一张表：原始行 + 派生月份 + 逐格扫出来的问题。"""

    path: Path
    columns: list[str]
    rows: list[dict[str, str]]
    issues: list[CellIssue] = dc_field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def row_count(self) -> int:
        return len(self.rows)


def load_table(path: Path) -> Table:
    """读 CSV：先核对表头与数据字典一致，再逐格扫空值与非数字。

    表头不一致就拒绝——列变了，白名单的意义也就没了（多出来的列可能是别的项目的字段）。
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise AnalysisRejected(f"文件是空的：{path.name}") from None
        columns = [cell.strip() for cell in header]
        if columns != COLUMNS:
            missing = [name for name in COLUMNS if name not in columns]
            extra = [name for name in columns if name not in COLUMNS]
            detail = []
            if missing:
                detail.append("缺少列：" + "、".join(missing))
            if extra:
                detail.append("多出列：" + "、".join(extra))
            detail.append("列顺序与数据字典不一致" if not detail else "")
            raise AnalysisRejected(f"{path.name} 的表头与数据字典不一致（{'；'.join(d for d in detail if d)}）")
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            if not any(cell.strip() for cell in raw_row):
                continue
            if len(rows) >= MAX_ROWS:
                raise AnalysisRejected(f"数据超过 {MAX_ROWS} 行上限，先确认文件是不是拿错了")
            row = dict(zip(columns, (cell.strip() for cell in raw_row)))
            row[MONTH_FIELD] = row.get("submitted_at", "")[:7]
            rows.append(row)
    return Table(path=path, columns=COLUMNS, rows=rows, issues=scan_issues(rows))


def scan_issues(rows: list[dict[str, str]]) -> list[CellIssue]:
    """逐格扫描：空值与非数字全部记下来（提示用），一个都不静默吞掉。

    为什么扫全表而不是只扫用到的字段：数据质量提示是给运营看的，
    "订单金额有 2 行是空的"和"退款金额有 3 行不是数字"都该在第一次打开时就说清楚。
    """
    issues: list[CellIssue] = []
    for index, row in enumerate(rows, start=2):  # 第 1 行是表头
        case_id = row.get("case_id", "")
        for name in NUMERIC_FIELDS:
            raw = row.get(name, "")
            if not raw.strip():
                issues.append(CellIssue(index, case_id, name, raw, "空值，不参与数值统计"))
            elif _to_number(raw) is None:
                issues.append(CellIssue(index, case_id, name, raw, "非数字，不参与数值统计"))
        for name in CATEGORY_FIELDS:
            raw = row.get(name, "")
            if not raw.strip():
                issues.append(CellIssue(index, case_id, name, raw, "空值，分组时归入「（空值）」"))
    return issues


def _to_number(raw: str) -> float | None:
    """只认普通十进制数字；空、中文、'N/A'、科学计数法之外的一律不算数字。"""
    text = (raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return value


# ---------------------------------------------------------------- ⑤ 口径校验


@dataclass(frozen=True)
class AnalysisRequest:
    """一次统计的口径：算子 + 字段（+ 可选筛选）。模型只能产出这个，不能产出代码。"""

    metric: str
    field: str = ""
    value_field: str = ""
    filters: dict[str, str] = dc_field(default_factory=dict)

    def render(self) -> str:
        text = METRIC_LABELS.get(self.metric, self.metric)
        if self.field:
            text += f"（{self.field}）"
        if self.value_field:
            text += f" 按 {self.value_field}"
        if self.filters:
            text += "；筛选：" + "、".join(f"{key}={value}" for key, value in self.filters.items())
        return text


def validate_request(payload: Any) -> AnalysisRequest:
    """把模型给的 JSON 变成可执行的口径；任何越权都在这里当场拒绝。

    这是"模型与数据之间"的那道门：模型只能从预先开放的算子与字段里挑，
    给不出 SQL、给不出文件路径、更给不出一段要执行的代码。
    """
    if not isinstance(payload, dict):
        raise AnalysisRejected(f"统计口径必须是 JSON 对象，实际是：{type(payload).__name__}")
    metric = payload.get("metric")
    if not isinstance(metric, str) or metric.strip() not in ALLOWED_METRICS:
        raise AnalysisRejected(
            f"算子未授权：{metric!r}；只有 {'、'.join(sorted(ALLOWED_METRICS))} 可以选"
        )
    metric = metric.strip()
    field = _clean_field(payload.get("field"), metric)
    value_field = _clean_field(payload.get("value_field"), metric, optional=True)

    if metric == METRIC_COUNT:
        if field:
            raise AnalysisRejected("count 不需要字段（统计的是筛选后的行数）")
    elif metric in VALUE_METRICS:
        if not field:
            raise AnalysisRejected(f"{metric} 必须指定数值字段")
        if field not in NUMERIC_FIELDS:
            raise AnalysisRejected(
                f"{metric} 只能用于数值字段（{'、'.join(sorted(NUMERIC_FIELDS))}），实际给了：{field!r}"
            )
    else:  # 分组类
        if not field:
            raise AnalysisRejected(f"{metric} 必须指定分组字段")
        if field not in GROUP_FIELDS:
            # 走统一的白名单校验：标识列与表外字段的措辞要能看出"为什么不行"。
            _check_field_allowed(field, GROUP_FIELDS, f"{metric} 的分组字段")
        if metric in AGGREGATE_METRICS:
            if not value_field:
                raise AnalysisRejected(f"{metric} 还需要一个数值字段（value_field）")
            if value_field not in NUMERIC_FIELDS:
                raise AnalysisRejected(
                    f"{metric} 的数值字段只能是（{'、'.join(sorted(NUMERIC_FIELDS))}），"
                    f"实际给了：{value_field!r}"
                )
        elif value_field:
            raise AnalysisRejected(f"{metric} 不接受 value_field")

    filters_raw = payload.get("filters")
    if filters_raw in (None, {}):
        filters: dict[str, str] = {}
    elif isinstance(filters_raw, dict):
        if len(filters_raw) > MAX_FILTERS:
            raise AnalysisRejected(f"筛选条件最多 {MAX_FILTERS} 个，实际给了 {len(filters_raw)} 个")
        filters = {}
        for key, value in filters_raw.items():
            if not isinstance(key, str):
                raise AnalysisRejected(f"筛选字段名必须是字符串，实际是：{type(key).__name__}")
            _check_field_allowed(key, FILTER_FIELDS, "筛选字段")
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise AnalysisRejected(
                    f"筛选值类型不正确：{key}={type(value).__name__}；只接受字符串或数字"
                )
            text = str(value).strip()
            if not text or len(text) > MAX_VALUE_CHARS:
                raise AnalysisRejected(f"筛选值不合法：{key}={value!r}")
            filters[key] = text
    else:
        raise AnalysisRejected(f"filters 必须是对象，实际是：{type(filters_raw).__name__}")

    return AnalysisRequest(metric=metric, field=field, value_field=value_field, filters=filters)


def _clean_field(value: Any, metric: str, optional: bool = False) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        if optional:
            return ""
        return ""
    if not isinstance(value, str):
        raise AnalysisRejected(f"{metric} 的字段必须是字段名（字符串），实际是：{type(value).__name__}")
    return value.strip()


def _check_field_allowed(name: str, allowed: frozenset[str], label: str) -> None:
    """字段白名单：表外的字段、表内但不许统计的标识列，都在这里被拦下。"""
    if name in FORBIDDEN_FIELDS:
        raise AnalysisRejected(
            f"{label} {name!r} 是标识列，不在可分析白名单内：统计编号没有业务含义，也可能泄露客户信息"
        )
    if name not in allowed:
        if name in ALLOWED_FIELDS:
            raise AnalysisRejected(f"{label} {name!r} 不能这样用")
        raise AnalysisRejected(
            f"{label} {name!r} 不在白名单内：本组数据字典里没有这个可分析字段"
        )


# ---------------------------------------------------------------- ⑥ 统计


@dataclass
class GroupStat:
    name: str
    count: int
    value: float | None = None
    skipped: int = 0


@dataclass
class AnalysisResult:
    """一次分析的全部产出；渲染与验收只读这份数据，不重跑。"""

    question: str
    source: str
    request: AnalysisRequest | None = None
    rows_total: int = 0
    rows_matched: int = 0
    rows_used: int = 0
    rows_skipped: int = 0
    value: float | None = None
    unit: str = UNIT_TICKET
    groups: list[GroupStat] = dc_field(default_factory=list)
    formula: str = ""
    sample: list[tuple[str, str]] = dc_field(default_factory=list)
    issues: list[CellIssue] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)
    explanation: str = ""
    number_notes: list[str] = dc_field(default_factory=list)
    model_calls: int = 0
    choose_calls: int = 0
    explain_calls: int = 0
    elapsed_ms: float = 0.0
    model_name: str = ""
    rejected: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.rejected and self.error == ""

    @property
    def has_data(self) -> bool:
        return self.rows_matched > 0 and (self.value is not None or bool(self.groups))


def _fmt(number: float) -> str:
    """整数不带小数点，其余保留两位——金额与占比都能直接抄去核对。"""
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.2f}"


def _match(row: dict[str, str], filters: dict[str, str]) -> bool:
    return all(str(row.get(key, "")).strip() == value for key, value in filters.items())


def compute(table: Table, request: AnalysisRequest) -> AnalysisResult:
    """按口径算一遍：筛选 → 跳过坏值 → 聚合 → 留下可复核的算式与抽样。"""
    result = AnalysisResult(question="", source=table.name, request=request)
    result.rows_total = table.row_count
    matched = [row for row in table.rows if _match(row, request.filters)]
    result.rows_matched = len(matched)
    result.issues = list(table.issues)
    result.unit = UNIT_YUAN if (request.field in NUMERIC_FIELDS
                                or request.value_field in NUMERIC_FIELDS) else UNIT_TICKET

    if not matched:
        result.warnings.append(
            "没有符合条件的行：筛选条件在表里没有匹配到数据，结果为空（不是出错）"
        )
        result.formula = f"共 {result.rows_total} 行，筛选后 0 行：无可统计的数据"
        return result

    if request.metric in GROUP_METRICS:
        _compute_groups(result, matched, request)
        return result

    if request.metric == METRIC_COUNT:
        result.value = float(len(matched))
        result.rows_used = len(matched)
        result.formula = f"共 {result.rows_total} 行，筛选后 {len(matched)} 行：直接计数 = {len(matched)} {result.unit}"
        # 抽样带上筛选字段的取值：复核时能看清"这一单是因为什么被算进来"。
        result.sample = [
            (
                row.get("case_id", ""),
                "、".join(f"{key}={row.get(key, '')}" for key in request.filters) or "计入",
            )
            for row in matched[:SAMPLE_ROWS]
        ]
        return result

    skipped = 0
    values: list[tuple[str, float]] = []
    for row in matched:
        number = _to_number(row.get(request.field, ""))
        if number is None:
            skipped += 1
            continue
        values.append((row.get("case_id", ""), number))
    result.rows_used = len(values)
    result.rows_skipped = skipped
    if skipped:
        result.warnings.append(
            f"{skipped} 行的{request.field}为空值或非数字，未计入聚合（明细见数据质量提示）"
        )
    if not values:
        result.warnings.append(
            f"{request.field} 在筛选后的 {len(matched)} 行里没有可用数值（全部为空值或非数字）"
        )
        result.formula = f"筛选后 {len(matched)} 行，其中可用数值 0 行：无法计算"
        return result

    numbers = [value for _case, value in values]
    total = sum(numbers)
    if request.metric == METRIC_SUM:
        result.value = total
        result.formula = (
            f"{_fmt(total)} = {len(values)} 行有效数值相加"
            + (f"（跳过 {skipped} 行：空值或非数字）" if skipped else "（无跳过）")
        )
    elif request.metric == METRIC_MEAN:
        result.value = total / len(values)
        result.formula = (
            f"均值 = 总和 {_fmt(total)} ÷ 有效行 {len(values)} = {_fmt(result.value)} {result.unit}"
            + (f"（跳过 {skipped} 行）" if skipped else "")
        )
    elif request.metric in (METRIC_MAX, METRIC_MIN):
        picked = max(values, key=lambda item: item[1]) if request.metric == METRIC_MAX else min(
            values, key=lambda item: item[1]
        )
        result.value = picked[1]
        label = "最大" if request.metric == METRIC_MAX else "最小"
        result.formula = (
            f"{label}值 = {_fmt(picked[1])}（工单 {picked[0]}），取自 {len(values)} 行有效数值"
            + (f"（跳过 {skipped} 行）" if skipped else "")
        )
        result.sample = [(picked[0], f"{_fmt(picked[1])} {result.unit}")]
        return result
    result.sample = [(case, f"{_fmt(value)} {result.unit}") for case, value in values[:SAMPLE_ROWS]]
    return result


def _compute_groups(result: AnalysisResult, matched: list[dict[str, str]], request: AnalysisRequest) -> None:
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in matched:
        key = str(row.get(request.field, "")).strip() or EMPTY_GROUP
        buckets.setdefault(key, []).append(row)
    total_rows = len(matched)
    stats: list[GroupStat] = []
    skipped_all = 0
    for name in sorted(buckets):
        rows = buckets[name]
        skipped = 0
        numbers: list[float] = []
        if request.metric in AGGREGATE_METRICS:
            for row in rows:
                number = _to_number(row.get(request.value_field, ""))
                if number is None:
                    skipped += 1
                else:
                    numbers.append(number)
        skipped_all += skipped
        value: float | None = None
        if request.metric == METRIC_GROUP_SUM:
            value = sum(numbers) if numbers else None
        elif request.metric == METRIC_GROUP_MEAN:
            value = (sum(numbers) / len(numbers)) if numbers else None
        stats.append(GroupStat(name=name, count=len(rows), value=value, skipped=skipped))
    result.groups = stats
    result.rows_used = total_rows
    result.rows_skipped = skipped_all
    result.value = float(total_rows)
    if request.metric == METRIC_GROUP_COUNT:
        result.formula = (
            f"各组单数相加 = {total_rows} 行；占比 = 该组单数 ÷ {total_rows}"
        )
    else:
        label = "总和" if request.metric == METRIC_GROUP_SUM else "均值"
        result.formula = (
            f"每组 {label} = 该组内{request.value_field}的有效数值{'相加' if request.metric == METRIC_GROUP_SUM else '相加 ÷ 有效行数'}"
            + (f"；共跳过 {skipped_all} 行（空值或非数字）" if skipped_all else "")
        )
    # 抽样：按分组顺序各取一行，复核时能看到"这一单确实落在这组"。
    sample: list[tuple[str, str]] = []
    for stat in stats:
        row = buckets[stat.name][0]
        # 分组计数时抽样只写"这一单落在哪组"；分组聚合时才带上这一行的取值。
        if request.value_field:
            shown = f"{stat.name}={str(row.get(request.value_field, ''))}"
        else:
            shown = stat.name
        sample.append((row.get("case_id", ""), shown))
        if len(sample) >= SAMPLE_ROWS:
            break
    result.sample = sample
    if skipped_all:
        result.warnings.append(f"{skipped_all} 行的{request.value_field}为空值或非数字，未计入分组聚合")
    if EMPTY_GROUP in buckets:
        result.warnings.append(f"有 {len(buckets[EMPTY_GROUP])} 行的{request.field}为空，已归入「{EMPTY_GROUP}」而不是丢弃")


# ---------------------------------------------------------------- ⑦ 模型：选口径与解释

TAG_CHOOSE = "【步骤：选统计口径】"
TAG_EXPLAIN = "【步骤：解释结果】"

CHOOSE_SYSTEM = (
    TAG_CHOOSE
    + "\n你是电商售后团队的数据分析助手。运营主管会用业务语言提一个问题，"
    "你要做的是从下面给出的**固定算子**和**字段白名单**里挑一个组合。\n"
    "严格要求：\n"
    "1. 只输出一个 JSON 对象，不要解释、不要代码块、不要写任何代码或 SQL；\n"
    "2. 字段名必须是白名单里出现过的英文列名，不许出现工单编号、订单编号这类标识列；\n"
    "3. 你拿不到数据，也不要猜任何数字；\n"
    "4. 拿不准就选最直接的那个算子，不要为了显得全面而多加筛选。"
)

EXPLAIN_SYSTEM = (
    TAG_EXPLAIN
    + "\n你是电商售后团队的数据分析助手，负责把**已经算好的统计结果**翻译成业务语言。\n"
    "严格要求：\n"
    "1. 只能使用结果里出现的数字，不许新增、不许估算、不许四舍五入出新数字；\n"
    "2. 提到金额要带「元」，提到占比要带「%」，提到数量要带「单」；\n"
    "3. 结果里若有跳过行、空值或非数字，必须如实说明（这是运营最该知道的）；\n"
    "4. 200 字以内，写给运营主管看，不要复述 JSON。"
)

REQUEST_EXAMPLE = '{"metric": "group_count", "field": "issue_type", "value_field": "", "filters": {}}'


def render_dictionary() -> str:
    """第5位成员要交的数据字典：字段 / 中文名 / 类型 / 能不能统计 / 说明。"""
    lines = ["数据字典（电商售后工单.csv）", "字段\t中文名\t类型\t可分析\t说明"]
    for spec in FIELDS:
        lines.append(
            f"{spec.name}\t{spec.label}\t{spec.kind}\t{'是' if spec.analyzable else '否'}\t{spec.note}"
        )
    lines.append(f"{MONTH_FIELD}\t提交月份\t派生\t是\t由 submitted_at 取前 7 位得到，只用于分组")
    lines.append("")
    lines.append(f"可分析字段（白名单）：{'、'.join(sorted(ALLOWED_FIELDS))}")
    lines.append(f"表里存在但**不在白名单**的字段：{'、'.join(sorted(FORBIDDEN_FIELDS))}")
    return "\n".join(lines)


def render_whitelist() -> str:
    """白名单与算子清单：模型看得见的清单 = 程序肯执行的清单。"""
    lines = ["字段白名单与固定算子（模型只能从这里选）", "算子\t含义\t需要的字段"]
    lines.append(f"{METRIC_COUNT}\t{METRIC_LABELS[METRIC_COUNT]}\t无（只统计筛选后的行数）")
    for metric in (METRIC_SUM, METRIC_MEAN, METRIC_MAX, METRIC_MIN):
        lines.append(f"{metric}\t{METRIC_LABELS[metric]}\t数值字段：{'、'.join(sorted(NUMERIC_FIELDS))}")
    lines.append(f"{METRIC_GROUP_COUNT}\t{METRIC_LABELS[METRIC_GROUP_COUNT]}\t分组字段：{'、'.join(sorted(GROUP_FIELDS))}")
    for metric in (METRIC_GROUP_SUM, METRIC_GROUP_MEAN):
        lines.append(
            f"{metric}\t{METRIC_LABELS[metric]}\t分组字段 + 数值字段（{'、'.join(sorted(NUMERIC_FIELDS))}）"
        )
    lines.append("")
    lines.append(f"筛选字段只能是：{'、'.join(sorted(FILTER_FIELDS))}（最多 {MAX_FILTERS} 个）")
    lines.append("输出示例：" + REQUEST_EXAMPLE)
    lines.append("模型给不出路径、给不出 SQL、也给不出代码：它只能从上面这张表里挑。")
    return "\n".join(lines)


def _choose_prompt(question: str) -> str:
    return "\n".join([
        "下面是本组数据字典、字段白名单与可用算子：",
        render_dictionary(),
        "",
        render_whitelist(),
        "",
        f"运营主管的问题：{question}",
        f"请只输出一个 JSON 对象，形如：{REQUEST_EXAMPLE}",
    ])


def _explain_prompt(question: str, result: AnalysisResult) -> str:
    payload = {
        "问题": question,
        "数据来源": result.source,
        "统计口径": result.request.render() if result.request else "（无）",
        "总行数": result.rows_total,
        "筛选后行数": result.rows_matched,
        "实际参与行数": result.rows_used,
        "跳过行数": result.rows_skipped,
        "结果值": None if result.value is None else round(result.value, 2),
        "单位": result.unit,
        "分组": [
            {
                "组": stat.name,
                "单数": stat.count,
                "占比": f"{stat.count / result.rows_matched * 100:.1f}%" if result.rows_matched else "0%",
                "聚合值": None if stat.value is None else round(stat.value, 2),
            }
            for stat in result.groups
        ],
        "算式": result.formula,
        "数据质量提示": [issue.render() for issue in result.issues[:8]],
        "提醒": result.warnings,
    }
    return "\n".join([
        "下面是已经算好的统计结果（数字都是 Python 算出来的，不要改动）：",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "",
        f"请用业务语言回答运营主管的问题：{question}",
        "只能使用上面出现的数字，不许新增数字，不许写代码。",
    ])


def choose_analysis(model: Any, question: str) -> AnalysisRequest:
    """第一次模型调用：只让它选口径（不许写代码、不许要数据）。"""
    reply = _ask(model, CHOOSE_SYSTEM, _choose_prompt(question))
    payload = _parse_json(reply)
    return validate_request(payload)


def _context_target(model: Any) -> Any | None:
    """找到真正需要数字上下文的模型。

    模型可能被包了一层（比如证据脚本里的记录器会把每次提示词留下来），
    这时要透过 `inner` 找到里面的剧本模型，否则 `{rows_total}` 之类填不进去。
    """
    seen, current = 0, model
    while current is not None and seen < 4:
        if hasattr(current, "context"):
            return current
        current = getattr(current, "inner", None)
        seen += 1
    return None


def explain_result(model: Any, question: str, result: AnalysisResult) -> str:
    """第二次模型调用：只让它把结果翻译成业务语言。"""
    target = _context_target(model)
    if target is not None:
        object.__setattr__(target, "context", _number_context(result))
    reply = _ask(model, EXPLAIN_SYSTEM, _explain_prompt(question, result))
    text = " ".join(reply.split())
    if not text:
        raise DataProtocolError("解释为空：模型没有给出结论")
    return text


def _ask(model: Any, system: str, human: str) -> str:
    reply = model.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    content = getattr(reply, "content", reply)
    if not isinstance(content, str):
        content = str(content)
    text = content.strip()
    if not text:
        raise DataProtocolError("模型返回了空内容")
    return text


def _parse_json(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataProtocolError(f"模型输出的不是 JSON：{text[:PREVIEW_CHARS]}") from exc
    if not isinstance(payload, dict):
        raise DataProtocolError(f"统计口径必须是 JSON 对象，实际是：{type(payload).__name__}")
    return payload


# ---------------------------------------------------------------- ⑧ 数字核验


def allowed_numbers(result: AnalysisResult) -> set[float]:
    """结果里出现过的所有数字：解释里出现的带单位数字必须能在这里找到出处。"""
    numbers: set[float] = {
        float(result.rows_total), float(result.rows_matched),
        float(result.rows_used), float(result.rows_skipped),
    }
    if result.value is not None:
        numbers.add(round(result.value, 2))
        numbers.add(round(result.value, 1))
        numbers.add(round(result.value))
    for stat in result.groups:
        numbers.add(float(stat.count))
        if stat.value is not None:
            numbers.add(round(stat.value, 2))
            numbers.add(round(stat.value, 1))
            numbers.add(round(stat.value))
        if result.rows_matched:
            ratio = stat.count / result.rows_matched * 100
            numbers.add(round(ratio, 1))
            numbers.add(round(ratio))
    for _case, shown in result.sample:
        for found in re.findall(r"\d+(?:\.\d+)?", shown):
            numbers.add(round(float(found), 2))
    # 业务解释里常说"前三类合计"：若干组相加也能由结果算出来，不算新数字。
    if len(result.groups) <= 8:
        for size in range(2, len(result.groups) + 1):
            for combo in combinations(result.groups, size):
                total = sum(stat.count for stat in combo)
                numbers.add(float(total))
                if result.rows_total:
                    ratio = total / result.rows_total * 100
                    numbers.add(round(ratio, 1))
                    numbers.add(round(ratio))
                values = [stat.value for stat in combo if stat.value is not None]
                if len(values) == size:
                    summed = sum(values)
                    numbers.add(round(summed, 2))
                    numbers.add(round(summed, 1))
                    numbers.add(round(summed))
    # 由结果直接算得出的占比也算"有出处"：占比是复核的常用口径，不是新事实。
    if result.rows_total:
        for part in (result.rows_matched, result.rows_used, result.rows_skipped):
            ratio = part / result.rows_total * 100
            numbers.add(round(ratio, 1))
            numbers.add(round(ratio))
        if result.value is not None:
            ratio = result.value / result.rows_total * 100
            numbers.add(round(ratio, 1))
            numbers.add(round(ratio))
    return numbers


NUMBER_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:元|%|单|条|行)")


def check_explanation(text: str, result: AnalysisResult) -> list[str]:
    """解释里的每个带单位数字都要能在结果里找到出处；找不到就点出来，不悄悄放过。

    为什么只查"带单位"的数字：像"2026年7月"这种日期数字不是统计结论，
    硬查会把正常的业务表述判成编造。`元 / % / 单 / 条 / 行` 才是结论的样子。
    """
    allowed = allowed_numbers(result)
    notes: list[str] = []
    seen: set[float] = set()
    for raw in NUMBER_PATTERN.findall(text):
        value = round(float(raw), 2)
        if value in seen:
            continue
        seen.add(value)
        if not any(abs(value - item) <= 0.051 for item in allowed):
            notes.append(f"{raw} 不在统计结果里（结果中的数字：{sorted(allowed)[:8]}…）")
    return notes


def _number_context(result: AnalysisResult) -> dict[str, Any]:
    """离线剧本解释里可以引用的数字：模板里的 {rows_total} 之类由它替换。"""
    top_name, top_value, top_ratio = "", 0, "0%"
    if result.groups:
        top = max(result.groups, key=lambda stat: stat.count)
        top_name, top_value = top.name, top.count
        if result.rows_matched:
            top_ratio = f"{top.count / result.rows_matched * 100:.0f}%"
    elif result.value is not None and result.rows_total:
        # 没有分组时（比如带筛选的计数），占比按"结果 ÷ 总行数"算。
        top_value = int(result.value) if float(result.value).is_integer() else result.value
        top_ratio = f"{result.value / result.rows_total * 100:.0f}%"
    return {
        "rows_total": result.rows_total,
        "rows_matched": result.rows_matched,
        "rows_used": result.rows_used,
        "rows_skipped": result.rows_skipped,
        "value": "—" if result.value is None else _fmt(result.value),
        "unit": result.unit,
        "top_name": top_name,
        "top_value": top_value,
        "top_ratio": top_ratio,
        "source": result.source,
    }


# ---------------------------------------------------------------- ⑨ 一次完整分析


def run_analysis(
    model: Any,
    question: Any,
    source: str = DATA_FILE,
    request_payload: Any | None = None,
    skip_explain: bool = False,
) -> AnalysisResult:
    """跑一次分析：守门 → 选口径 → 校验 → 统计 → 解释 → 数字核验。"""
    started = time.perf_counter()
    clean_question = validate_question(question)
    path = resolve_source(source)
    table = load_table(path)

    result = AnalysisResult(question=clean_question, source=table.name)
    result.model_name = getattr(model, "model_name", "") or type(model).__name__
    try:
        if request_payload is not None:
            # 口径由剧本/调用方直接给出（故障注入场景）：这一步根本不调模型。
            request = validate_request(request_payload)
            result.choose_calls = 0
        else:
            request = choose_analysis(model, clean_question)
            result.choose_calls = 1
        result.request = request
        computed = compute(table, request)
        computed.question = clean_question
        computed.model_name = result.model_name
        computed.choose_calls = result.choose_calls
        result = computed
        if not skip_explain:
            result.explanation = explain_result(model, clean_question, result)
            result.number_notes = check_explanation(result.explanation, result)
            result.explain_calls = 1
    except AnalysisRejected as error:
        result.rejected = True
        result.error = str(error)
    except DataProtocolError as error:
        result.error = str(error)
    finally:
        # 调用次数按"实际发了几次"数，不依赖模型自己记账（真实模型没有 calls 字段）。
        result.model_calls = result.choose_calls + result.explain_calls
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    return result


# ---------------------------------------------------------------- ⑩ 离线剧本模型


@dataclass
class DataScript:
    """离线剧本：口径队列 + 解释模板队列（模板里的 {rows_total} 等会被真实数字替换）。"""

    requests: list[Any] = dc_field(default_factory=list)
    explanations: list[str] = dc_field(default_factory=list)


class ScriptedDataModel(BaseChatModel):
    """离线剧本模型：占住 ChatDeepSeek 那一格，没密钥也能把分析跑完。

    按 system 消息里的步骤标记分派：选口径取 `requests` 队列，解释取 `explanations`
    队列（按调用顺序 pop）。认不出步骤就抛错——宁可停下来，也不要把解释当口径解析。
    """

    model_name: str = OFFLINE_MODEL_NAME
    requests: list[Any] = []
    explanations: list[str] = []
    context: dict[str, Any] = {}
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return OFFLINE_MODEL_NAME

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        object.__setattr__(self, "calls", int(self.calls) + 1)
        tag = _step_tag(messages)
        if tag == TAG_CHOOSE:
            content = self._take_request()
        elif tag == TAG_EXPLAIN:
            content = self._take_explanation()
        else:
            raise DataProtocolError("提示词里没有步骤标记，剧本模型不知道该干什么")
        message = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _take_request(self) -> str:
        if not self.requests:
            payload: dict[str, Any] = {"metric": METRIC_COUNT, "field": "", "value_field": "", "filters": {}}
        else:
            payload = self.requests.pop(0)
        if isinstance(payload, str):
            return payload
        return json.dumps(payload, ensure_ascii=False)

    def _take_explanation(self) -> str:
        text = self.explanations.pop(0) if self.explanations else "（离线剧本）暂无解释。"
        if "{" in text and self.context:
            try:
                return text.format(**self.context)
            except (KeyError, IndexError, ValueError):
                return text
        return text


def _step_tag(messages: list[BaseMessage]) -> str:
    """步骤标记只认 system 消息：human 里可能出现同样的词（见单元22 那个坑）。"""
    for message in messages:
        if isinstance(message, SystemMessage):
            text = str(message.content)
            for tag in (TAG_CHOOSE, TAG_EXPLAIN):
                if tag in text:
                    return tag
    raise DataProtocolError("提示词里没有步骤标记（system 消息缺失）")


def build_data_model(script: DataScript | None = None, provider: str | None = None):
    """按环境变量挑模型：`deepseek` 走真实 ChatDeepSeek，`mock` 走离线剧本。"""
    chosen = (provider or os.getenv("LLM_PROVIDER", "mock")).strip().lower()
    if chosen == "deepseek":
        from app.plan_solver import build_plan_model

        return build_plan_model(provider="deepseek")
    if chosen != "mock":
        raise ValueError("LLM_PROVIDER 只能是 mock 或 deepseek")
    script = script or DataScript()
    return ScriptedDataModel(
        requests=list(script.requests),
        explanations=list(script.explanations),
    )


# ---------------------------------------------------------------- ⑪ 渲染


def render_flow() -> str:
    """第1节要交的流程图：问题 → 选口径 → 校验 → 统计 → 解释 → 数字核验。"""
    return "\n".join([
        "数据分析 Agent 流程（本组电商售后工单）",
        "",
        "  运营问题（业务语言）",
        "      │",
        "      ▼  ① 模型只**选口径**：从 8 个固定算子 + 字段白名单里挑一个组合",
        "      │     （输出 JSON，不含任何代码）",
        "      ▼  ② Python 校验：字段越权 / 算子未知 / 筛选不合法 → 当场拒绝（统计 0 次）",
        "      ▼  ③ Python 读 CSV 并按算子统计：空值与非数字逐格提示，绝不静默当 0",
        "      ▼  ④ 模型只**解释**：把结果翻译成业务语言，不许新增数字",
        "      ▼  ⑤ Python 复核：解释里每个带单位的数字都要能在结果里找到出处",
        "",
        "全程没有 eval / exec：模型给的是「选择」，不是「代码」。",
    ])


def render_result(result: AnalysisResult) -> str:
    """一次分析的完整输出：口径、结果、算式、抽样、数据质量、解释与核验。"""
    lines: list[str] = []
    lines.append(f"问题：{result.question}")
    lines.append(f"数据来源：{result.source}（数据目录内读取，模型不碰路径）")
    if result.rejected:
        lines.append(f"统计口径：已拒绝 —— {result.error}")
        lines.append(f"统计：0 次（拒绝发生在统计之前）；模型调用 {result.model_calls} 次")
        return "\n".join(lines)
    if result.error:
        lines.append(f"未产出结果：{result.error}")
        return "\n".join(lines)
    if result.request is None:
        lines.append("未产出结果：没有统计口径")
        return "\n".join(lines)

    lines.append(f"统计口径：{result.request.render()}")
    lines.append(
        f"行数：共 {result.rows_total} 行，筛选后 {result.rows_matched} 行，"
        f"实际参与 {result.rows_used} 行，跳过 {result.rows_skipped} 行"
    )

    if result.groups:
        lines.append(f"结果（{len(result.groups)} 组）：")
        lines.append("组\t单数\t占比\t聚合值")
        for stat in result.groups:
            ratio = f"{stat.count / result.rows_matched * 100:.1f}%" if result.rows_matched else "—"
            value = "—" if stat.value is None else f"{_fmt(stat.value)} {result.unit}"
            lines.append(f"{stat.name}\t{stat.count}\t{ratio}\t{value}")
    elif result.value is not None:
        lines.append(f"结果：{_fmt(result.value)} {result.unit}")
    else:
        lines.append("结果：（无——筛选后没有可用数据）")

    if result.formula:
        lines.append(f"算式（可手工复核）：{result.formula}")
    if result.sample:
        lines.append("抽样复核（前 5 行）：" + "；".join(
            f"{case}={shown}" for case, shown in result.sample
        ))

    if result.issues:
        lines.append(f"数据质量提示（{len(result.issues)} 处，逐格扫描得到）：")
        for issue in result.issues[:8]:
            lines.append(f"  · {issue.render()}")
        if len(result.issues) > 8:
            lines.append(f"  · …… 其余 {len(result.issues) - 8} 处略")
    else:
        lines.append("数据质量：无空值、无非数字，全部行参与统计")

    for warning in result.warnings:
        lines.append(f"提醒：{warning}")

    if result.explanation:
        lines.append(f"业务解释：{result.explanation}")
        if result.number_notes:
            lines.append("数字核验：" + "；".join(result.number_notes))
        else:
            lines.append("数字核验：解释里每个带单位的数字都能在统计结果里找到出处")
    lines.append(
        f"模型调用：{result.model_calls} 次（选口径 {result.choose_calls} + 解释 {result.explain_calls}"
        + ("；口径由剧本直接给出，未调模型" if result.choose_calls == 0 else "")
        + f"）；耗时 {result.elapsed_ms} ms"
    )
    return "\n".join(lines)


CONCEPTS = [
    ("字段白名单", "只有业务上允许分析的字段能进统计口径", f"ALLOWED_FIELDS，共 {len(ALLOWED_FIELDS)} 个"),
    ("固定算子", "模型只能从 8 个算子里挑，挑不出代码", "ALLOWED_METRICS，没有 eval/exec"),
    ("空值提示", "空单元格逐格记录，不静默当 0", "scan_issues() → CellIssue"),
    ("非数字提示", "「待补」「未知」这类值不参与聚合", "同上，数值聚合跳过并报数"),
    ("可复核", "结果带行数、算式与抽样明细", "formula + sample，可手工对账"),
    ("越权拒绝", "标识列与表外字段一律不许统计", "validate_request() 在统计之前拦下"),
]


def render_concepts() -> str:
    lines = ["概念记录（第1节）", "概念\t含义\t在本项目里落在哪"]
    lines.extend(f"{name}\t{meaning}\t{here}" for name, meaning, here in CONCEPTS)
    return "\n".join(lines)


__all__ = [
    "AGGREGATE_METRICS",
    "ALLOWED_FIELDS",
    "ALLOWED_METRICS",
    "AnalysisRejected",
    "AnalysisRequest",
    "AnalysisResult",
    "CellIssue",
    "DataProtocolError",
    "DataScript",
    "DATA_FILE",
    "DATA_ROOT",
    "DIRTY_FILE",
    "EMPTY_GROUP",
    "FIELDS",
    "FORBIDDEN_FIELDS",
    "GROUP_FIELDS",
    "METRIC_COUNT",
    "METRIC_GROUP_COUNT",
    "METRIC_GROUP_MEAN",
    "METRIC_GROUP_SUM",
    "METRIC_MAX",
    "METRIC_MEAN",
    "METRIC_MIN",
    "METRIC_SUM",
    "MONTH_FIELD",
    "NUMERIC_FIELDS",
    "ScriptedDataModel",
    "Table",
    "allowed_numbers",
    "build_data_model",
    "check_explanation",
    "choose_analysis",
    "compute",
    "explain_result",
    "load_table",
    "render_concepts",
    "render_dictionary",
    "render_flow",
    "render_result",
    "render_whitelist",
    "resolve_source",
    "run_analysis",
    "scan_issues",
    "validate_request",
    "validate_question",
]
