"""单元23 验收证据：用真实运行抓取输出，不手写数字。

跑法：
    uv run --extra langchain python scripts/generate_evidence_unit23.py

产出：`docs/单元23验收证据.md`。统计结果、跳过行数、越权提示、数字核验结论全部来自
实际运行；密钥出现即掩码。

哪几段用真实模型：第二节（一次完整分析——口径与业务解释都由 DeepSeek 自己写，
最能说明"模型只做选择、不做计算"）。其余段落**强制用剧本**：真实模型不会主动去读
一份脏样本、也不会故意要一个越权字段，那几条异常路径就演示不到，而且每次运行的
数字都不一样，证据没法复核。这一点在文档里明写，不藏。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs" / "单元23验收证据.md"

BOOTSTRAP = '''
import os, sys
from pathlib import Path
ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT))
os.environ.pop("ACC_PRODUCT_CONFIG_V3", None)
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LLM_PROVIDER"] = "{provider}"
# 密钥只在 .env 里：先固定 LLM_PROVIDER，再加载 .env（load_dotenv 不覆盖已有变量），
# 否则子进程读不到 DEEPSEEK_API_KEY，真实模型段落会全部失败。
try:
    from dotenv import load_dotenv
    load_dotenv(r"{root}/.env")
except ImportError:
    pass
from app.data_agent import (
    ALLOWED_FIELDS, ALLOWED_METRICS, COLUMNS, DATA_FILE, DATA_ROOT, DIRTY_FILE,
    FORBIDDEN_FIELDS, MAX_FILE_BYTES, AnalysisRejected, build_data_model, load_table,
    render_concepts, render_dictionary, render_flow, render_result, render_whitelist,
    run_analysis, validate_request,
)
from app.data_lab import (
    QUESTION_OVERVIEW, SCENARIOS, render_file_guards, render_header, render_invalid,
    render_question_list, render_rejects, render_scenario, render_standards,
    render_summary_table, run_class, run_file_guards, run_invalid, run_rejects,
    run_scenario, source_without_execution,
)
'''

PREPARE_FLOW = '''
table = load_table(DATA_ROOT / DATA_FILE)
dirty = load_table(DATA_ROOT / DIRTY_FILE)
print("数据与权限（第3位成员负责的那一摊）：")
print("  数据目录：", DATA_ROOT.relative_to(ROOT).as_posix(), "（仓库内相对路径）")
print("  本组数据：", DATA_FILE, "（", table.row_count, "行 ×", len(table.columns), "列）")
print("  脏数据样本：", DIRTY_FILE, "（", dirty.row_count, "行，", len(dirty.issues), "处坏值；第4节用）")
print("  单个文件上限：", MAX_FILE_BYTES, "字节；只允许 .csv，且文件名不带目录分隔符")
print("  可分析字段（白名单）：", "、".join(sorted(ALLOWED_FIELDS)))
print("  表里存在、但不在白名单：", "、".join(sorted(FORBIDDEN_FIELDS)))
print("  固定算子：", "、".join(sorted(ALLOWED_METRICS)), "（共", len(ALLOWED_METRICS), "个）")
print("  列顺序与数据字典一致：", table.columns == COLUMNS)
print()
print(render_flow())
print()
print(render_dictionary())
print()
print(render_whitelist())
print()
print(render_question_list())
print()
print(render_concepts())
print()
print("脏数据样本是怎么来的：从", DATA_FILE, "派生，改了 7 处——")
for issue in dirty.issues:
    print("  ·", issue.render())
'''

REAL_FLOW = '''
result = run_analysis(build_data_model(provider="deepseek"), QUESTION_OVERVIEW)
print("任务：", result.question)
print()
print(render_result(result))
print()
print("这一段的要点：口径是模型自己选的（", result.request.render(), "），")
print("解释也是它自己写的，但金额与单数是 Python 算的——解释里每个带单位的数字都要")
print("能在统计结果里找到出处，找不到就会被点名（本段核验结论：",
      result.number_notes or "全部有出处", "）。")
'''

STEPS_FLOW = '''
class Recorder:
    """把每次喂给模型的提示词留下来：第3节要证明的就是"这两步各自读到了什么"。"""
    def __init__(self, inner):
        self.inner = inner
        self.seen = []
    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.inner.invoke(messages)

from app.data_agent import TAG_CHOOSE, TAG_EXPLAIN

model = Recorder(build_data_model(SCENARIOS["overview"].script, provider="mock"))
result = run_analysis(model, QUESTION_OVERVIEW)
print("两次模型调用各自读到了什么（逐次核对提示词，不靠猜）：")
print()
print("第几次\\t步骤\\t提示词字数\\t含数据字典\\t含白名单\\t含统计结果\\t模型输出")
for index, messages in enumerate(model.seen, start=1):
    text = "\\n".join(str(getattr(m, "content", "")) for m in messages)
    step = "选口径" if TAG_CHOOSE in text else ("解释" if TAG_EXPLAIN in text else "?")
    print(f"{index}\\t{step}\\t{len(text)}\\t"
          f"{'有' if '数据字典' in text else '无'}\\t"
          f"{'有' if '字段白名单' in text else '无'}\\t"
          f"{'有' if '算式' in text else '无'}\\t"
          f"{'JSON 口径' if step == '选口径' else '业务解释'}")
print()
print("第一步：模型只看到数据字典、白名单与算子，看不到数据，也拿不到路径。")
print("第二步：模型只看到已经算好的结果（含行数、算式、数据质量提示），被要求不许新增数字。")
print("数据本体从头到尾只在 Python 手里：模型既没读到 CSV，也没参与任何一次加法。")
print()
print("这次跑出来的结果：")
print(render_result(result))
'''

REFUND_FLOW = '''
for key in ("refund", "handoff"):
    print(render_scenario(run_scenario(key)))
    print()
print("数值聚合与带筛选计数：金额是 Python 逐行相加得到的，模型只说了「算哪个字段」。")
print("带筛选的计数连抽样都带上筛选字段的取值（needs_handoff=是），方便逐单复核。")
'''

DIRTY_FLOW = '''
print("故障注入：读派生出来的脏样本（真实模型不会主动去读它）")
print(render_scenario(run_scenario("dirty")))
print()
result = run_scenario("dirty").result
print("总行数：", result.rows_total, "；跳过：", result.rows_skipped, "；实际参与：", result.rows_used)
print("提示条数：", len(result.issues), "（全表逐格扫描，含空值与非数字两类）")
print("如果把坏值当成 0 算均值：", round(result.rows_used * result.value / result.rows_total, 2))
print("实际均值：", round(result.value, 2), "——差别就是「不静默当 0」的证据")
print("提醒：", "；".join(result.warnings))
'''

FORBIDDEN_FLOW = '''
print("故障注入：让模型去要订单编号（表里真有这一列，但它不在白名单）")
print(render_scenario(run_scenario("forbidden")))
print()
result = run_scenario("forbidden").result
print("是否被拒：", result.rejected, "；原因：", result.error)
print("统计次数：0（拒绝发生在统计之前）；解释调用：", result.explain_calls)
print("结果字段：", result.value, "；总行数：", result.rows_total)
'''

MISSING_FLOW = '''
print("故障注入：筛选一个表里不存在的状态（已关闭）")
print(render_scenario(run_scenario("missing")))
print()
result = run_scenario("missing").result
print("是否被拒：", result.rejected, "；匹配行数：", result.rows_matched)
print("提醒：", "；".join(result.warnings))
print("查不到 ≠ 失败：结果是空的，但要说清楚为什么空，而不是报错或猜一个数。")
'''

GUARD_FLOW = '''
print(render_invalid(run_invalid()))
print()
print(render_file_guards(run_file_guards()))
print()
print(render_rejects(run_rejects()))
print()
print("三道门：问题不合格、文件名越界、口径越权——都在统计之前拦下，一次都没算。")
'''

NOCODE_FLOW = '''
clean, note = source_without_execution()
print("源码核对：", note)
print("模块：app/data_agent.py（口径校验、读表、统计、数字核验都在这里）")
print()
print("算子白名单是封闭的：", "、".join(sorted(ALLOWED_METRICS)))
print("模型想塞一段表达式进来会怎样：")
try:
    validate_request({"metric": "sum(x for x in rows)", "field": "refund_amount"})
    print("  竟然通过了（异常！）")
except AnalysisRejected as error:
    print("  已拒绝：", error)
print()
print("模型拿不到什么：文件路径、SQL、任意代码、CSV 原文。它只输出一个 JSON 口径。")
'''

STANDARDS_FLOW = '''
run = run_class(offline=True)
print(render_header(run))
print()
print(render_summary_table(run))
print()
print(render_standards(run))
'''

SECTIONS: list[tuple[str, str, bool]] = [
    ("一、准备：本组 CSV、数据字典、字段白名单与固定算子", PREPARE_FLOW, False),
    ("二、第2节：一次完整分析（口径与业务解释都由真实模型写）", REAL_FLOW, True),
    ("三、第3节：两次模型调用各自读到什么、写回什么", STEPS_FLOW, False),
    ("四、第3节：数值统计与带筛选计数（金额由 Python 相加）", REFUND_FLOW, False),
    ("五、第4节：空值与非数字有提示 → 跳过并如实报数", DIRTY_FLOW, False),
    ("六、第4节：越权字段被白名单拦下（统计 0 次）", FORBIDDEN_FLOW, False),
    ("七、第4节：筛选后 0 行 → 如实说没有数据", MISSING_FLOW, False),
    ("八、第4节：无效问题、目录越界与越权口径被守门拦下", GUARD_FLOW, False),
    ("九、不执行任意生成代码（源码核对 + 算子白名单封闭）", NOCODE_FLOW, False),
    ("十、四条完成标准逐条验收", STANDARDS_FLOW, False),
]

BOUNDARIES = [
    ("问题为空 / 纯空白 / 非字符串 / 过短 / 超长", "`validate_question` 拦在问模型之前，模型调用 0 次"),
    ("文件名带目录、绝对路径、含 `..`", "`resolve_source` 拒绝：模型给不出路径，只能给文件名"),
    ("非 `.csv` 后缀 / 文件不存在 / 超大小", "`resolve_source` 拒绝；单文件上限 512 KB"),
    ("表头与数据字典不一致", "`load_table` 拒绝：列变了，白名单的意义也就没了"),
    ("字段不在白名单（标识列 / 表外字段）", "`validate_request` 在统计之前拒绝，统计 0 次、解释也不调"),
    ("算子不在 8 个固定算子内", "`validate_request` 拒绝；算子值是表达式（想塞代码）同样被拒"),
    ("数值字段为空 / 不是数字", "记 `CellIssue` 并**跳过**，绝不静默当 0；分组字段为空归入「（空值）」"),
    ("筛选后 0 行", "不是错误：记为提醒，结果为空并说明「表里没有这个取值」"),
    ("解释里出现结果之外的数字", "`check_explanation` 点名列出（只查带单位的数字，日期不算）"),
    ("会不会执行模型给的代码", "不会：模块里没有 `eval(` / `exec(` / `__import__(`，模型只输出 JSON 口径"),
    ("数据落盘", "不落：只读 CSV，不改不写；与单元19 的长期记忆相反"),
    ("脏数据样本", "从本组 CSV 派生的演示文件（7 处改动），只用于第4节；任务卡原表是干净的"),
]

MANUAL = [
    "统计结果与安全提示的**截图**：一次正常分析（口径 + 结果 + 业务解释），以及一次越权/空值提示。",
    "口头讲解：字段白名单为什么只放业务字段、空值为什么不静默当 0、"
    "结果怎么拿计算器复核、为什么「不执行任意生成代码」是本单元最关键的一条。",
    "共享进度表登记（本单元任务卡为 11 月 3 日上午，记在 11-03 的日报里）。",
]


def mask_secrets(text: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***（已掩码）", text)


def run_code(code: str, provider: str, root: Path) -> str:
    """另起一个进程跑一段代码；返回标准输出（失败也要把原因带回去）。"""
    full = BOOTSTRAP.format(root=root, provider=provider) + code
    env = dict(os.environ)
    env.pop("ACC_PRODUCT_CONFIG_V3", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["LLM_PROVIDER"] = provider
    result = subprocess.run(
        [sys.executable, "-c", full],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        output += f"\n[进程退出码 {result.returncode}]\n{result.stderr.strip()[-800:]}"
    return mask_secrets(output)


def build_document(root: Path, provider_real: str) -> str:
    lines: list[str] = []
    lines.append("# 单元23 验收证据：数据分析 Agent（安全读 CSV + 字段白名单）")
    lines.append("")
    lines.append(
        "本文件由 `scripts/generate_evidence_unit23.py` 抓取真实运行输出生成，"
        "统计结果、跳过行数、越权提示与数字核验结论都来自实际执行，不手写。"
    )
    lines.append("")
    lines.append("- 单元：23 数据分析 Agent（4 课时）")
    lines.append("- 本组选题：电商售后服务智能体（数据用任务卡给的 `电商售后工单.csv`）")
    lines.append(f"- 第二节的分析由 `{provider_real}` 真实跑一遍；其余段落用离线剧本模型（可复现）")
    lines.append("- 第五～七节是**故障注入**：强制用剧本，真实模型不会主动读脏样本、也不会故意要越权字段")
    lines.append("- 密钥：只在 `.env` 里，输出统一 `sk-` 掩码")
    lines.append("- 底线：**不执行任何生成代码**——模型只输出 JSON 口径（见第九节的源码核对）")
    lines.append("")

    for title, code, real in SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("```")
        lines.append(run_code(code, provider_real if real else "mock", root))
        lines.append("```")
        lines.append("")
        if title.startswith("五、"):
            lines.append(
                "这条是「空值和非数字有提示」的直接证据：坏值既不静默当 0，也不整行丢弃——"
                "跳过多少行、跳的是哪几行，全部点名；参与计算的行数也如实报出来。"
            )
            lines.append("")
        if title.startswith("六、"):
            lines.append(
                "这条是「字段白名单有效 / 越权字段被拒绝」的直接证据：`order_id` 在表里真的存在，"
                "但它不是度量、还涉及客户信息，所以统计一次都没发生，解释模型也没再调用。"
            )
            lines.append("")

    lines.append("## 十一、边界与注意事项（本单元核对过的地方）")
    lines.append("")
    lines.append("| 情况 | 处理 |")
    lines.append("| --- | --- |")
    for name, handling in BOUNDARIES:
        lines.append(f"| {name} | {handling} |")
    lines.append("")
    lines.append("## 十二、需要人工完成的部分")
    lines.append("")
    for index, item in enumerate(MANUAL, start=1):
        lines.append(f"{index}. {item}")
    return "\n".join(lines)


def main() -> None:
    provider = (os.getenv("LLM_PROVIDER", "deepseek") or "deepseek").strip().lower()
    if provider not in {"deepseek", "mock"}:
        provider = "deepseek"
    document = build_document(ROOT, provider)
    TARGET.write_text(document, encoding="utf-8")
    print(f"已生成：{TARGET}")
    print(f"字数：{len(document)}")


if __name__ == "__main__":
    main()
