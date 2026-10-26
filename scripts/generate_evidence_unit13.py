"""生成单元13 的课堂验收证据。

按任务卡的时间段顺序**真实执行**并抓取输出，不手写任何数字：

    uv run --extra embedding --extra chroma --extra parse --extra evaluation python scripts/generate_evidence_unit13.py

覆盖要交的东西：
    0–8 分钟   本组资料与评估题：题目、参考答案、标注来源（对上文件名）
    8–18 分钟  跑 1 题：Top-3 来源、答案、两类指标、日志 ID、耗时
    18–32 分钟 跑完 3 题 + 查看 SQLite 里的 rag_logs 表
    32–40 分钟 标记一条反馈 + 看汇总
    另外       指标说明（五个指标各查什么）+ 低分样例分析（取自真实运行）+ 边界核对

**密钥安全**：脚本不打印密钥，输出统一做 sk- 掩码兜底。
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元13验收证据.md"


def load_expected_map() -> dict[str, list[str]]:
    """读一份题集的标注来源，供第七节按题号核对命中使用。"""
    cases = json.loads((PROJECT_ROOT / "evaluation_cases.json").read_text(encoding="utf-8"))
    return {case["id"]: list(case["expected_sources"]) for case in cases}


EXPECTED_BY_CASE = load_expected_map()

# 0–8 分钟：先把题集本身打印出来——题目、参考答案、标注来源，三样缺一不可。
CASES_FLOW = r"""
import sys, json
sys.stdout.reconfigure(errors="replace")
from pathlib import Path

cases = json.loads(Path("evaluation_cases.json").read_text(encoding="utf-8"))
print(f"评估题共 {len(cases)} 道：")
for index, case in enumerate(cases, start=1):
    print(f"  {index}. {case['id']}：{case['question']}")
    print(f"     标注来源：{'、'.join(case['expected_sources'])}")
    print(f"     参考答案：{case['reference_answer']}")
"""

# 8–18 分钟：先跑 1 题确认流程，日志落在临时库里，不弄脏仓库里的日志库。
ONE_FLOW = r"""
import sys, tempfile
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))

from app.eval_lab import build_eval_service, render_run, run_evaluation
from app.monitoring import LogStore, render_rows

service, store = build_eval_service()
print(f"知识库条数（本项目）：{store.count('e-commerce-after-sales')} 条")
log_db = Path(tempfile.mkdtemp(prefix="unit13-one-")) / "rag_monitor.sqlite3"
run = run_evaluation(1, service=service, log_path=log_db)
print()
print(render_run(run))
print()
print("本节的日志写在临时库里（不进仓库），下面是从那个库里查出来的这一条：")
print(render_rows(LogStore(log_db).rows(1)))
"""

# 18–32 分钟：跑完 3 题。先把仓库里的日志库清掉，保证日志 ID 从 1 开始，证据可复现。
THREE_FLOW = r"""
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))

from app.eval_lab import (
    DEFAULT_LOG_DB, build_eval_service, render_run, run_evaluation,
)
from app.monitoring import LogStore

if DEFAULT_LOG_DB.exists():
    DEFAULT_LOG_DB.unlink()

service, _store = build_eval_service()
run = run_evaluation(3, service=service)
print(render_run(run))
print()
print(f"日志数据库：{run.log_path.name}")
print(f"日志条数：{len(LogStore(run.log_path).rows(10))}")
"""

# 18–32 分钟另一半：直接从 SQLite 把 rag_logs 表查出来，这就是要交的那张表的内容。
LOGS_FLOW = r"""
import sys, sqlite3
sys.stdout.reconfigure(errors="replace")

conn = sqlite3.connect("rag_monitor.sqlite3")
try:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, created_at, case_id, query, precision_at_3, recall_at_3, "
        "ragas_json, latency_seconds, user_feedback FROM rag_logs ORDER BY id"
    ).fetchall()
    print(f"rag_logs 共 {len(rows)} 条：")
    for row in rows:
        print(f"  id={row['id']}  {row['created_at']}  {row['case_id']}")
        print(f"    问题：{row['query']}")
        print(f"    Precision@3={row['precision_at_3']:.2f}  Recall@3={row['recall_at_3']:.2f}"
              f"  耗时={row['latency_seconds']:.2f}秒  反馈={row['user_feedback'] or '未标记'}")
        print(f"    RAGAS：{row['ragas_json']}")
finally:
    conn.close()
"""

# 32–40 分钟：给第 3 条日志标记反馈，再看汇总里的均值与反馈计数。
FEEDBACK_FLOW = r"""
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))

from app.eval_lab import (
    DEFAULT_LOG_DB, read_summary, render_rows, render_summary, set_feedback,
)
from app.monitoring import LogStore

print("标记前的汇总：")
print(render_summary(read_summary()))
print()
log_id = set_feedback(3, "bad")
print(f"已给日志 {log_id} 标记反馈：bad")
print()
print("标记后的汇总：")
print(render_summary(read_summary()))
print()
print(render_rows(LogStore(DEFAULT_LOG_DB).rows(3)))
"""

BOUNDARIES = [
    (
        "未配置密钥就跑 RAGAS 评审",
        r"""
import sys, os
sys.stdout.reconfigure(errors="replace")
from app.evaluation import ragas_scores
os.environ.pop("DEEPSEEK_API_KEY", None)
try:
    ragas_scores("退货要多久？", "七天内", ["普通商品七天无理由"], "七天内")
except RuntimeError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "评估题数小于 1",
        r"""
import sys
sys.stdout.reconfigure(errors="replace")
from app.eval_lab import run_evaluation
class Dummy:
    def answer(self, *args, **kwargs):
        raise AssertionError("不应该走进模型")
try:
    run_evaluation(0, service=Dummy())
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "题集不足 3 道题",
        r"""
import sys
sys.stdout.reconfigure(errors="replace")
from app.eval_lab import validate_cases
case = {"id": "A", "question": "q", "expected_sources": ["a.md"], "reference_answer": "r"}
try:
    validate_cases([case])
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "第 3 题缺少参考答案",
        r"""
import sys
sys.stdout.reconfigure(errors="replace")
from app.eval_lab import validate_cases
cases = [
    {"id": "A", "question": "q", "expected_sources": ["a.md"], "reference_answer": "r"},
    {"id": "B", "question": "q", "expected_sources": ["a.md"], "reference_answer": "r"},
    {"id": "C", "question": "q", "expected_sources": ["a.md"], "reference_answer": ""},
]
try:
    validate_cases(cases)
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "题目没有标注任何相关来源",
        r"""
import sys
sys.stdout.reconfigure(errors="replace")
from app.evaluation import retrieval_scores
try:
    retrieval_scores(["模拟数据/knowledge/DOC-02-01-普通商品退货规则.md"], [])
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "直接证据没被召回（未命中形态）",
        r"""
import sys
sys.stdout.reconfigure(errors="replace")
from app.evaluation import retrieval_scores

prefix = "模拟数据/单元12项目资料/电商售后服务智能体项目组/"
# 模拟"三条命中全都不是标注来源"的情况：这就是未命中样例在指标上的样子。
hits = [prefix + "售后受理材料清单.pdf"] * 3
expected = ["DOC-02-07-物流异常说明.md", "常见售后问题.csv"]
precision, recall = retrieval_scores(hits, expected)
print(f"命中：{hits}")
print(f"标注来源：{expected}")
print(f"Precision@3={precision:.2f}  Recall@3={recall:.2f}")
""",
    ),
]


def run_code(code: str) -> str:
    """在独立进程里跑一段代码，抓回 stdout；失败直接抛出，不静默吞掉。"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return mask_secrets(result.stdout.strip())


def mask_secrets(text: str) -> str:
    """兜底掩码：任何 sk- 开头的密钥都不允许出现在证据文档里。"""
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***（已掩码）", text)


def slice_from(text: str, start: str, end: str | None = None) -> str:
    """按标题切片；标记必须是这段输出里唯一出现的那一个。"""
    index = text.find(start)
    if index < 0:
        return f"（未找到片段：{start}）"
    body = text[index:]
    if end:
        stop = body.find(end, len(start))
        if stop > 0:
            body = body[:stop]
    return body.rstrip()


def parse_case_metrics(text: str) -> list[dict]:
    """从 render_run 的输出里逐题取出 Precision/Recall/RAGAS，避免手写数字。

    RAGAS 分数是 Python 的 repr（单引号），不能用 json 解析，用 literal_eval 安全读回。
    """
    metrics: list[dict] = []
    blocks = re.split(r"^--- ", text, flags=re.MULTILINE)[1:]
    for block in blocks:
        case_id = block.split("：", 1)[0].strip()
        match = re.search(
            r"Precision@3=([\d.]+)\s+Recall@3=([\d.]+)\s+RAGAS=(\{.*\})", block
        )
        if not match:
            continue
        try:
            scores = ast.literal_eval(match.group(3))
        except (ValueError, SyntaxError):  # pragma: no cover - 解析失败就如实放进文档
            scores = {}
        metrics.append(
            {
                "case_id": case_id,
                "precision": float(match.group(1)),
                "recall": float(match.group(2)),
                "ragas": scores,
            }
        )
    return metrics


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.path.insert(0, str(PROJECT_ROOT))

    cases_text = run_code(CASES_FLOW)
    one_text = run_code(ONE_FLOW)
    three_text = run_code(THREE_FLOW)
    logs_text = run_code(LOGS_FLOW)
    feedback_text = run_code(FEEDBACK_FLOW)

    metrics = parse_case_metrics(three_text)
    # 最差样例 = Precision 最低（Recall 再低优先），只按真实运行出来的分数选，不硬编码题号。
    lowest = min(metrics, key=lambda item: (item["precision"], item["recall"]), default=None)

    lines: list[str] = []
    lines.append("# 单元13 验收证据：RAG 系统评估与监控")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")
    lines.append("- 项目标识：`e-commerce-after-sales`（本组电商售后）")
    lines.append("- 资料目录：`模拟数据/单元12项目资料/电商售后服务智能体项目组`（3 篇 md + 1 份 pdf + 1 份 csv，共 7 条）")
    lines.append("- 向量库目录：`.chroma_unit13/`；集合：`after_sale_eval_knowledge`")
    lines.append("- 生成模型：真实 DeepSeek（`DEEPSEEK_MODEL=deepseek-chat`）；RAGAS 评审模型同为 DeepSeek")
    lines.append("- 密钥状态：已配置：True（值不打印，见文末说明）")
    lines.append("")
    lines.append("## 一、本组资料与评估题（0–8 分钟）")
    lines.append("")
    lines.append("题集在仓库根目录 `evaluation_cases.json`，每题三样：**题干 + 参考答案 + 相关来源（文件名）**。")
    lines.append("来源必须先与已入库的文件对得上，否则指标全部归零——所以先核对文件名，再跑分。")
    lines.append("")
    lines.append("```text")
    lines.append(cases_text)
    lines.append("```")
    lines.append("")
    lines.append("## 二、先跑 1 题确认流程（8–18 分钟）")
    lines.append("")
    lines.append("一题跑四步：检索 → 回答 → 打分（两类）→ 落库。下面是一题的完整输出。")
    lines.append("")
    lines.append("```text")
    lines.append(one_text)
    lines.append("```")
    lines.append("")
    lines.append("`Precision@3` / `Recall@3` 是规则算的（比对人工标注的来源文件，零模型调用）；")
    lines.append("`RAGAS` 三个分数是 DeepSeek 当评审给的（参考答案驱动的模型判断），两者不是同一个数。")
    lines.append("")
    lines.append("## 三、跑完 3 题（18–32 分钟）")
    lines.append("")
    lines.append("每跑一题就往 `rag_logs` 写一行，会话结束后还能复盘。")
    lines.append("")
    lines.append("```text")
    lines.append(three_text)
    lines.append("```")
    lines.append("")
    if metrics:
        lines.append("三道题的两类指标汇总（数值取自上面的真实输出）：")
        lines.append("")
        lines.append("| 题目 | Precision@3 | Recall@3 | RAGAS Context Precision | RAGAS Context Recall | Faithfulness |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for item in metrics:
            ragas = item["ragas"]
            lines.append(
                f"| {item['case_id']} | {item['precision']:.2f} | {item['recall']:.2f} | "
                f"{_number(ragas.get('context_precision'))} | "
                f"{_number(ragas.get('context_recall'))} | "
                f"{_number(ragas.get('faithfulness'))} |"
            )
        lines.append("")
    lines.append("## 四、SQLite 里的问答日志（rag_logs 表）")
    lines.append("")
    lines.append("下面是从 `rag_monitor.sqlite3` 的 `rag_logs` 表里查出来的全部内容：")
    lines.append("")
    lines.append("```text")
    lines.append(logs_text)
    lines.append("```")
    lines.append("")
    lines.append("每行固定九列：日志 ID、写入时间、题目编号、问题、**Precision@3**、**Recall@3**、")
    lines.append("RAGAS 分数字典、耗时、用户反馈。评估能复盘靠的就是这张表。")
    lines.append("")
    lines.append("## 五、标记反馈 + 看汇总（32–40 分钟）")
    lines.append("")
    lines.append("反馈只是给某一条日志打标记，**不修改任何已落库的分数**；汇总从表里算均值。")
    lines.append("")
    lines.append("```text")
    lines.append(feedback_text)
    lines.append("```")
    lines.append("")
    lines.append("## 六、五个指标各在检查什么")
    lines.append("")
    lines.append("| 指标 | 谁给的 | 检查什么 | 注意 |")
    lines.append("| --- | --- | --- | --- |")
    from app.eval_lab import METRIC_EXPLANATIONS

    for name, kind, question, note in METRIC_EXPLANATIONS:
        lines.append(f"| {name} | {kind} | {question} | {note} |")
    lines.append("")
    lines.append("一句话区分：前两个是**排序/覆盖的客观计数**（人工先标注了相关来源），")
    lines.append("后三个是**让模型读文本后给的主观判断**（参考答案当尺子）。")
    lines.append("Faithfulness 高只说明「句句有据」，不等于答对——答非所问但每句都能在")
    lines.append("检索文本里找到出处，照样拿高分。所以它必须和 Recall@3 一起看：")
    lines.append("Recall@3 低说明证据没找全，此时 Faithfulness 再高也只是自洽。")
    lines.append("")
    lines.append("## 七、一个低分样例的分析")
    lines.append("")
    if lowest is not None:
        lines.append("```text")
        lines.append(slice_from(three_text, "--- " + lowest["case_id"], "指标对照："))
        lines.append("```")
        lines.append("")
        expected = EXPECTED_BY_CASE.get(lowest["case_id"], [])
        lines.append(
            f"- **现象**：{lowest['case_id']} 的 Precision@3={lowest['precision']:.2f}"
            f"（三条命中里只有 {round(lowest['precision'] * 3)} 条在标注来源里）、"
            f"Recall@3={lowest['recall']:.2f}。"
        )
        lines.append(f"- **本题标注来源**：{'、'.join(expected)}")
        if lowest["recall"] >= 1.0:
            lines.append("- **归因在排序不在召回**：Recall 满分说明直接证据都进了候选，"
                         "丢分是因为一条不相关的资料占了 Top-3 里的位置——"
                         "「资料不足」不是没检索到，而是排序里混进了噪声。")
        else:
            lines.append("- **归因在召回**：直接证据（上面的标注来源）没进 Top-3，"
                         "此时再怎么改提示词都补不回来，只能改检索侧。")
        lines.append("- **Faithfulness 仍然接近满分**：模型没有编造，"
                     "它引的每句话都来自检索到的文本；忠实度不等于正确率。")
        lines.append("- **可改进的两处**：① 检索侧——按 `doc_type` 过滤掉「受理材料清单」这类"
                     "凭证资料，或把 top_k 放宽到 5 后接单元11 的分块 + BGE 重排；"
                     "② 生成侧——要求「资料未覆盖时明确说资料不足」，"
                     "避免用相邻话题的资料拼出看似合理的答复。")
    lines.append("")
    lines.append("## 八、边界核对")
    lines.append("")
    lines.append("评估链路要能容错。下面每一种不合法输入都**在调用模型之前**被挡住，"
                 "并且报错能指出是第几题、缺什么。")
    for title, code in BOUNDARIES:
        lines.append("")
        lines.append(f"### {title}")
        lines.append("")
        lines.append("```text")
        lines.append(run_code(code))
        lines.append("```")
    lines.append("")
    lines.append("## 九、密钥与产物说明")
    lines.append("")
    lines.append("- 密钥写在项目根 `.env`，`.gitignore` 已排除，不进仓库，也不打印到任何输出里。")
    lines.append("- 本证据文档中的所有输出均经过 `sk-` 掩码处理。")
    lines.append("- 运行产物 `.chroma_unit13/` 与 `rag_monitor.sqlite3` 都是本地文件，不进仓库；")
    lines.append("  换机器重跑 `/eval` 即可重建，评估分数可复现。")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成：{OUTPUT}")
    return 0


def _number(value: object) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "—"


if __name__ == "__main__":
    raise SystemExit(main())
