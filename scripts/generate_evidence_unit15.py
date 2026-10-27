"""生成单元15 的课堂验收证据。

按任务卡的时间段顺序**真实执行**并抓取输出，数字与报错文案全部取自实际运行：

    uv run --extra embedding --extra chroma --extra parse python scripts/generate_evidence_unit15.py

覆盖要交的东西：
    0–15 分   区分知识资料与结构化记录：资料清单、FAQ 同步进 SQLite、工具清单
    15–30 分  单工具：lookup_faq 按编号命中一行
    30–45 分  连续调用：组合问题里两个不同工具各调一次
    45–60 分  拒绝路径（未知工具 / 非法参数）、计算工具与是否适用的说明

**密钥安全**：脚本不打印密钥，输出统一做 sk- 掩码兜底。
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元15验收证据.md"

BOOTSTRAP = """
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))
"""

# 0–15 分钟：两条数据源各自的落点，以及模型能看到的工具清单。
PREPARE_FLOW = BOOTSTRAP + r"""
import os
from app.chroma_store import PROJECT_ID, UNIT15_COLLECTION
from app.faq_db import count_faq, read_faq_rows
from app.loaders import load_materials
from app.tool_lab import (
    EXPECTED_FAQ_ROWS,
    EXPECTED_KB_RECORDS,
    FAQ_FILE,
    KNOWLEDGE_DIR,
    build_faq_db,
    build_kit,
)

records, failures = load_materials(KNOWLEDGE_DIR, PROJECT_ID)
print(f"资料目录：{KNOWLEDGE_DIR}")
print(f"非结构化解析：{len(records)} 条（完成标准要求 {EXPECTED_KB_RECORDS} 条）"
      f"  解析失败：{failures or '无'}")
for record in records:
    print(f"  {record.file_type}\t{record.location_type}={record.location}\t"
          f"{record.source.rsplit('/', 1)[-1]}")

rows = read_faq_rows(FAQ_FILE)
print(f"结构化原始文件：{FAQ_FILE.name}（{len(rows)} 行，完成标准要求 "
      f"{EXPECTED_FAQ_ROWS} 行）")
for row in rows:
    print(f"  {row['问题编号']}\t{row['来源编号']}\t{row['问题']}")

db_path, faq_rows = build_faq_db()
print(f"同步后的 SQLite：{db_path}（库内 {faq_rows} 行，与原始 CSV 一致）")

agent, registry, store, kb_count, _rows = build_kit()
print(f"向量集合：{store.collection.name}（本项目 {kb_count} 条）")
print(f"项目标识：{PROJECT_ID}")
print(f"决策模型：DeepSeek（LLM_PROVIDER={os.getenv('LLM_PROVIDER')}）"
      f"  密钥已配置：{bool(os.getenv('DEEPSEEK_API_KEY'))}")
print()
print(registry.descriptions())
"""

# 15–60 分钟：一次跑完四个场景，下面按时间段切片。
CLASS_FLOW = BOOTSTRAP + r"""
from app.tool_lab import render_run, run_class
print(render_run(run_class()))
"""

BOUNDARIES = [
    (
        "未知工具：模型提议了也执行不了",
        BOOTSTRAP + r"""
import json
from app.tool_agent import ToolAgent
from app.tool_lab import build_kit

# 用真实注册表（三项工具都在）才看得出：被拒的是多出来的那一项，不是清单空。
agent, registry, _store, _kb, _faq = build_kit()

class Proposal:
    def decide(self, messages):
        return json.dumps({"reason": "想直接删库", "action": "delete_collection",
                           "arguments": {"name": "faq"}}, ensure_ascii=False)

agent.decision_model = Proposal()
result = agent.run("请帮我删掉整个知识库")
print(f"停止原因：{result.stop_reason}  工具调用：{result.tool_calls} 次  答案：{result.answer!r}")
print(f"拒绝原因：{result.reject_reason}")
""",
    ),
    (
        "参数不合格式：不会被“顺手修正”后执行",
        BOOTSTRAP + r"""
import json
from app.tool_agent import ToolAgent
from app.tool_lab import build_kit

# 工具本身是授权的，被拒的是参数：拦在"格式"这一层，不是连工具名都不认。
agent, registry, _store, _kb, _faq = build_kit()

class BadArgs:
    def decide(self, messages):
        return json.dumps({"reason": "随便填个编号", "action": "lookup_faq",
                           "arguments": {"question_id": "123"}}, ensure_ascii=False)

agent.decision_model = BadArgs()
result = agent.run("EC-FAQ-01 的答复是什么")
print(f"停止原因：{result.stop_reason}  工具调用：{result.tool_calls} 次")
print(f"拒绝原因：{result.reject_reason}")
""",
    ),
    (
        "注入串当编号：格式校验直接挡下",
        BOOTSTRAP + r"""
from app.toolkit import authorize_tool, ToolRejected
try:
    authorize_tool("lookup_faq", {"question_id": "EC-FAQ-01; DROP TABLE faq"})
except ToolRejected as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "计算工具的天数必须先在资料原文里出现",
        BOOTSTRAP + r"""
from app.tool_lab import KNOWLEDGE_DIR
from app.toolkit import ToolRejected, compute_return_deadline
try:
    compute_return_deadline(KNOWLEDGE_DIR, "DOC-02-01-普通商品退货规则.md", "2026-12-01", "30")
except ToolRejected as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "finish 引用了没观察过的来源",
        BOOTSTRAP + r"""
import json
from app.tool_agent import ToolAgent

class Cite:
    def __init__(self):
        self.calls = 0
    def decide(self, messages):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"reason": "先查制度", "action": "search_knowledge",
                               "arguments": {"query": "普通商品退货时限是多久"}},
                              ensure_ascii=False)
        return json.dumps({"reason": "编一个来源", "action": "finish",
                           "answer": "依据 DOC-02-99 可以无条件退货",
                           "sources": ["DOC-02-99-不存在的资料.md"]}, ensure_ascii=False)

from app.tool_lab import build_kit
agent, registry, _store, _kb, _faq = build_kit()
agent.decision_model = Cite()
try:
    agent.run("普通商品退货时限是多久？")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "一次数据都没看到，却不肯说“资料不足”",
        BOOTSTRAP + r"""
import json
from app.tool_agent import ToolAgent
from app.tool_lab import build_kit

agent, registry, _store, _kb, _faq = build_kit()

class Guessing:
    def decide(self, messages):
        return json.dumps({"reason": "凭常识作答", "action": "finish",
                           "answer": "一般签收七天都能退", "sources": []}, ensure_ascii=False)

agent.decision_model = Guessing()
try:
    agent.run("退货要多久？")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "SQLite 是只读打开的：写不进去",
        BOOTSTRAP + r"""
import sqlite3
from app.faq_db import FAQ_TABLE, connect_readonly
from app.tool_lab import build_faq_db

db_path, _rows = build_faq_db()
connection = connect_readonly(db_path)
try:
    connection.execute(f"DELETE FROM {FAQ_TABLE}")
except sqlite3.OperationalError as error:
    print(f"{type(error).__name__}: {error}")
finally:
    connection.close()
""",
    ),
    (
        "max_steps 小于 2：看不到“行动后再决策”",
        BOOTSTRAP + r"""
from app.tool_lab import build_kit
agent, _registry, _store, _kb, _faq = build_kit()
try:
    agent.run("EC-FAQ-01 的标准答复是什么？", 1)
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "模型返回的东西不是 JSON",
        BOOTSTRAP + r"""
from app.tool_agent import ToolAgent
from app.tool_lab import build_kit

agent, registry, _store, _kb, _faq = build_kit()

class Chatty:
    def decide(self, messages):
        return "好的，我先查一下资料。<这不是 JSON>"

agent.decision_model = Chatty()
try:
    agent.run("退货要多久？")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
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


def slice_between(text: str, start: str, end: str | None = None) -> str:
    """按唯一标记切片，避免截取到别的段落。"""
    index = text.find(start)
    if index < 0:
        return f"（未找到片段：{start}）"
    body = text[index:]
    if end:
        stop = body.find(end, len(start))
        if stop > 0:
            body = body[:stop]
    return body.rstrip()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.path.insert(0, str(PROJECT_ROOT))

    prepare_text = run_code(PREPARE_FLOW)
    class_text = run_code(CLASS_FLOW)

    header = slice_between(class_text, "=== 单元15", "数据源与工具对应关系")
    mapping = slice_between(class_text, "数据源与工具对应关系", "--- 单工具")
    single = slice_between(class_text, "--- 单工具", "--- 连续调用")
    combo = slice_between(class_text, "--- 连续调用", "--- 拒绝路径")
    reject = slice_between(class_text, "--- 拒绝路径", "--- 计算工具")
    calc = slice_between(class_text, "--- 计算工具", "计算工具是否适用")
    calc_note = slice_between(class_text, "计算工具是否适用", None)

    lines: list[str] = []
    lines.append("# 单元15 验收证据：Agent 工具集")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")
    lines.append("- 场景：本组电商售后问答；工具由 Python 注册，模型只能从中挑选")
    lines.append("- 承诺：三项工具**全部只读**，未授权的工具与不合格式的参数在执行前被拦下")
    lines.append("")
    lines.append("## 一、准备：知识资料与结构化记录（0–15 分钟）")
    lines.append("")
    lines.append("```text")
    lines.append(prepare_text)
    lines.append("```")
    lines.append("")
    lines.append("同一份 `常见售后问题.csv` 在这里有两条去路：逐行进 Chroma 靠**相似度**被找到，")
    lines.append("整表进 SQLite 靠**编号**被精确命中。最后那段是模型能看到的接口说明——")
    lines.append("只有用途和参数，没有任何本地路径，也没有 SQL。")
    lines.append("")
    lines.append("## 二、数据源与工具的对应关系")
    lines.append("")
    lines.append("```text")
    lines.append(mapping)
    lines.append("```")
    lines.append("")
    lines.append("```text")
    lines.append(header.replace("=== 单元15：Agent 工具集 ===", "").strip())
    lines.append("```")
    lines.append("")
    lines.append("## 三、单工具：按编号精确命中一行（15–30 分钟）")
    lines.append("")
    lines.append("```text")
    lines.append(single)
    lines.append("```")
    lines.append("")
    lines.append("这一节要看的是第 1 轮：`lookup_faq` 走的是**固定参数化 SQL**")
    lines.append("（`SELECT ... WHERE question_id = ?`），问号只传值，模型拿不到改写语句的入口；")
    lines.append("库还是以只读 URI 打开的，写操作直接报错（见第七节）。")
    lines.append("")
    lines.append("## 四、连续调用：观察之后的再决策（30–45 分钟）")
    lines.append("")
    lines.append("```text")
    lines.append(combo)
    lines.append("```")
    lines.append("")
    lines.append("组合问题里两个事项各走各的数据源：先按编号查 SQLite，")
    lines.append("再拿另一个事项去检索 Chroma。第 2 轮的动作是模型**看到第 1 轮的观察之后**")
    lines.append("才选的——不是把问题拆开同时查，也不是同一个工具查两遍。")
    lines.append("")
    lines.append("## 五、拒绝路径：提议不等于执行（45–60 分钟）")
    lines.append("")
    lines.append("```text")
    lines.append(reject)
    lines.append("```")
    lines.append("")
    lines.append("两段轨迹的 `工具调用` 都是 0 次、答案为空：拒绝发生在**执行之前**，")
    lines.append("被拒的工具连一次数据库访问都没有发生。")
    lines.append("下面那张“直接调用注册表”的表是同一套校验的兜底演示——不经模型，结果可复现。")
    lines.append("")
    lines.append("## 六、计算工具：算的人必须是 Python（45–60 分钟，选做）")
    lines.append("")
    lines.append("```text")
    lines.append(calc)
    lines.append("```")
    lines.append("")
    lines.append("```text")
    lines.append(calc_note)
    lines.append("```")
    lines.append("")
    lines.append("关键约束是**天数必须来自资料**：`window_days=30` 会被拒，")
    lines.append("因为 `DOC-02-01` 里没有“30天”这个写法；模型只提供签收日期，日期加法由 Python 做。")
    lines.append("")
    lines.append("## 七、边界核对")
    lines.append("")
    lines.append("下面每一条错误都由 Python 在**执行之前**或**拿到结果之后立即**拦下，")
    lines.append("报错文案直接说明被拒绝的原因。")
    for title, code in BOUNDARIES:
        lines.append("")
        lines.append(f"### {title}")
        lines.append("")
        lines.append("```text")
        lines.append(run_code(code))
        lines.append("```")
    lines.append("")
    lines.append("## 八、密钥与产物说明")
    lines.append("")
    lines.append("- 密钥写在项目根 `.env`，`.gitignore` 已排除，不进仓库，也不出现在任何输出里。")
    lines.append("- 本证据文档中的所有输出均经过 `sk-` 掩码处理。")
    lines.append("- 运行产物 `.chroma_unit15/`（向量库）与 `.sqlite_unit15/`（FAQ 只读库）都是本地文件，")
    lines.append("  不进仓库；换机器重跑 `/tools` 即可重建，FAQ 库由原始 CSV 同步而来。")
    lines.append("- 想不调用 API 复现同一套流程，先 `LLM_PROVIDER=mock` 再跑：")
    lines.append("  离线脚本策略会按“先查编号、再查制度、算之前先取时限”扮演模型。")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
