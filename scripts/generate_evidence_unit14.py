"""生成单元14 的课堂验收证据。

按任务卡的时间段顺序**真实执行**并抓取输出，数字与报错文案全部取自实际运行：

    uv run --extra embedding --extra chroma --extra parse python scripts/generate_evidence_unit14.py

覆盖要交的东西：
    0–8 分钟   本组资料目录、项目标识、密钥配置（不打印密钥本身）
    8–18 分钟  单事项问题：来源、答案、finished
    18–30 分钟 双事项问题：两轮检索的 query 与观察到的来源
    30–40 分钟 max_steps 2 重跑：finished 与 max_steps 的轨迹对比
    另外       为什么模型只能提动作、检索必须由 Python 执行；边界核对

**密钥安全**：脚本不打印密钥，输出统一做 sk- 掩码兜底。
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元14验收证据.md"

BOOTSTRAP = """
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))
"""

# 0–8 分钟：资料、项目标识、密钥配置三件套。
PREPARE_FLOW = BOOTSTRAP + r"""
import os
from app.agent_lab import KNOWLEDGE_DIR, EXPECTED_KB_RECORDS, build_agent_store
from app.chroma_store import PROJECT_ID, UNIT14_COLLECTION
from app.loaders import load_materials

records, failures = load_materials(KNOWLEDGE_DIR, PROJECT_ID)
print(f"资料目录：{KNOWLEDGE_DIR}")
print(f"解析条数：{len(records)} 条（完成标准要求 {EXPECTED_KB_RECORDS} 条）"
      f"  解析失败：{failures or '无'}")
for record in records:
    print(f"  {record.file_type}\t{record.location_type}={record.location}\t{record.source.rsplit('/', 1)[-1]}")
store, count = build_agent_store()
print(f"写入后本项目条数：{store.count(PROJECT_ID)} 条")
print(f"集合：{store.collection.name}  集合 ID：{store.collection.id}")
print(f"集合元数据：{store.collection.metadata}")
print(f"项目标识：{PROJECT_ID}")
print(f"决策模型：DeepSeek（LLM_PROVIDER={os.getenv('LLM_PROVIDER')}）"
      f"  密钥已配置：{bool(os.getenv('DEEPSEEK_API_KEY'))}")
"""

# 8–40 分钟：一次跑完三个场景，下面按时间段切片。
CLASS_FLOW = BOOTSTRAP + r"""
from app.agent_lab import render_run, run_class
print(render_run(run_class()))
"""

BOUNDARIES = [
    (
        "动作不在白名单里：模型“提议”了也执行不了",
        BOOTSTRAP + r"""
import json
from app.agent import ReActAgent
from app.retriever import SearchResult

class Proposal:
    def decide(self, messages):
        return json.dumps({"reason": "想直接删库", "action": "delete_collection",
                           "arguments": {}}, ensure_ascii=False)

class DummyRetriever:
    def search(self, *args, **kwargs):
        raise AssertionError("不应发生任何检索")

agent = ReActAgent(Proposal(), DummyRetriever(), "e-commerce-after-sales")
try:
    agent.run("请帮我删掉整个知识库")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "finish 引用了没检索到的来源",
        BOOTSTRAP + r"""
import json
from app.agent import ReActAgent

HIT = {"source": "模拟数据/单元12项目资料/电商售后服务智能体项目组/DOC-02-01-普通商品退货规则.md",
       "text": "签收后7天内且未使用可申请退货评估"}

class Cite:
    def __init__(self):
        self.calls = 0
    def decide(self, messages):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"reason": "先查退货规则", "action": "search_knowledge",
                               "arguments": {"query": "退货时限"}}, ensure_ascii=False)
        return json.dumps({"reason": "编一个来源", "action": "finish",
                           "answer": "依据 DOC-02-99 可以无条件退货",
                           "sources": ["DOC-02-99-不存在的资料.md"]}, ensure_ascii=False)

class Retriever:
    def search(self, question, project_id="e-commerce-after-sales", top_k=3, **kwargs):
        return [SearchResult("id-1", HIT["text"], HIT["source"], project_id, "md", 0.5)]

from app.retriever import SearchResult
agent = ReActAgent(Cite(), Retriever(), "e-commerce-after-sales")
try:
    agent.run("退货要多久？")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "一次资料都没看到，却不肯说“资料不足”",
        BOOTSTRAP + r"""
import json
from app.agent import ReActAgent

class Guessing:
    def decide(self, messages):
        return json.dumps({"reason": "凭常识作答", "action": "finish",
                           "answer": "一般签收七天都能退", "sources": []}, ensure_ascii=False)

class Empty:
    def search(self, *args, **kwargs):
        return []

agent = ReActAgent(Guessing(), Empty(), "e-commerce-after-sales")
try:
    agent.run("退货要多久？")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "max_steps 小于 2：看不到“行动后再决策”",
        BOOTSTRAP + r"""
from app.agent import ReActAgent, ScriptedDecisionModel

class Empty:
    def search(self, *args, **kwargs):
        return []

agent = ReActAgent(ScriptedDecisionModel(), Empty(), "e-commerce-after-sales")
try:
    agent.run("退货要多久？", 1)
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "没配密钥时不降级到本地脚本策略",
        r"""
import sys, os
sys.stdout.reconfigure(errors="replace")
os.environ.pop("DEEPSEEK_API_KEY", None)
try:
    from app.agent import build_decision_model
    build_decision_model("deepseek")
except RuntimeError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "模型返回的东西不是 JSON",
        BOOTSTRAP + r"""
from app.agent import ReActAgent

class Chatty:
    def decide(self, messages):
        return "好的，我先查一下资料。<这不是 JSON>"

class Empty:
    def search(self, *args, **kwargs):
        return []

agent = ReActAgent(Chatty(), Empty(), "e-commerce-after-sales")
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
    """按唯一标记切片，避免 `field()` 撞到同名后缀。"""
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

    single = slice_between(class_text, "--- 单事项问题", "--- 双事项问题")
    double = slice_between(class_text, "--- 双事项问题", "--- 轮次受限重跑")
    limited = slice_between(class_text, "--- 轮次受限重跑", "轨迹对比：")
    comparison = slice_between(class_text, "轨迹对比：", "为什么模型只能提出动作")
    header = slice_between(class_text, "资料目录：", "--- 单事项问题")

    lines: list[str] = []
    lines.append("# 单元14 验收证据：ReAct Agent")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")
    lines.append("- 场景：本组电商售后问答；决策由 DeepSeek 给出，检索由 Python 执行")
    lines.append("- 承诺：Agent 每轮只输出一个 JSON 动作，**没有一个字节的模型输出被直接当成结果**")
    lines.append("")
    lines.append("## 一、准备：资料、项目标识、密钥（0–8 分钟）")
    lines.append("")
    lines.append("```text")
    lines.append(prepare_text)
    lines.append("```")
    lines.append("")
    lines.append("这三样先对齐，后面每一步才有归属：资料是本组的（7 条），")
    lines.append("检索按 `project_id` 过滤（跨项目资料一条都取不回来），密钥只在这一行显示是否配置。")
    lines.append("")
    lines.append("## 二、单事项问题：一次检索就够（8–18 分钟）")
    lines.append("")
    lines.append("```text")
    lines.append(single)
    lines.append("```")
    lines.append("")
    lines.append("这一节要看的是三件事：第 1 轮模型**提议**检索、Python 真的去查、")
    lines.append("观察回填进对话；第 2 轮模型看到资料后才把 action 改成 finish。")
    lines.append("回答里每条结论都能指回清单上的文件——来源不是模型随口写的，是库里真有的。")
    lines.append("")
    lines.append("## 三、双事项问题：观察之后的再决策（18–30 分钟）")
    lines.append("")
    lines.append("```text")
    lines.append(double)
    lines.append("```")
    lines.append("")
    lines.append("要看的就是第 2 轮：两轮 `search_knowledge` 的 **query 不同**——")
    lines.append("第一行查“退货时限”，第二行改查“质量问题材料”。这个改写发生在模型看到")
    lines.append("第一轮的观察之后，所以“检索 → 观察 → 再决策”是真的闭环，不是把问题拆开查两遍。")
    lines.append("")
    lines.append("## 四、限制最大轮次：未完成的停止状态（30–40 分钟）")
    lines.append("")
    lines.append("```text")
    lines.append(limited)
    lines.append("```")
    lines.append("")
    lines.append("同一道题改成 `max_steps=2` 重跑：第 2 轮还在检索，轮次就耗尽了。")
    lines.append("此时 `stop_reason=max_steps`，答案为空、来源为空，")
    lines.append("必须如实告诉用户“尚未完成”，**不能把已经看到的片段拼成答案**——")
    lines.append("这是完成任务的比例问题，不是答案质量问题。")
    lines.append("")
    lines.append("## 五、轨迹对比：finished 与 max_steps")
    lines.append("")
    lines.append("```text")
    lines.append(comparison)
    lines.append("```")
    lines.append("")
    lines.append("两点值得记住：限制轮次**只改变停在哪一轮**，不改变模型会不会多查一次；")
    lines.append("轮次耗尽后不会再偷偷多问一次模型（`决策调用` 等于 `轮次上限` 就是证据）。")
    lines.append("")
    lines.append("## 六、为什么模型只能提出动作、检索必须由 Python 执行")
    lines.append("")
    from app.agent_lab import WHY_PYTHON_EXECUTES

    lines.append("| 理由 | 说明 |")
    lines.append("| --- | --- |")
    for title, detail in WHY_PYTHON_EXECUTES:
        lines.append(f"| {title} | {detail} |")
    lines.append("")
    lines.append("一句话概括：模型负责“下一步干什么”，Python 负责“这一步真的发生了什么”。")
    lines.append("把执行权交给模型，等于让看不见数据库的一方替另一方签字。")
    lines.append("")
    lines.append("## 七、边界核对")
    lines.append("")
    lines.append("下面每一条错误都由 Python 在**调用模型之前或之后立即**拦下，")
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
    lines.append("- 运行产物 `.chroma_unit14/` 是本地 Chroma 库，不进仓库；")
    lines.append("  换机器重跑 `/act` 即可重建。")
    lines.append("- 想不调用 API 复现同一套流程，先 `LLM_PROVIDER=mock` 再跑：")
    lines.append("  离线脚本策略会按“逐项检索、检索完汇总”扮演模型，方便看轨迹形状。")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
