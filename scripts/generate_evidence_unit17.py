"""生成单元17 的课堂验收证据。

按任务卡的四节课顺序**真实执行**并抓取输出，数字与报错文案全部取自实际运行：

    uv run --extra deepseek --extra embedding --extra chroma --extra parse --extra langchain \
        python scripts/generate_evidence_unit17.py

覆盖要交的东西：
    第1节  执行器的部件与项目问题清单；工具接入情况
    第2节  最小业务场景（一个工具跑通）
    第3节  执行器 + 步骤追踪：连续调用两步，工具报错定位到第几步
    第4节  步数上限、清单外的工具、无效输入，以及同题对照单元15 手写的循环

**密钥安全**：脚本不打印密钥，输出统一做 sk- 掩码兜底。
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元17验收证据.md"

BOOTSTRAP = """
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))
"""

# 第1节：资料清单 + 执行器配置 + 工具接线 + 概念与问题清单。
PREPARE_FLOW = BOOTSTRAP + r"""
import os
from app.chroma_store import PROJECT_ID, resolve_unit15_db_path
from app.loaders import load_materials
from app.lc_agent import render_executor
from app.lc_agent_lab import KNOWLEDGE_DIR, build_stack, render_concepts, render_tool_binding
from app.parse_lab import EXPECTED_RECORDS

records, failures = load_materials(KNOWLEDGE_DIR, PROJECT_ID)
print(f"资料目录：{KNOWLEDGE_DIR}")
print(f"解析结果：{len(records)} 条（完成标准要求 {EXPECTED_RECORDS} 条）  解析失败：{failures or '无'}")
for record in records:
    print(f"  {record.file_type}\t{record.location_type}={record.location}\t"
          f"{record.source.rsplit('/', 1)[-1]}")

executor, _legacy, registry, store, kb_count, faq_rows = build_stack()
print()
print(f"向量集合：{store.collection.name}（本项目 {kb_count} 条，库目录 {resolve_unit15_db_path()}）")
print(f"项目标识：{PROJECT_ID}；FAQ 库 {faq_rows} 行")
print(f"模型：{type(executor.model).__name__}（LLM_PROVIDER={os.getenv('LLM_PROVIDER')}）"
      f"  密钥已配置：{bool(os.getenv('DEEPSEEK_API_KEY'))}")
print()
print(render_executor(executor))
print()
print(render_tool_binding(registry, executor))
print()
print(render_concepts())
"""

# 第2节：最小业务场景——只用到一个工具。
MINIMAL_FLOW = BOOTSTRAP + r"""
from app.lc_agent_lab import MINIMAL_QUESTION, build_stack, render_scenario, run_scenario

executor, _legacy, _registry, _store, _kb, _rows = build_stack()
print(render_scenario(run_scenario("minimal", MINIMAL_QUESTION, executor)))
"""

# 第3节：连续调用两个工具，并逐条列出每一步。
COMBO_FLOW = BOOTSTRAP + r"""
from app.lc_agent_lab import COMBO, build_stack, render_scenario, run_scenario

executor, _legacy, _registry, _store, _kb, _rows = build_stack()
print(render_scenario(run_scenario("combo", COMBO, executor)))
"""

# 第3、4节：异常路径——工具报错定位、步数上限、清单外的工具、无效输入。
EDGE_FLOW = BOOTSTRAP + r"""
from app.lc_agent_lab import (
    BAD_ARG_QUESTION,
    LIMIT_MAX_STEPS,
    LIMIT_QUESTION,
    PLANS,
    UNKNOWN_QUESTION,
    build_stack,
    render_invalid,
    render_scenario,
    run_invalid,
    run_scenario,
)

executor, _legacy, _registry, _store, _kb, _rows = build_stack()
print(render_scenario(run_scenario(
    "error", BAD_ARG_QUESTION, executor, plan=PLANS["error"], scripted=True)))
print()
print(render_scenario(run_scenario(
    "limit", LIMIT_QUESTION, executor, plan=PLANS["limit"],
    max_steps=LIMIT_MAX_STEPS, scripted=True)))
print()
print(render_scenario(run_scenario(
    "unknown", UNKNOWN_QUESTION, executor, plan=PLANS["unknown"], scripted=True)))
print()
print(render_invalid(run_invalid()))
"""

# 第4节：同一问题对照单元15 手写的循环。
COMPARE_FLOW = BOOTSTRAP + r"""
from app.lc_agent_lab import MINIMAL_QUESTION, build_stack, compare_with_legacy, render_comparison

executor, legacy_agent, _registry, _store, _kb, _rows = build_stack()
print(render_comparison(compare_with_legacy(MINIMAL_QUESTION, executor, legacy_agent)))
"""

# 换模型不换工具：同一批工具、同一份资料，换成离线剧本模型再跑一次。
SWAP_FLOW = BOOTSTRAP + r"""
from app.lc_agent import ScriptedToolModel
from app.lc_agent_lab import MINIMAL_QUESTION, PLANS, build_stack, render_result, run_agent

executor, _legacy, _registry, _store, _kb, _rows = build_stack(
    model=ScriptedToolModel(plan=PLANS["minimal"]))
run = run_agent(executor, MINIMAL_QUESTION)
print(f"模型：{run.model_name}")
print(f"调用的工具：{[step.tool for step in run.steps]}（与真实模型那一次是同一批工具）")
print(f"停止原因：{run.stop_reason}；工具调用 {len(run.steps)} 次")
print(f"回答：{run.answer[:200]}")
"""

BOUNDARIES = [
    (
        "问题为空：守门在跑执行器之前，模型与工具都不会被触发",
        BOOTSTRAP + r"""
from app.lc_agent import validate_question
try:
    validate_question("   ")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "问题不是字符串：类型也要先看一眼",
        BOOTSTRAP + r"""
from app.lc_agent import validate_question
try:
    validate_question({"question": "退货"})
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "步数上限写 0：构图时就报错，而不是跑到一半才发现",
        BOOTSTRAP + r"""
from app.lc_agent import ScriptedToolModel, build_executor
from app.lc_agent_lab import build_stack

executor, _legacy, registry, _store, _kb, _rows = build_stack()
try:
    build_executor(registry, ScriptedToolModel(), max_steps=0)
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "工具参数格式错：失败原因连同步号一起回填给模型",
        BOOTSTRAP + r"""
from app.lc_agent import run_agent, tool_step, ScriptedToolModel, build_executor
from app.lc_agent_lab import build_stack
from app.toolkit import TOOL_FAQ

executor, _legacy, registry, _store, _kb, _rows = build_stack()
bad = build_executor(registry, ScriptedToolModel(plan=[tool_step(TOOL_FAQ, question_id="123")]))
run = run_agent(bad, "EC-FAQ-01 的标准答复是什么？")
for step in run.steps:
    print(f"第{step.index}步 {step.tool}（{step.arguments}）成功={step.ok} 原因={step.error or '无'}")
print(f"我们自己的日志里这一条被记为：{bad.log.records[0].status}（失败也算一次调用）")
""",
    ),
    (
        "清单外的工具：模型提了也执行不了，我们自己的日志里一条都没有",
        BOOTSTRAP + r"""
from app.lc_agent import ScriptedToolModel, answer_step, build_executor, run_agent, tool_step
from app.lc_agent_lab import build_stack

executor, _legacy, registry, _store, _kb, _rows = build_stack()
rebel = build_executor(registry, ScriptedToolModel(plan=[
    tool_step("delete_collection", name="faq"),
    answer_step("该工具不在执行器的工具清单里。"),
]), log=executor.log)
run = run_agent(rebel, "帮我把知识库清空一下")
step = run.steps[0]
print(f"第{step.index}步：提议 {step.tool}({step.arguments})；成功={step.ok}")
print(f"定位信息：{step.error}")
print(f"我们的执行日志条数：{executor.log.total}（0 表示真的没执行）")
""",
    ),
    (
        "步数上限触顶：不产出业务答案，也不偷偷多问一次模型",
        BOOTSTRAP + r"""
from app.lc_agent import ScriptedToolModel, build_executor, run_agent, tool_step
from app.lc_agent_lab import build_stack
from app.toolkit import TOOL_FAQ

executor, _legacy, registry, _store, _kb, _rows = build_stack()
loop = build_executor(registry, ScriptedToolModel(
    plan=[tool_step(TOOL_FAQ, question_id="EC-FAQ-01") for _ in range(6)]), max_steps=2)
run = run_agent(loop, "一直查下去，越多越好")
print(f"停止原因：{run.stop_reason}")
print(f"工具调用次数：{len(run.steps)}；模型决策次数上限：{run.max_steps}")
print(f"最终答案：{run.answer or '（空）'}")
print(f"执行器提示：{run.limit_message}")
""",
    ),
    (
        "工具描述里没有内部细节：给模型看的只有用途与参数",
        BOOTSTRAP + r"""
from app.lc_agent_lab import build_stack

executor, _legacy, _registry, _store, _kb, _rows = build_stack()
for tool in executor.tools:
    print(f"{tool.name}：{tool.description}")
    print(f"  参数：{list(tool.args_schema.model_fields)}")
""",
    ),
    (
        "没配密钥时：真实模式当场报错，不静默降级到离线模型",
        BOOTSTRAP + r"""
import os
from app.lc_agent import build_agent_model

saved = os.environ.pop("DEEPSEEK_API_KEY", None)
try:
    build_agent_model(provider="deepseek")
except RuntimeError as error:
    print(f"{type(error).__name__}: {error}")
finally:
    if saved is not None:
        os.environ["DEEPSEEK_API_KEY"] = saved
""",
    ),
    (
        "provider 写错：不猜，直接说只支持两种",
        BOOTSTRAP + r"""
from app.lc_agent import build_agent_model
try:
    build_agent_model(provider="openai")
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


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.path.insert(0, str(PROJECT_ROOT))

    prepare_text = run_code(PREPARE_FLOW)
    minimal_text = run_code(MINIMAL_FLOW)
    combo_text = run_code(COMBO_FLOW)
    edge_text = run_code(EDGE_FLOW)
    compare_text = run_code(COMPARE_FLOW)
    swap_text = run_code(SWAP_FLOW)

    lines: list[str] = []
    lines.append("# 单元17 验收证据：LangChain Agents（执行器 + 工具接入 + 步骤日志）")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")
    lines.append("- 场景：本组电商售后问答；三项只读工具与资料**全部沿用单元15/12**，本单元不另造")
    lines.append("- 承诺：`app/toolkit.py`、`app/tool_agent.py`（单元15 手写的循环）**一行没删**，"
                 "新执行器不覆盖旧实现")
    lines.append("")
    lines.append("## 一、准备：执行器配置与工具接入（第1节）")
    lines.append("")
    lines.append("```text")
    lines.append(prepare_text)
    lines.append("```")
    lines.append("")
    lines.append("工具清单那一栏要看的是「接线」列：三项工具的**名字、说明、参数全部来自单元15 的注册表**，")
    lines.append("执行器只是把它们包了一层 BaseTool，模型挑名字、填参数，真正执行的还是注册时的那个函数。")
    lines.append("")
    lines.append("## 二、最小业务场景：一个工具先跑通（第2节）")
    lines.append("")
    lines.append("```text")
    lines.append(minimal_text)
    lines.append("```")
    lines.append("")
    lines.append("第2节只要求「跑通」，所以只用到 `lookup_faq`：按编号从只读 SQLite 取一行标准答复。")
    lines.append("")
    lines.append("## 三、连续调用与步骤追踪（第3节）")
    lines.append("")
    lines.append("```text")
    lines.append(combo_text)
    lines.append("```")
    lines.append("")
    lines.append("第2步发生在第1步的观察回来之后——这一步的先后由执行器决定，不是脚本排好的；")
    lines.append("每一步的输入、输出、耗时都留在步骤表里，所以事后能复述「它到底查了什么」。")
    lines.append("")
    lines.append("## 四、异常路径：工具报错、步数上限、清单外的工具（第3、4节）")
    lines.append("")
    lines.append("```text")
    lines.append(edge_text)
    lines.append("```")
    lines.append("")
    lines.append("这三个异常场景都用**离线剧本模型**跑（模型那一栏写着 `scripted-tool-agent`）：")
    lines.append("要的是可复现的异常路径——让模型「配合」一次坏参数、一次触顶——与真实模型好不好无关；")
    lines.append("工具、白名单、步骤日志和步数上限都是真的。")
    lines.append("")
    lines.append("工具报错那一栏要看的是**步号**：失败原因连同步号一起回填给模型，")
    lines.append("而不是一句笼统的「调用失败」。步数上限那一栏要看的是**回答为空**——")
    lines.append("触顶就是没答完，不能拿它当答案用。")
    lines.append("")
    lines.append("## 五、同题对照：执行器 vs 单元15 手写的循环（第4节）")
    lines.append("")
    lines.append("```text")
    lines.append(compare_text)
    lines.append("```")
    lines.append("")
    lines.append("两条路径用的是**同一批工具、同一份资料**；比的是调用的工具与数字这类硬事实，")
    lines.append("措辞不同不算问题，工具或数字不同才要人工核对。")
    lines.append("")
    lines.append("## 六、换模型不换工具")
    lines.append("")
    lines.append("```text")
    lines.append(swap_text)
    lines.append("```")
    lines.append("")
    lines.append("同一批工具、同一份资料，只把模型换成离线剧本模型：工具接线、白名单与步骤日志一行没动，")
    lines.append("说明「谁在决策」和「谁在执行」是分开的两件事。")
    lines.append("")
    lines.append("## 七、边界核对")
    lines.append("")
    lines.append("下面每一条都由 Python 在跑执行器**之前**或**执行过程中**拦下，")
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
    lines.append("- 运行产物 `.chroma_unit15/`（向量库）与 `.sqlite_unit15/`（FAQ 库）是本地文件，")
    lines.append("  不进仓库；换机器重跑 `/lcagent` 即可重建。")
    lines.append("- 想不调用 API 复现同一套流程，先 `LLM_PROVIDER=mock` 再跑：")
    lines.append("  模型换成离线剧本模型，工具、白名单、步骤日志与步数上限完全一致。")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
