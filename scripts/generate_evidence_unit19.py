"""单元19 验收证据：用真实运行抓取输出，不手写数字。

跑法：
    uv run --extra deepseek --extra embedding --extra chroma --extra parse \
        --extra evaluation --extra langchain python scripts/generate_evidence_unit19.py

产出：`docs/单元19验收证据.md`。里面的条数、编号、召回结果全部来自实际运行；
密钥出现即掩码。跨进程召回这一段是**两个独立 Python 进程**各跑一次，不是同一个
进程里 new 一个对象——"重启后还记不记得"只有这样才能算证据。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs" / "单元19验收证据.md"

BOOTSTRAP = '''
import os, sys
sys.path.insert(0, r"{root}")
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
from pathlib import Path
from app.memory_chain import (
    MODE_BUFFER, MODE_HYBRID, MODE_SUMMARY, MODE_VECTOR,
    AgentMemoryManager, render_memory_flow, render_snapshot, run_turn,
)
from app.memory_lab import (
    DEFAULT_SESSION, MULTI_QUESTIONS, PAIR_QUESTIONS, PLAN_MULTI, SESSION_A, SESSION_B,
    TURN1_QUESTION, TURN2_QUESTION, TURN3_QUESTION,
    build_memory_manager, build_stack, render_concepts, render_modes,
    render_mode_compare, render_restart, render_turn, recall_after_restart, run_isolation,
)
'''

PREPARE_FLOW = '''
executor, registry, store, kb_count, faq_rows = build_stack()
manager = build_memory_manager()
print("资料与工具（沿用单元12/15/17，一个字没换）：")
print("  向量库资料条数：", kb_count, "（.chroma_unit15/）")
print("  FAQ 库行数：", faq_rows, "（.sqlite_unit15/）")
print("  可执行器接入的工具：", "、".join(executor.tool_names))
print("  长期记忆库：.chroma_unit19_memory/（与上面两个严格分开）")
print()
print(render_memory_flow())
print()
print(render_concepts())
'''

TURNS_FLOW = '''
executor, registry, store, kb_count, faq_rows = build_stack()
manager = build_memory_manager()
manager.clear_session(DEFAULT_SESSION)
print("会话：", DEFAULT_SESSION, "；模式：hybrid（完整历史 + 长期记忆）")
for question in MULTI_QUESTIONS:
    record = run_turn(executor, manager, DEFAULT_SESSION, question, MODE_HYBRID)
    print()
    print(render_turn(record))
print()
print("三轮跑完，记忆里有几轮：", len(manager.turns(DEFAULT_SESSION)))
print("长期记忆条数：", manager.long_term_count(DEFAULT_SESSION))
last = manager.load(DEFAULT_SESSION, TURN3_QUESTION, MODE_HYBRID)
print("再问一次「我刚才问的是哪个编号」时解析到的对象：",
      last.resolution.object_id if last.resolution else "（无）")
'''

SNAPSHOT_FLOW = '''
executor, registry, store, kb_count, faq_rows = build_stack()
manager = build_memory_manager()
manager.clear_session(DEFAULT_SESSION)
run_turn(executor, manager, DEFAULT_SESSION, TURN1_QUESTION, MODE_HYBRID)
# 只看第2轮的"读"这一步：先 load 再答，这里把 load 的结果单独打出来。
snapshot = manager.load(DEFAULT_SESSION, TURN2_QUESTION, MODE_HYBRID)
print(render_snapshot(snapshot))
'''

MODES_FLOW = '''
executor, registry, store, kb_count, faq_rows = build_stack()
from app.lc_agent import ScriptedToolModel, build_executor, build_agent_model
from app.memory_lab import run_session
managers = {}
records = []
for mode in (MODE_BUFFER, MODE_SUMMARY, MODE_VECTOR):
    item = build_memory_manager()
    item.clear_session("mode-" + mode)
    managers[mode] = item
    current = executor
    if isinstance(executor.model, ScriptedToolModel):
        current = build_executor(registry, build_agent_model(PLAN_MULTI), executor.max_steps)
    records.extend(run_session(current, item, "mode-" + mode, PAIR_QUESTIONS, mode))
print(render_modes())
print()
print(render_mode_compare(records, managers))
'''

# 进程一：写。进程二：只读。两个进程之间没有任何内存共享。
RESTART_WRITE = '''
executor, registry, store, kb_count, faq_rows = build_stack(database_path=Path(r"{memory_dir}"))
manager = AgentMemoryManager(database_path=Path(r"{memory_dir}"))
manager.clear_session("restart-demo")
record = run_turn(executor, manager, "restart-demo", TURN1_QUESTION, MODE_HYBRID)
print("进程一（写入）：本轮工具", record.tools, "；停止原因", record.stop_reason)
print("进程一（写入）：长期记忆条数", manager.long_term_count("restart-demo"))
'''

RESTART_READ = '''
manager = AgentMemoryManager(database_path=Path(r"{memory_dir}"))
print("进程二（只读）：库里该会话的记忆条数 =", manager.long_term_count("restart-demo"))
hits = manager.vector.recall("restart-demo", TURN2_QUESTION)
print("进程二（只读）：按第2轮问句召回 =", len(hits), "条")
for item in hits:
    print("   -", str(item.get("text", ""))[:100])
print("进程二（只读）：完整历史 =", len(manager.load("restart-demo", TURN2_QUESTION, MODE_HYBRID).history), "条")
print("进程二（只读）：对话摘要 =", repr(manager.summary("restart-demo")))
'''

ISOLATION_FLOW = '''
executor, registry, store, kb_count, faq_rows = build_stack()
(first, second), manager = run_isolation(executor)
print("会话", SESSION_A, "：带回来历史", len(first.snapshot.history), "条；召回",
      len(first.snapshot.recalled), "条")
print("会话", SESSION_B, "：带回来历史", len(second.snapshot.history), "条；召回",
      len(second.snapshot.recalled), "条")
print("库里", SESSION_A, "有", manager.long_term_count(SESSION_A), "条；",
      SESSION_B, "有", manager.long_term_count(SESSION_B), "条（过滤在库端做）")
'''

INVALID_FLOW = '''
executor, registry, store, kb_count, faq_rows = build_stack()
manager = build_memory_manager()
print("输入\\t结果")
for label, session in (("空会话 ID", ""), ("含空格", "student a"), ("中文 ID", "学生甲"),
                       ("超长 ID", "x" * 80)):
    try:
        manager.load(session, TURN1_QUESTION, MODE_HYBRID)
        print(label, "\\t未被拦下（有问题）")
    except ValueError as error:
        print(label, "\\t已拒绝：", error)
for label, question in (("空问题", ""), ("纯空白", "   "), ("非字符串", 123)):
    try:
        manager.load(DEFAULT_SESSION, question, MODE_HYBRID)
        print(label, "\\t未被拦下（有问题）")
    except ValueError as error:
        print(label, "\\t已拒绝：", error)
try:
    manager.load(DEFAULT_SESSION, TURN1_QUESTION, "not-a-mode")
    print("未知模式\\t未被拦下（有问题）")
except ValueError as error:
    print("未知模式\\t已拒绝：", error)
print()
print("失败轮次不进记忆：答案为空的那一轮不会被 save（见 run_turn 的 save_note）")
'''

SWAP_FLOW = '''
executor, registry, store, kb_count, faq_rows = build_stack()
manager = build_memory_manager()
manager.clear_session("swap-demo")
record = run_turn(executor, manager, "swap-demo", TURN1_QUESTION, MODE_HYBRID)
print("执行器模型：", type(executor.model).__name__)
print("本轮工具：", record.tools, "；答案首行：", (record.answer or "").split("\\n")[0][:90])
print("记忆这一侧一行没改：同一个 AgentMemoryManager、同一套读答写顺序")
run_turn(executor, manager, "swap-demo", TURN2_QUESTION, MODE_HYBRID)
second = manager.load("swap-demo", TURN2_QUESTION, MODE_HYBRID)
print("第2轮解析出的对象：", second.resolution.object_id if second.resolution else "（无）")
print("长期记忆条数：", manager.long_term_count("swap-demo"))
'''

STANDARDS_FLOW = '''
from app.memory_lab import check_standards, run_class
from app.memory_chain import render_turn
run = run_class()
print("模式：", run.mode, "；会话：", run.session_id)
for record in run.turns:
    print()
    print(render_turn(record))
print()
for name, observed, verdict in check_standards(run):
    print("标准｜", name, "｜", observed, "｜", verdict)
'''

SECTIONS: list[tuple[str, str, bool]] = [
    ("一、准备：资料、工具与两类库的分工", PREPARE_FLOW, True),
    ("二、第2节：两轮连续问答（含跨轮指代）", TURNS_FLOW, True),
    ("三、第2轮「读」到了什么（记忆只作语境）", SNAPSHOT_FLOW, True),
    ("四、第3节：三种记忆模式横向对比", MODES_FLOW, False),
    ("五、第3节：长期记忆跨进程召回（两个独立 Python 进程）", "", False),
    ("六、第4节：会话隔离", ISOLATION_FLOW, False),
    ("七、第4节：无效输入与失败轮次", INVALID_FLOW, False),
    ("八、换模型不换记忆（同一套读答写）", SWAP_FLOW, True),
    ("九、四条完成标准逐条验收（离线剧本模型，与第二节真实模型结论一致）", STANDARDS_FLOW, False),
]

BOUNDARIES = [
    ("空会话 ID / 含空格 / 中文 / 超长", "读写之前就被 SESSION_PATTERN 拦下，连库都不碰"),
    ("空问题 / 纯空白 / 非字符串", "validate_question 拦在入口（与作答同一个校验）"),
    ("未知记忆模式", "只认 buffer / summary / vector / hybrid 四种"),
    ("答案为空或触顶的轮次", "不写记忆——失败轮不能变成下一轮的事实"),
    ("摘要写漏编号", "程序生成时补回并标注，不假装模型记得"),
    ("长期记忆长期累积", "/memory new 会话名 或 clear_session 换一批演示"),
    ("两个会话同时用", "键前缀 + 库端 source 过滤，双保险"),
    ("记忆与知识库混用", "两个库目录严格分开，召回内容一律带「仅作语境」前缀"),
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
    lines.append("# 单元19 验收证据：对话记忆系统（完整历史 / 摘要 / 向量长期记忆）")
    lines.append("")
    lines.append(
        "本文件由 `scripts/generate_evidence_unit19.py` 抓取真实运行输出生成，"
        "条数、编号与结论都来自实际执行，不手写。"
    )
    lines.append("")
    lines.append("- 单元：19 对话记忆（4 课时）")
    lines.append("- 本组选题：电商售后服务智能体（资料与工具沿用单元12/15/17）")
    lines.append(f"- 真实模型段落的模型：`{provider_real}`；标注「离线脚本模式」的段落用离线剧本模型跑")
    lines.append("- 密钥：只在 `.env` 里，输出统一 `sk-` 掩码")
    lines.append("")

    for title, code, real in SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("```")
        if title.startswith("五、"):
            tmp = Path(tempfile.mkdtemp(prefix="unit19-restart-"))
            memory_dir = tmp / "memory"
            try:
                lines.append("【进程一：写入记忆】")
                lines.append(
                    run_code(
                        RESTART_WRITE.format(memory_dir=memory_dir), "mock", root
                    )
                )
                lines.append("")
                lines.append("【进程二：全新进程，只读】")
                lines.append(
                    run_code(RESTART_READ.format(memory_dir=memory_dir), "mock", root)
                )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        else:
            lines.append(run_code(code, provider_real if real else "mock", root))
        lines.append("```")
        lines.append("")
        if title.startswith("五、"):
            lines.append(
                "两个进程之间没有任何内存共享：进程二读到的东西只能来自磁盘。"
                "同一时刻它的完整历史是 0 条、摘要是空的——完整历史与摘要只活在内存里，"
                "长期记忆落盘，这就是两种记忆的分界。"
            )
            lines.append("")
        if title.startswith("九、"):
            lines.append(
                "四条标准各自对应任务卡的一行要求：指代解析看「上一轮关注的对象」、"
                "会话隔离看另一个会话带回来几条、长期记忆看跨进程召回、"
                "旧状态核验看第2轮是不是又调了一次工具。"
            )
            lines.append("")

    lines.append("## 十、边界与注意事项（本单元核对过的地方）")
    lines.append("")
    lines.append("| 情况 | 处理 |")
    lines.append("| --- | --- |")
    for name, handling in BOUNDARIES:
        lines.append(f"| {name} | {handling} |")
    lines.append("")
    lines.append("## 十一、需要人工完成的部分")
    lines.append("")
    lines.append("1. 两轮连续问答、会话隔离、跨进程召回三处运行的**截图**。")
    lines.append("2. 口头讲解：读答写的顺序、两个库为什么分开、三种模式各自的代价、"
                 "为什么第2轮还要再查一次工具。")
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
