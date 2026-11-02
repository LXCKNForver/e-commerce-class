"""单元22 验收证据：用真实运行抓取输出，不手写数字。

跑法：
    uv run --extra deepseek --extra embedding --extra chroma --extra parse \
        --extra evaluation --extra langchain python scripts/generate_evidence_unit22.py

产出：`docs/单元22验收证据.md`。消息流、终止原因、估算成本、复核结论全部来自
实际运行；密钥出现即掩码。

哪几段用真实模型：第二节（一次正常协作——规划、草稿、复核都由 DeepSeek 自己写，
最能说明"消息真的在群里流动"）。其余段落**强制用剧本**：真实模型不会故意少说一轮、
也不会自己写一份不合格的草稿，那样"轮次用完""复核不通过"就演示不到，而且每次运行
的数字都不一样，证据没法复核。这一点在文档里明写，不藏。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs" / "单元22验收证据.md"

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
from app.chat_crew import (
    CHARS_PER_TOKEN, DEFAULT_TOKEN_BUDGET, MAX_ROUNDS, ROLE_ORDER, ROLE_TAGS,
    STOP_NOTES, render_agents, render_concepts, render_evidence, render_flow,
    render_result, render_termination, render_transcript, render_verdicts, role_name,
)
from app.chat_lab import (
    BUDGET_SMALL, GUARD_CASES, INVALID_CASES, build_stack, render_guards, render_header,
    render_invalid, render_question_list, render_scenario, render_standards,
    render_state_table, render_summary_table, run_class, run_guards, run_invalid,
    run_scenario,
)
'''

PREPARE_FLOW = '''
from app.tool_lab import build_kit
_agent, registry, _store, kb, rows = build_kit()
print("资料与工具（沿用单元12/15，一个字没换）：")
print("  向量库资料条数：", kb, "（.chroma_unit15/）")
print("  FAQ 库行数：", rows, "（.sqlite_unit15/）")
print("  注册表里的工具：", "、".join(registry.names()))
print("  轮次上限：", MAX_ROUNDS, "轮（max_round = 轮次 + 1，发起者那条也占一格）")
print("  常规成本预算：", DEFAULT_TOKEN_BUDGET, "token；成本演示值：", BUDGET_SMALL)
print("  成本估算口径：", CHARS_PER_TOKEN, "字符 ≈ 1 token（不引 tokenizer，可手算复核）")
print()
print(render_flow())
print()
print(render_agents())
print()
print(render_question_list())
print()
print(render_concepts())
print()
print("关于 AutoGen：任务卡允许「AutoGen 或等价轻量实现」。")
print("  参考实现的 ConversableAgent / GroupChat / GroupChatManager 在本项目里对应")
print("  ChatAgent / GroupChat / GroupChatManager，行为一致：轮流发言、消息累积、")
print("  max_round 计入发起者那条、管理器自己不调模型。")
print("  差别只有一个：不引入 pyautogen 依赖（它会带进 openai/docker/flaml 一整套），")
print("  也不改写 langchain-core 1.7 的依赖树。取舍写在 app/chat_crew.py 文件头。")
'''

DELIVER_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("deliver", registry=registry, inject=False)
collab = scenario.collaboration
print("任务：", scenario.goal)
print()
print(render_evidence(collab.evidence))
print()
print(render_transcript(collab))
print()
print(render_termination(collab))
'''

STEPS_FLOW = '''
class Recorder:
    """把每次喂给模型的提示词留下来：第3节要证明的就是"这一位到底读到了什么"。"""
    def __init__(self, inner):
        self.inner = inner
        self.seen = []
    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.inner.invoke(messages)

from app.chat_crew import build_chat_model
from app.chat_lab import ANSWER_TEXT, PLAN_TEXT, SCENARIOS

registry, _model, kb, rows = build_stack()
recorder = Recorder(build_chat_model(SCENARIOS["deliver"].script, provider="mock"))
scenario = run_scenario("deliver", registry=registry, model=recorder, inject=False)
collab = scenario.collaboration
print("每一位发言的人从群里读到了什么（逐位核对提示词，不靠猜）：")
print()
print("第几位\\t角色\\t提示词字数\\t任务\\t工具证据\\t规划员原话\\t执行员原话")
for index, messages in enumerate(recorder.seen, start=1):
    text = "\\n".join(str(getattr(m, "content", "")) for m in messages)
    head = str(getattr(messages[0], "content", ""))
    role = next((r for r, tag in ROLE_TAGS.items() if tag in head), "?")
    print(f"{index}\\t{role_name(role)}\\t{len(text)}\\t"
          f"{'有' if '用户任务：' in text else '无'}\\t"
          f"{'有' if '平台工具证据' in text else '无'}\\t"
          f"{'有' if PLAN_TEXT in text else '无'}\\t"
          f"{'有' if ANSWER_TEXT in text else '无'}")
print()
print("写回群里的消息：")
for message in collab.agent_messages:
    print(f"  第{message.turn}位 {message.sender_name}：{message.chars} 字，"
          f"估算 {message.tokens} token——{' '.join(message.content.split())[:36]}")
print()
print("规划员是第一位：只看得到任务与证据；执行员多了规划员的原话；")
print("评审员看得到前两位的**全部原话**——这就是消息流，也是它与单元21 黑板的分界：")
print("黑板交接的是字段化的产物，群聊里流动的是每个人说过的话。")
'''

TIGHT_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("tight", registry=registry, inject=True)
print("故障注入：只给 2 轮（max_round = 3），复核角色根本没有轮到")
print(render_scenario(scenario))
print()
collab = scenario.collaboration
print("申请轮次：", collab.rounds_requested, "；max_round：", collab.max_round)
print("实际消息：", len(collab.messages), "条；实际发言：", len(collab.agent_messages), "次")
print("复核发过言吗：", collab.reviewed, "；交付的答复：", repr(collab.answer))
print("模型调用：", collab.model_calls, "次（到上限就停，没有偷偷补一轮给复核）")
'''

BUDGET_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("budget", registry=registry, inject=True)
print("故障注入：轮次给足 3 轮，但把 token 预算压到", BUDGET_SMALL)
print(render_scenario(scenario))
print()
collab = scenario.collaboration
print("估算 token：", collab.tokens_used, "> 预算", collab.token_budget)
print("消息条数：", len(collab.messages), "< max_round", collab.max_round,
      "——轮次还没用完，是钱先用完了")
print("终止原因：", collab.stop_reason, "→", STOP_NOTES[collab.stop_reason])
print("交付的答复：", repr(collab.answer))
'''

REJECT_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("reject", registry=registry, inject=True)
print("故障注入：复核结论按剧本给「不通过」（真实模型不会故意写坏草稿）")
print(render_scenario(scenario))
print()
collab = scenario.collaboration
print("复核结论：", collab.review)
print("交付的答复：", repr(collab.answer), "；是否交付：", collab.delivered)
print("草稿仍留在消息流里：第", len(collab.agent_messages), "条（人工复核时能看到它卡在哪一句）")
'''

REVISE_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("revise", registry=registry, inject=True)
print("故障注入：第1次复核不通过，群聊继续，执行员按意见再改一版")
print(render_scenario(scenario))
print()
collab = scenario.collaboration
print("复核次数：", len(collab.review_history), "；草稿版数：",
      len([m for m in collab.agent_messages if m.sender == "executor"]))
print("消息条数：", len(collab.messages), "（上限", collab.max_round, "）")
print("模型调用：", collab.model_calls, "次 = 发言条数", len(collab.agent_messages))
print("答复取自最后一版草稿：", collab.answer.splitlines()[0][:40])
print()
print("这一条说明：群聊不是一次就收尾——不通过就再来一轮，")
print("但轮次和成本数着走，不会无限往复。")
'''

GAP_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("gap", registry=registry, inject=True)
print("编号格式合法（EC-FAQ-99）但库里没有：证据照样取，取不到就如实说")
print(render_scenario(scenario))
print()
collab = scenario.collaboration
print("工具返回：", collab.evidence["faq"][0]["status"])
print("是否交付：", collab.delivered, "；答复里有没有说清资料不足：", "未找到" in collab.answer)
print("复核结论：", collab.review)
'''

GUARD_FLOW = '''
registry, _model, kb, rows = build_stack()
print(render_invalid(run_invalid(registry)))
print()
print(render_guards(run_guards(registry)))
print()
print("守门在让任何角色开口之前：任务、轮次、预算任一不合格，模型调用都是 0 次。")
'''

STANDARDS_FLOW = '''
run = run_class(offline=True)
print(render_header(run))
print()
print(render_summary_table(run))
print()
print(render_state_table(run))
print()
print(render_standards(run))
'''


SECTIONS: list[tuple[str, str, bool]] = [
    ("一、准备：资料、工具、群聊结构与角色表", PREPARE_FLOW, False),
    ("二、第2节：一次正常协作（规划、草稿、复核都由真实模型写）", DELIVER_FLOW, True),
    ("三、第3节：每位发言的人读到什么、写回什么", STEPS_FLOW, False),
    ("四、第4节：只给 2 轮 → 复核没轮到 → 未交付", TIGHT_FLOW, False),
    ("五、第4节：成本上限 → 钱先用完，轮次还有余 → 未交付", BUDGET_FLOW, False),
    ("六、第4节：复核不通过 → 草稿不变成答复", REJECT_FLOW, False),
    ("七、复核不通过后继续一轮 → 再改一版 → 二次复核通过", REVISE_FLOW, False),
    ("八、第4节：编号查不到 → 如实说明，复核照样通过", GAP_FLOW, False),
    ("九、第4节：无效任务与越界参数被守门拦下", GUARD_FLOW, False),
    ("十、四条完成标准逐条验收", STANDARDS_FLOW, False),
]

BOUNDARIES = [
    ("任务为空 / 纯空白 / 非字符串 / 过短 / 超长", "`validate_goal` 拦在让任何角色开口之前，模型调用 0 次"),
    ("轮次 <= 0 或 > 6、轮次不是整数", "`validate_rounds` 拒绝；写成 bool 也拒（True 会被当成 1）"),
    ("token 预算 <= 0 或不是整数", "`validate_budget` 拒绝——预算不设上限等于没有成本控制"),
    ("max_round 的口径", "消息**条数**上限，**含发起者那条**（与 AutoGen 一致）；3 个角色各发言一次要写 4"),
    ("复核输出不是 JSON / 缺布尔 passed", "一律判**不通过**（`parse_verdict` 返回 protocol_error），绝不「看不懂就当通过」"),
    ("某位角色返回空内容", "`ChatProtocolError`；不拿空字符串当发言"),
    ("复核通过但执行员没发过言", "不交付并记录异常：协作本身不成立，不能凭空交付"),
    ("模型调用次数", "等于发言条数；管理器 0 次（它只转发与推进轮次，这段在 `GroupChatManager.model_calls` 可核对）"),
    ("成本估算", "按字符估算（1.6 字符 ≈ 1 token），真值以服务商账单为准；演示值写进常量，谁都可用计算器复算"),
    ("群聊状态落盘", "不落：消息流只活在内存里，与单元19 的长期记忆相反，与单元20/21 的运行内状态一致"),
    ("工具调用", "群聊开始前由 Python 调单元15 的只读工具；群聊里没有工具，模型无法新增事实"),
    ("AutoGen 依赖", "未引入 pyautogen；按任务卡「或等价轻量实现」自行实现，取舍写在 `app/chat_crew.py` 文件头"),
]

MANUAL = [
    "一次完整协作的**截图**（消息流里三位角色的发言 + 最终答复），以及一次终止场景的提示。",
    "口头讲解：三个角色的输入输出、消息流为什么让后发言的人看得到前面的话、"
    "max_round 为什么要把发起者那条算进去、成本上限为什么独立于轮次、为什么复核才有放行权。",
    "共享进度表登记（本单元任务卡为 11 月 2 日下午，与单元21 合并记在同一天的日报里）。",
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
    lines.append("# 单元22 验收证据：AutoGen 多智能体实战（三角色群聊）")
    lines.append("")
    lines.append(
        "本文件由 `scripts/generate_evidence_unit22.py` 抓取真实运行输出生成，"
        "消息流、终止原因、估算成本与复核结论都来自实际执行，不手写。"
    )
    lines.append("")
    lines.append("- 单元：22 AutoGen 多智能体实战（4 课时）")
    lines.append("- 本组选题：电商售后服务智能体（资料与工具沿用单元12/15）")
    lines.append(f"- 第二节的协作由 `{provider_real}` 真实跑一遍；其余段落用离线剧本模型（可复现）")
    lines.append("- 第四～八节是**故障注入**：强制用剧本，真实模型不会故意少说一轮或写坏草稿")
    lines.append("- 密钥：只在 `.env` 里，输出统一 `sk-` 掩码")
    lines.append("- 实现取舍：未引入 `pyautogen`，按任务卡「或等价轻量实现」自行实现三个对象（见第一节末）")
    lines.append("")

    for title, code, real in SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("```")
        lines.append(run_code(code, provider_real if real else "mock", root))
        lines.append("```")
        lines.append("")
        if title.startswith("四、"):
            lines.append(
                "这条是「轮次受控」的直接证据：`max_round` 吃满就停，**不会**因为"
                "「还差一位没发言」而多给一轮。停下来时答复是空的——没复核完就不算答完。"
            )
            lines.append("")
        if title.startswith("五、"):
            lines.append(
                "这条要和上一条对着看：轮次还剩一格（`max_round` 没满），但估算 token 已经超预算，"
                "所以下一位不再开口。两条上限互相独立，任一条触发都算「未交付」。"
            )
            lines.append("")
        if title.startswith("六、"):
            lines.append(
                "这条是「结果由复核角色确认」的反面证据：草稿写得再像答复，复核不通过就"
                "**不是**答复。答复字段是空字符串，草稿只留在消息流里供人工追溯。"
            )
            lines.append("")
        if title.startswith("七、"):
            lines.append(
                "这条是「复核不通过不等于立刻收尾」的证据：群聊继续下一轮，执行员看到复核意见"
                "再改一版，第二轮复核通过后才交付，而且答复取的是**最后一版**草稿。"
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
