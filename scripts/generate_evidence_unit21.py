"""单元21 验收证据：用真实运行抓取输出，不手写数字。

跑法：
    uv run --extra deepseek --extra embedding --extra chroma --extra parse \
        --extra evaluation --extra langchain python scripts/generate_evidence_unit21.py

产出：`docs/单元21验收证据.md`。角色消息、观察内容、评审结论、交付与否全部来自
实际运行；密钥出现即掩码。

哪几段用真实模型：第二节（一次正常协作——计划、答复、评审都由 DeepSeek 自己写，
最有说服力）。第4节的四段**强制用剧本**：真实模型不会故意写出不合格的草稿，
那就演示不到"拒绝 → 修订 → 停止交付"这三条了；这一点在文档里明写，不藏。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs" / "单元21验收证据.md"

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
from app.multi_agent import (
    MAX_REVISIONS, ROLES, ROLE_COORDINATOR, STATUS_ABORTED, STATUS_DELIVERED,
    STATUS_HUMAN, render_concepts, render_flow, render_handoff, render_result,
    render_roles,
)
from app.multi_lab import (
    build_stack, render_header, render_invalid, render_question_list, render_scenario,
    render_standards, render_summary_table, run_class, run_invalid, run_scenario,
)
'''

PREPARE_FLOW = '''
from app.tool_lab import build_kit
_agent, registry, _store, kb, rows = build_kit()
print("资料与工具（沿用单元12/15，一个字没换）：")
print("  向量库资料条数：", kb, "（.chroma_unit15/）")
print("  FAQ 库行数：", rows, "（.sqlite_unit15/）")
print("  注册表里的工具：", "、".join(registry.names()))
print("  评审后最多修订：", MAX_REVISIONS, "次")
print("  黑板：只活在内存里（不落库，与单元19 的长期记忆相反）")
print()
print(render_flow())
print()
print(render_roles())
print()
print(render_question_list())
print()
print(render_concepts())
'''

DELIVER_FLOW = '''
registry, _model, kb, rows = build_stack()
# inject=False：计划、答复、评审都由真实模型写（LLM_PROVIDER=deepseek 时）。
scenario = run_scenario("deliver", registry=registry, inject=False)
board = scenario.board
print("目标：", scenario.goal)
print(render_result(board))
print()
print(render_handoff(board))
'''

STEPS_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("deliver", registry=registry, inject=True)
board = scenario.board
print("第3节要看的就是这张表：每个角色交出去的字段与拿回来的字段")
print("① 规划智能体 → 计划：")
for index, step in enumerate(board.plan, start=1):
    print(f"   第{index}步 {step['tool']}  参数：{step['arguments']}  理由：{step['reason']}")
print("② 执行智能体 → 观察：")
for row in board.observations:
    print(f"   第{row['step']}步 {row['tool']}  输入：{row['arguments']}"
          f"  返回字段：{sorted(row['observation'])}  耗时ms：{row['elapsed_ms']}")
print("③ 证据核验智能体 →", board.evidence)
print("④ 答复智能体 → 草稿", len(board.drafts), "版")
print("⑤ 评审智能体 →", board.review)
print()
print("结论：", board.status, "；答复是否交付：", bool(board.answer))
'''

REVISE_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("revise", registry=registry, inject=True)
print("故障注入：草稿故意不引用依据文件，评审结论按剧本给（真实模型不会故意写坏）")
print(render_scenario(scenario))
print()
board = scenario.board
print("修订次数：", board.revisions, "（上限", MAX_REVISIONS, "）")
print("草稿版数：", len(board.drafts), "；评审次数：", len(board.review_history))
print("修订后交付：", board.delivered)
'''

REJECT_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("reject", registry=registry, inject=True)
print("故障注入：两次评审都不通过（修订稿仍然没有引用依据文件）")
print(render_scenario(scenario))
print()
board = scenario.board
print("交付的答复：", repr(board.answer))
print("是否自动交付：", board.delivered)
print("状态：", board.status)
print("留在黑板上的草稿版数：", len(board.drafts), "（供人工复核）")
'''

GAP_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("gap", registry=registry, inject=True)
print("编号格式合法（EC-FAQ-99）但库里没有：证据核验判「不足」，答复如实说明")
print(render_scenario(scenario))
print()
board = scenario.board
print("证据是否充分：", board.evidence_sufficient)
print("缺什么：", board.evidence.get("missing"))
print("答复里有没有说清资料不足：", "资料不足" in board.answer)
'''

ABORT_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("abort", registry=registry, inject=True)
print("故障注入：把计划里的 question_id 改成 BAD（只在运行时改，原始资料与库没动）")
print(render_scenario(scenario))
print()
board = scenario.board
print("草稿版数：", len(board.drafts), "；评审次数：", len(board.review_history))
print("模型调用次数：", board.model_calls, "（规划 1 + 重规划 1，后面三个角色没启动）")
print("异常记录：", board.error)
'''

INVALID_FLOW = '''
registry, _model, kb, rows = build_stack()
print(render_invalid(run_invalid(registry)))
print()
print("守门在问任何角色之前：目标不合格就一个角色都不启动，模型调用 0 次。")
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
    ("一、准备：资料、工具、架构与角色", PREPARE_FLOW, False),
    ("二、第2节：一次正常协作（计划、答复、评审都由真实模型写）", DELIVER_FLOW, True),
    ("三、第3节：每个角色的输入字段与输出字段", STEPS_FLOW, False),
    ("四、第4节：评审拒绝 → 修订一次 → 通过 → 交付", REVISE_FLOW, False),
    ("五、第4节：修订后仍不通过 → 停止自动交付、转人工复核", REJECT_FLOW, False),
    ("六、第4节：证据不足 → 如实说明而不是编", GAP_FLOW, False),
    ("七、第4节：执行阶段终止 → 连草稿都没有，评审不启动", ABORT_FLOW, False),
    ("八、第4节：无效输入被守门拦下", INVALID_FLOW, False),
    ("九、三条完成标准逐条验收", STANDARDS_FLOW, False),
]

BOUNDARIES = [
    ("目标为空 / 纯空白 / 非字符串 / 过短 / 超长", "validate_goal 拦在问任何角色之前，一次模型调用都没有"),
    ("角色输出不是 JSON 或不是对象", "CrewProtocolError 直接报错，不把自由文本当结构化结论"),
    ("证据核验 / 评审缺字段", "报错；协议不符就停，绝不猜一个默认值"),
    ("答复草稿为空", "报错；空草稿不能进入评审"),
    ("评审不通过", "把理由交回答复智能体，**只修订一次**，只改表述不改工具事实"),
    ("修订后仍不通过", "答复留空、状态转人工，草稿与评审意见留在黑板上"),
    ("计划参数格式错（BAD 编号）", "执行阶段重规划一次（沿用单元20）；仍失败则终止，评审不启动"),
    ("编号格式合法但库里没有", "观察返回「未找到」，证据判不足，答复如实说明并建议转人工"),
    ("协调者", "不调模型，只做调度与放行判断；不许替任何角色写产物"),
    ("黑板落盘", "不落：一次协作的产物活在内存里，与单元19 的长期记忆相反"),
]

MANUAL = [
    "一次正常协作的**截图**（角色消息 + 最终答复）与一次评审拒绝后的修订或人工复核提示。",
    "口头讲解：五个角色各自的输入输出、共享黑板为什么能追溯、"
    "评审不通过为什么只修订一次、为什么仍未通过必须停止自动交付。",
    "共享进度表登记（本单元任务卡为 11 月 2 日，与当天课程合并记录）。",
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
    lines.append("# 单元21 验收证据：多智能体系统架构（五角色 + 共享黑板 + 协调者）")
    lines.append("")
    lines.append(
        "本文件由 `scripts/generate_evidence_unit21.py` 抓取真实运行输出生成，"
        "角色消息、观察内容、评审结论与交付与否都来自实际执行，不手写。"
    )
    lines.append("")
    lines.append("- 单元：21 多智能体系统架构（4 课时）")
    lines.append("- 本组选题：电商售后服务智能体（资料与工具沿用单元12/15）")
    lines.append(f"- 第二节的协作由 `{provider_real}` 真实跑一遍；其余段落用离线剧本模型（可复现）")
    lines.append("- 第四、五、六、七节是**故障注入**：强制用剧本，真实模型不会故意写坏草稿")
    lines.append("- 密钥：只在 `.env` 里，输出统一 `sk-` 掩码")
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
                "这条是「仍未通过不得自动交付」的直接证据：答复是空的、状态是转人工，"
                "但**草稿两版和两次评审意见都留在黑板上**——停下来不等于丢掉证据，"
                "人工复核时能看清它到底卡在哪一句。"
            )
            lines.append("")
        if title.startswith("七、"):
            lines.append(
                "执行阶段就终止时，证据核验、答复、评审三个角色**根本不启动**："
                "没有观察就没有草稿，没有草稿就无从评审。宁可空着，也不让后面三个"
                "角色对着空气写出一份像样的答复。"
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
