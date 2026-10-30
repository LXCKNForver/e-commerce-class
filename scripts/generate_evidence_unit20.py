"""单元20 验收证据：用真实运行抓取输出，不手写数字。

跑法：
    uv run --extra deepseek --extra embedding --extra chroma --extra parse \
        --extra evaluation --extra langchain python scripts/generate_evidence_unit20.py

产出：`docs/单元20验收证据.md`。步数、观察内容、重规划次数、终止原因全部来自
实际运行；密钥出现即掩码。

哪几段用真实模型：第二、三节（计划由 DeepSeek 自己拆，最有说服力）。
故障验证的三段（重规划 / 终止 / 未找到）**强制用剧本**——真实模型会把 BAD 编号
自己改对，那就演示不到失败与终止了；这一点在文档里明写，不藏。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs" / "单元20验收证据.md"

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
from app.plan_lab import (
    GOAL_MINIMAL, GOAL_MULTI, SCENARIO_ORDER, build_stack, render_flow, render_header,
    render_invalid, render_question_list, render_scenario, render_standards,
    render_summary_table, run_class, run_invalid, run_scenario,
)
from app.plan_solver import (
    MAX_PLAN_STEPS, MAX_REPLANS, PLANNER_ALLOWED, STATUS_ABORTED, STATUS_DONE,
    build_plan_model, render_concepts, render_plan, render_result, render_trajectory, solve,
)
'''

PREPARE_FLOW = '''
from app.tool_lab import build_kit
_agent, registry, _store, kb, rows = build_kit()
print("资料与工具（沿用单元12/15，一个字没换）：")
print("  向量库资料条数：", kb, "（.chroma_unit15/）")
print("  FAQ 库行数：", rows, "（.sqlite_unit15/）")
print("  注册表里的工具：", "、".join(registry.names()))
print("  规划器白名单：", "、".join(sorted(PLANNER_ALLOWED)))
print("  计划上限：", MAX_PLAN_STEPS, "步；重规划上限：", MAX_REPLANS, "次")
print("  执行状态：只活在内存里（TaskPlan 不落库）")
print()
print(render_flow())
print()
print(render_question_list())
print()
print(render_concepts())
'''

MINIMAL_FLOW = '''
registry, _model, kb, rows = build_stack()
model = build_plan_model(provider=os.environ.get("LLM_PROVIDER", "deepseek"))
print("目标：", GOAL_MINIMAL)
run = solve(GOAL_MINIMAL, registry, model)
print(render_result(run))
'''

MULTI_FLOW = '''
registry, _model, kb, rows = build_stack()
model = build_plan_model(provider=os.environ.get("LLM_PROVIDER", "deepseek"))
print("目标：", GOAL_MULTI)
run = solve(GOAL_MULTI, registry, model)
print(render_result(run))
'''

STEPS_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("multi", registry=registry, inject=True)
run = scenario.run
print("第3节要看的就是这张表：每一步的输入字段与输出字段")
for item in run.steps:
    print(f"第{item.step}步 {item.tool}")
    print("  输入参数：", item.arguments)
    print("  返回字段：", sorted(item.observation))
    print("  关键内容：", item.preview(120))
    print("  耗时ms：", item.elapsed_ms)
print()
print(render_trajectory(run))
'''

REPLAN_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("replan", registry=registry, inject=True)
run = scenario.run
print("故障注入：把计划里的 question_id 改成 BAD（只在运行时改，原始资料与库没动）")
print(render_result(run))
print()
print("重规划前后各有一版计划（上面「第1轮计划 / 第2轮计划」），改的是哪一步看得见。")
'''

ABORT_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("abort", registry=registry, inject=True)
run = scenario.run
print("故障注入：修订后的计划仍然是 BAD → 终止，不重试第三次")
print(render_result(run))
print()
print("状态：", run.status)
print("答复：", repr(run.answer))
print("是否算完成：", run.finished)
print("模型调用次数：", run.model_calls, "（1 次规划 + 1 次重规划，总结那一次没发生）")
'''

MISSING_FLOW = '''
registry, _model, kb, rows = build_stack()
scenario = run_scenario("missing", registry=registry, inject=True)
run = scenario.run
print("编号格式合法（EC-FAQ-99）但库里没有：只是「未找到」，不是失败")
print(render_result(run))
print()
print("重规划次数：", run.replans, "（应为 0）")
'''

INVALID_FLOW = '''
registry, _model, kb, rows = build_stack()
print(render_invalid(run_invalid(registry)))
print()
print("守门在问模型之前：目标不合格就一次规划都不发，工具更不会动。")
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
    ("一、准备：资料、工具与流程", PREPARE_FLOW, False),
    ("二、第2节：最小场景（计划由 DeepSeek 自己拆）", MINIMAL_FLOW, True),
    ("三、第3节：多步骤场景（三步计划 + 逐步执行）", MULTI_FLOW, True),
    ("四、第3节：每步的输入字段与输出字段", STEPS_FLOW, False),
    ("五、第4节：失败 → 重规划一次 → 跑完", REPLAN_FLOW, False),
    ("六、第4节：修订后仍失败 → 终止并报错", ABORT_FLOW, False),
    ("七、第4节：编号合法但查不到 → 未找到（不算失败）", MISSING_FLOW, False),
    ("八、第4节：无效输入被守门拦下", INVALID_FLOW, False),
    ("九、四条完成标准逐条验收", STANDARDS_FLOW, False),
]

BOUNDARIES = [
    ("目标为空 / 纯空白 / 非字符串 / 过短 / 超长", "validate_goal 拦在问模型之前，一次规划都不发"),
    ("模型返回的不是 JSON", "直接报错「规划结果不是约定的 JSON」，不把任意文本当计划"),
    ("计划步数不在 1~3", "报错；上限约束调用成本，也防止超长计划"),
    ("工具不在白名单", "报错；模型只能挑注册表里已有的名字"),
    ("参数类型不对（含 bool）", "报错；JSON 的 7 与 '7' 放行，格式交给执行时的正则"),
    ("工具执行失败（格式 / 资料里没这个天数）", "记一条失败 → 重规划一次 → 从第 1 步重跑"),
    ("重规划后仍失败", "PlanAborted：状态 aborted、答复为空、不调总结模型"),
    ("编号格式合法但库里没有", "返回「未找到」，是正常观察，不触发重规划"),
    ("已成功步骤的观察", "重规划后仍在列表里（审计证据），不会被冲掉"),
    ("执行状态落盘", "不落：任务卡明写只在本次运行内保存，与单元19 的长期记忆相反"),
]

MANUAL = [
    "正常运行（计划 + 工具结果 + 最终答复）与失败→重规划两处的**截图**。",
    "口头讲解：计划为什么是数据不是指令、重规划为什么只允许一次、"
    "为什么「没解决」必须报错而不能落成完成。",
    "共享进度表登记（本单元与单元19 同一天，合并记录）。",
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
    lines.append("# 单元20 验收证据：Plan-and-Solve 规划（先出计划，再逐步执行）")
    lines.append("")
    lines.append(
        "本文件由 `scripts/generate_evidence_unit20.py` 抓取真实运行输出生成，"
        "步数、观察内容、重规划次数与终止原因都来自实际执行，不手写。"
    )
    lines.append("")
    lines.append("- 单元：20 Plan-and-Solve（4 课时）")
    lines.append("- 本组选题：电商售后服务智能体（资料与工具沿用单元12/15）")
    lines.append(f"- 第二、三节的计划由 `{provider_real}` 自己拆；其余段落用离线剧本模型（可复现）")
    lines.append("- 第五、六、七节是**故障注入**：强制用剧本，真实模型会把 BAD 编号自己改对")
    lines.append("- 密钥：只在 `.env` 里，输出统一 `sk-` 掩码")
    lines.append("")

    for title, code, real in SECTIONS:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("```")
        lines.append(run_code(code, provider_real if real else "mock", root))
        lines.append("```")
        lines.append("")
        if title.startswith("六、"):
            lines.append(
                "这条是「未解决的失败不会被误报为完成」的直接证据：状态不是完成、"
                "答复是空的、连最后一次总结调用都没发生——宁可报错，也不给一个"
                "看起来答完了的结果。"
            )
            lines.append("")
        if title.startswith("七、"):
            lines.append(
                "任务卡的故障提示里专门点明了这一条：有效格式但**不存在**的编号只会"
                "返回「未找到」，不能当成异常分支——本组没有工单，对应的就是 FAQ 编号。"
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
