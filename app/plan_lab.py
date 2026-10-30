"""单元20 的编排层：装本组工具 → 出一版计划 → 逐步执行 → 失败重规划 → 终止报错。

分工与前面几个单元一样，本文件只负责把任务卡的四节课串成一条命令：

    第1节  分清"计划 / 执行 / 观察 / 重规划"，画出本组的 Plan-and-Solve 流程
    第2节  选一个多步骤业务目标，先跑最小场景（单步），核对计划与真实工具结果
    第3节  跑多步计划，逐条看工具调用与观察结果
    第4节  在**测试副本**里把计划里的编号改成 BAD → 看格式校验与重规划；
           再跑一次"修订后仍失败"→ 终止并报错；最后确认 EC-FAQ-99（格式合法但
           库里没有）只返回"未找到"，不触发重规划

    数据侧（一个字没换）：本组 3md + 1pdf + 1csv → Chroma，FAQ CSV → SQLite
    工具侧：app/toolkit.py（单元15 注册表）**保留不动**
    新   增：app/plan_solver.py（规划器 + 执行器 + 重规划 + 终止）

    任务卡的故障提示用的是"工单编号"，本组没有工单，对应的是 **FAQ 编号**：
    把计划里的 `question_id` 改成 BAD 就是同一件事（格式校验拒绝 → 重规划）。
    "测试副本"在这里的含义是：**只在运行时替换计划参数，不改动任何原始业务数据**，
    跑完也不用"撤销"——原始 CSV 与库从头到尾没被动过。

    uv run --extra embedding --extra chroma --extra langchain python -m app.plan_lab
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.plan_solver import (
    MAX_PLAN_STEPS,
    MAX_REPLANS,
    STATUS_ABORTED,
    STATUS_DONE,
    PlanRun,
    ScriptedPlanModel,
    build_plan_model,
    plan_step,
    render_concepts,
    render_result,
    solve,
)
from app.tool_lab import build_kit
from app.toolkit import TOOL_DEADLINE, TOOL_FAQ, TOOL_SEARCH

# ---------------------------------------------------------------- ① 本组业务目标

# 第2节的最小场景：一步就能答完，用来核对"计划 ↔ 真实工具结果"对不对得上。
GOAL_MINIMAL = "查一下 EC-FAQ-01 的标准答复，并说明处理时要关注什么"
# 第3节的多步骤目标：编号查答复 → 检索规则原文 → 按原文算截止日期（正好 3 步上限）。
GOAL_MULTI = (
    "买家 2026年12月1日签收的普通商品要退货：先查 EC-FAQ-01 的标准答复，"
    "再检索退货规则里对材料和时限的要求，最后按资料写明的时限算出退货评估截止日期"
)
# 第4节的故障场景：目标照旧，只是计划里的编号被改成 BAD（格式校验拒绝）。
GOAL_FAIL = GOAL_MINIMAL
GOAL_MISSING = "查一下 EC-FAQ-99 的标准答复，并说明处理时要关注什么"

# ---------------------------------------------------------------- ② 离线剧本

# 单步计划：按编号查标准答复（EC-FAQ-01 是库里真实存在的编号）。
STEPS_MINIMAL: list[dict[str, Any]] = [
    plan_step(TOOL_FAQ, "拿到这条问题的标准答复原文", question_id="EC-FAQ-01"),
]
# 三步计划：查答复 → 检索规则原文 → 算截止日期（天数与依据文件都来自本组资料）。
STEPS_MULTI: list[dict[str, Any]] = [
    plan_step(TOOL_FAQ, "先拿到这条问题的标准答复", question_id="EC-FAQ-01"),
    plan_step(TOOL_SEARCH, "再检索退货规则里对材料和时限的要求",
              query="普通商品退货需要哪些材料，时限是多久"),
    plan_step(TOOL_DEADLINE, "按资料写明的时限算出截止日期",
              sign_date="2026-12-01", window_days="7",
              basis_doc="DOC-02-01-普通商品退货规则.md"),
]
# 故障剧本：编号写成 BAD，格式正则（EC-FAQ-\d{2}）直接拒绝。
STEPS_BAD: list[dict[str, Any]] = [
    plan_step(TOOL_FAQ, "拿到这条问题的标准答复原文", question_id="BAD"),
]
# 修订剧本：把编号改回正确值——这是"重规划一次后跑完"的情形。
STEPS_FIXED: list[dict[str, Any]] = [
    plan_step(TOOL_FAQ, "改回正确编号再查一次", question_id="EC-FAQ-01"),
]
# 格式合法但库里没有：只会返回"未找到"，不算失败，不该触发重规划。
STEPS_MISSING: list[dict[str, Any]] = [
    plan_step(TOOL_FAQ, "查一个库里没有的编号", question_id="EC-FAQ-99"),
]

# 场景表：key → （标签，目标，首次计划，修订计划，是否故障注入）。
# 修订计划为 None 表示"原样重发"，用来演示"修订后仍失败则终止"。
# 故障注入 = 强制用离线剧本（相当于任务卡说的"在测试副本里把编号改成 BAD"）：
# 真实模型会自己把编号改对，那就演示不到失败与重规划了。
SCENARIOS: dict[str, tuple[str, str, list[dict], list[dict] | None, bool]] = {
    "minimal": ("最小场景（单步）", GOAL_MINIMAL, STEPS_MINIMAL, None, False),
    "multi": ("多步骤场景（三步）", GOAL_MULTI, STEPS_MULTI, None, False),
    "replan": ("失败 → 重规划一次 → 跑完", GOAL_FAIL, STEPS_BAD, STEPS_FIXED, True),
    "abort": ("修订后仍失败 → 终止", GOAL_FAIL, STEPS_BAD, None, True),
    "missing": ("格式合法但查不到 → 未找到", GOAL_MISSING, STEPS_MISSING, None, True),
}
SCENARIO_ORDER = ("minimal", "multi", "replan", "abort", "missing")
# 故障注入的场景：真实模型会自己把 BAD 改对，这几个场景必须用剧本才演示得到。
INJECT_KEYS = tuple(key for key, item in SCENARIOS.items() if item[4])

# 第4节：守门用例（全部停在问模型之前）。
INVALID_CASES: list[tuple[str, Any]] = [
    ("空目标", ""),
    ("纯空白目标", "   "),
    ("非字符串目标", 123),
    ("过短目标", "查"),
    ("超长目标", "查" * 400),
]

# 第1节要交的概念记录与项目问题清单。
CONCEPTS = [
    ("计划", "1~3 个「工具 + 参数」的步骤", "make_plan() 输出，只描述做什么"),
    ("执行", "按 cursor 逐步调注册表", "execute_plan()，走单元15 的白名单"),
    ("观察", "工具真实返回值", "StepObservation，含耗时与第几轮"),
    ("重规划", "失败后改一次计划", "上限 1 次，改完从第一步重跑"),
    ("终止", "修订后仍失败", "PlanAborted：无答复、不调总结模型"),
]

QUESTION_LIST = [
    ("复杂目标怎么拆？", f"「{GOAL_MULTI[:28]}…」→ {MAX_PLAN_STEPS} 步计划（第3节）"),
    ("计划里的每一步真跑了吗？", "轨迹逐条列工具、参数、真实返回值与耗时（第3节）"),
    ("参数写错了会怎样？", "BAD 编号被格式校验拒绝 → 重规划一次后跑完（第4节）"),
    ("改了一次还错呢？", "终止并报错，答复为空，不冒充完成（第4节）"),
    ("编号合法但库里没有呢？", "只返回「未找到」，是正常观察，不触发重规划（第4节）"),
]


# ---------------------------------------------------------------- ③ 数据结构


@dataclass
class PlanScenario:
    """一个场景的产出：目标、计划、轨迹、答复、状态。"""

    key: str
    label: str
    goal: str
    run: PlanRun | None = None
    error: str = ""


@dataclass
class PlanLabRun:
    """一次编排的全部产出；渲染与验收只从这份数据里取，不重跑。"""

    scenarios: list[PlanScenario] = field(default_factory=list)
    invalid_rows: list[tuple[str, str]] = field(default_factory=list)
    kb_records: int = 0
    faq_rows: int = 0
    model_name: str = ""
    provider: str = "mock"
    key_configured: bool = False
    standards: list[tuple[str, str, str]] = field(default_factory=list)


# ---------------------------------------------------------------- ④ 装配


def build_stack(
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
    model=None,
):
    """准备好"能跑一次规划"的两件套：单元15 的注册表 + 模型。

    工具与资料**全部沿用单元15**（`tool_lab.build_kit` 建的那一套）：本单元换的是
    "先出计划再执行"，不是换数据口径——这句要写进文档，否则证据说不清是哪份资料。
    """
    _agent, registry, _store, kb_count, faq_rows = build_kit(database_path, faq_db_path)
    chat_model = model if model is not None else build_plan_model()
    return registry, chat_model, kb_count, faq_rows


def _scenario_model(
    key: str,
    model=None,
    steps: list[dict] | None = None,
    revised: list[dict] | None = None,
    inject: bool = False,
):
    """挑模型：真实模式用同一个 ChatDeepSeek；离线模式按场景给不同剧本。

    `inject=True` 时一律用剧本（故障注入）：真实模型会把 BAD 编号自己改对，
    那就演示不到"失败 → 重规划"和"修订后仍失败 → 终止"这两条了。
    """
    if inject:
        return ScriptedPlanModel(steps=steps or [], revised=revised)
    if model is not None:
        return model
    provider = (os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower()
    if provider == "deepseek":
        return build_plan_model(provider="deepseek")
    return ScriptedPlanModel(steps=steps or [], revised=revised)


# ---------------------------------------------------------------- ⑤ 运行


def run_scenario(
    key: str,
    goal: str | None = None,
    registry=None,
    model=None,
    max_replans: int = MAX_REPLANS,
    inject: bool | None = None,
) -> PlanScenario:
    """跑一个场景；注册表可复用（同一批资料，只换目标与剧本）。

    `inject` 不传时按场景表里写好的走（故障场景默认注入）；
    测试里要稳定复现就显式传 `inject=True`，免得 `.env` 把它变成真实调用。
    """
    label, default_goal, steps, revised, default_inject = SCENARIOS[key]
    use_inject = default_inject if inject is None else inject
    scenario = PlanScenario(key=key, label=label, goal=goal or default_goal)
    if registry is None:
        registry, _model, _kb, _rows = build_stack()
    try:
        scenario.run = solve(
            scenario.goal,
            registry,
            _scenario_model(key, model, steps, revised, use_inject),
            max_replans=max_replans,
        )
    except (ValueError, RuntimeError) as error:
        scenario.error = f"{type(error).__name__}：{error}"
    return scenario


def run_invalid(registry=None) -> list[tuple[str, str]]:
    """第4节：五种坏目标分别撞一次守门，全部停在问模型之前。"""
    if registry is None:
        registry, _model, _kb, _rows = build_stack()
    rows: list[tuple[str, str]] = []
    for label, raw in INVALID_CASES:
        try:
            solve(raw, registry, _scenario_model("minimal", None, STEPS_MINIMAL, None))
            rows.append((f"{label}（{_shown(raw)}）", "竟然跑起来了（异常！）"))
        except (ValueError, TypeError) as error:
            rows.append((f"{label}（{_shown(raw)}）", f"已拒绝：{error}"))
    return rows


def run_class(
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
    model=None,
    offline: bool = False,
) -> PlanLabRun:
    """按任务卡顺序跑完四节课：最小场景 → 多步骤 → 重规划 → 终止 → 未找到 → 守门。

    `offline=True` 时五个场景全用剧本（测试要可复现时用；真实调用不该出现在
    单测里）。否则故障场景始终注入、其余场景按 `LLM_PROVIDER` 走。
    """
    registry, _chat_model, kb_count, faq_rows = build_stack(database_path, faq_db_path, model)
    run = PlanLabRun(
        kb_records=kb_count,
        faq_rows=faq_rows,
        provider=(os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower(),
        key_configured=bool(os.getenv("DEEPSEEK_API_KEY")),
    )
    # 离线模式下每个场景有自己的剧本，所以不把 build_stack 的模型传下去；
    # 真实模式下 `model` 由调用方给进来，五个场景共用同一个 ChatDeepSeek。
    for key in SCENARIO_ORDER:
        scenario = run_scenario(
            key, registry=registry, model=model, inject=True if offline else None
        )
        run.scenarios.append(scenario)
        if scenario.run is not None and not run.model_name:
            run.model_name = scenario.run.model_name
    run.invalid_rows = run_invalid(registry)
    run.standards = check_standards(run)
    return run


def check_standards(run: PlanLabRun) -> list[tuple[str, str, str]]:
    """四条完成标准逐条给结论；结论只从跑出来的数据里取，不手写。"""
    by_key = {scenario.key: scenario for scenario in run.scenarios}
    minimal = by_key.get("minimal")
    multi = by_key.get("multi")
    replan = by_key.get("replan")
    abort = by_key.get("abort")
    missing = by_key.get("missing")

    rows: list[tuple[str, str, str]] = []

    # 标准一：计划可逐步执行——最小场景的计划步数 = 成功执行步数。
    if minimal and minimal.run and minimal.run.plan:
        planned = len(minimal.run.plan.steps)
        done = len(minimal.run.steps)
        ok = planned == done and minimal.run.status == STATUS_DONE
        rows.append((
            "计划可逐步执行",
            f"计划 {planned} 步 → 实际执行 {done} 步，状态：{minimal.run.status}",
            "通过" if ok else "不通过",
        ))
    else:
        rows.append(("计划可逐步执行", "最小场景未跑起来", "不通过"))

    # 标准二：能查看工具调用与观察结果——多步场景每步都有参数与返回值。
    if multi and multi.run:
        detailed = [
            item for item in multi.run.steps if item.arguments and item.preview(40)
        ]
        ok = len(multi.run.steps) >= 2 and len(detailed) == len(multi.run.steps)
        rows.append((
            "能查看工具调用与观察结果",
            f"{len(multi.run.steps)} 步全部记录参数与返回值；例："
            f"{multi.run.steps[0].tool}({next(iter(multi.run.steps[0].arguments.values()), '')}) "
            f"→ {multi.run.steps[0].preview(60)}" if multi.run.steps else "无步骤",
            "通过" if ok else "不通过",
        ))
    else:
        rows.append(("能查看工具调用与观察结果", "多步骤场景未跑起来", "不通过"))

    # 标准三：失败后至多重规划一次——replan 场景恰好 1 次，abort 场景也只有 1 次。
    counts = {
        key: (scenario.run.replans if scenario.run else -1)
        for key, scenario in by_key.items()
    }
    ok = counts.get("replan") == 1 and counts.get("abort") == 1
    rows.append((
        "失败后至多重规划一次",
        f"重规划次数：replan={counts.get('replan')}、abort={counts.get('abort')}"
        f"（上限 {MAX_REPLANS}）",
        "通过" if ok else "不通过",
    ))

    # 标准四：未解决的失败不会被误报为完成——abort 状态、无答复、不调总结模型。
    if abort and abort.run:
        ok = (
            abort.run.status == STATUS_ABORTED
            and not abort.run.answer
            and not abort.run.finished
        )
        rows.append((
            "未解决的失败不会被误报为完成",
            f"状态：{abort.run.status}；答复：{abort.run.answer or '（空）'}；"
            f"模型调用 {abort.run.model_calls} 次（总结那一次没发生）",
            "通过" if ok else "不通过",
        ))
    else:
        rows.append(("未解决的失败不会被误报为完成", "终止场景未跑起来", "不通过"))

    # 附带一条：格式合法但查不到，只返回"未找到"，不触发重规划。
    if missing and missing.run:
        ok = missing.run.replans == 0 and missing.run.status == STATUS_DONE
        preview = missing.run.steps[0].preview(60) if missing.run.steps else "—"
        rows.append((
            "格式合法但查不到 ≠ 失败",
            f"重规划 {missing.run.replans} 次；观察：{preview}",
            "通过" if ok else "不通过",
        ))
    return rows


def _shown(raw: Any) -> str:
    text = repr(raw)
    return text if len(text) <= 24 else text[:24] + "…"


# ---------------------------------------------------------------- ⑥ 渲染


def render_flow() -> str:
    """第1节要交的流程图：计划 → 执行 → 观察 →（失败）重规划 → 终止。"""
    return "\n".join([
        "Plan-and-Solve 流程（本组电商售后）",
        "",
        "  目标（一句话业务问题）",
        "    │ ① 规划：模型第 1 次调用，只输出 JSON（工具 + 参数 + 理由）",
        "    ▼",
        "  TaskPlan（1~3 步，工具只能在白名单里挑）",
        "    │ ② 执行：Python 按 cursor 推进，走单元15 的注册表",
        "    │     ├ 成功 → 记一条观察（步号 / 参数 / 真实返回值 / 耗时）",
        "    │     └ 失败 → ③ 至多重规划一次（把错误与已有观察交回规划器）",
        "    │             ├ 修订后跑完 → replanned=True，照常出答复",
        "    │             └ 修订后仍失败 → 终止并报错（无答复）",
        "    ▼",
        "  ④ 答复：模型最后一次调用，只许用观察里出现过的事实",
        "",
        "执行状态只在本次运行内保存（TaskPlan 在内存里，不落库）。",
    ])


def render_header(run: PlanLabRun) -> str:
    provider = "DeepSeek 真实调用" if run.provider == "deepseek" else "离线脚本模式"
    return "\n".join([
        "单元20 Plan-and-Solve（电商售后服务智能体项目组）",
        f"模型：{run.model_name or '—'}（{provider}；密钥已配置：{run.key_configured}）",
        f"资料：知识库 {run.kb_records} 条；FAQ 库 {run.faq_rows} 行（沿用单元15，未改动）",
        f"工具：单元15 的三项只读工具；重规划上限 {MAX_REPLANS} 次，计划上限 {MAX_PLAN_STEPS} 步",
    ])


def render_scenario(scenario: PlanScenario) -> str:
    title = f"【{scenario.label}】"
    if scenario.key in INJECT_KEYS:
        title += "（故障注入：强制用剧本计划，真实模型不受影响）"
    if scenario.run is None:
        return f"{title}目标：{scenario.goal}\n未运行：{scenario.error}"
    return f"{title}\n" + render_result(scenario.run)


def render_summary_table(run: PlanLabRun) -> str:
    lines = ["五条路径横向对照", "场景\t目标\t计划步数\t成功步数\t失败次数\t重规划\t状态\t答复"]
    for scenario in run.scenarios:
        if scenario.run is None:
            lines.append(f"{scenario.label}\t{scenario.goal[:20]}…\t—\t—\t—\t—\t未运行\t{scenario.error}")
            continue
        item = scenario.run
        planned = len(item.plan.steps) if item.plan else 0
        lines.append(
            f"{scenario.label}\t{scenario.goal[:20]}…\t{planned}\t{len(item.steps)}\t"
            f"{len(item.failures)}\t{item.replans}\t{item.status}\t"
            f"{'有' if item.answer else '无'}"
        )
    return "\n".join(lines)


def render_standards(run: PlanLabRun) -> str:
    lines = ["完成标准逐条验收", "标准\t实测\t结论"]
    for name, evidence, verdict in run.standards:
        lines.append(f"{name}\t{evidence}\t{verdict}")
    return "\n".join(lines)


def render_invalid(rows: list[tuple[str, str]]) -> str:
    lines = ["无效输入（守门在问模型之前）", "输入\t结果"]
    lines.extend(f"{label}\t{verdict}" for label, verdict in rows)
    return "\n".join(lines)


def render_question_list() -> str:
    lines = ["第1节要交的项目问题清单", "问题\t在哪一节验证"]
    lines.extend(f"{question}\t{where}" for question, where in QUESTION_LIST)
    return "\n".join(lines)


def render_concept_table() -> str:
    lines = ["概念记录", "概念\t在本项目里落在哪"]
    lines.extend(f"{name}\t{here}" for name, _meaning, here in CONCEPTS)
    lines.append("")
    lines.append(render_concepts())
    return "\n".join(lines)


def _main() -> None:
    run = run_class()
    print(render_header(run))
    print()
    print(render_flow())
    print()
    print(render_question_list())
    print()
    for scenario in run.scenarios:
        print(render_scenario(scenario))
        print()
    print(render_summary_table(run))
    print()
    print(render_standards(run))
    print()
    print(render_invalid(run.invalid_rows))
    print()
    print("计划是数据不是指令：模型只能挑白名单里的工具并填参数，路径、SQL、代码一概碰不到。")
    print("没解决的失败就报错：状态 aborted、答复为空、连总结那次模型调用都不发生。")


if __name__ == "__main__":
    _main()
