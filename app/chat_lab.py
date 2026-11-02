"""单元22 的编排层：装本组工具 → 三角色群聊 → 按终止条件收尾 → 逐条验收。

任务卡的四节课在本文件里串成一条：

    第1节  看懂群聊协作：角色表、消息流、凭什么停（`/group roles`）
    第2节  定协作任务与终止规则，先跑最小业务场景（`/group` 不带参数）
    第3节  看消息流与每条的输入输出，补齐异常提示（`/group tight`）
    第4节  验证轮次上限、成本上限、复核不通过（`/group budget|reject|bad`）

    数据侧（一个字没换）：本组 3md + 1pdf + 1csv → Chroma，FAQ CSV → SQLite
    工具侧：app/toolkit.py（单元15 注册表）**保留不动**，本单元只"读"它
    新   增：app/chat_crew.py（群聊 + 三角色 + 管理器）、本文件（编排）

关于 AutoGen：任务卡允许"AutoGen 或等价轻量实现"，本组选后者——理由写在
`app/chat_crew.py` 文件头（不为一次演示引入 pyautogen 这一整套依赖）。

    uv run --extra embedding --extra chroma --extra langchain python -m app.chat_lab
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.chat_crew import (
    DEFAULT_ROUNDS,
    DEFAULT_TOKEN_BUDGET,
    MAX_ROUNDS,
    STATUS_BUDGET,
    STATUS_DELIVERED,
    STATUS_MAX_ROUND,
    STATUS_REJECTED,
    STOP_BUDGET,
    STOP_MAX_ROUND,
    STOP_REJECTED,
    STOP_REVIEWED,
    ChatScript,
    Collaboration,
    build_chat_model,
    render_agents,
    render_concepts,
    render_flow,
    render_result,
    role_name,
    run_collaboration,
)
from app.plan_lab import INVALID_CASES, _shown
from app.tool_lab import build_kit

# ---------------------------------------------------------------- ① 本组业务任务

# 第2节的协作任务：一句话里既有结构化编号、又有流程问题，天然要三个角色接力。
GOAL_GROUP = (
    "买家 2026年12月1日签收的普通商品要退货：查 EC-FAQ-01 的标准答复，"
    "并检索退货规则里对材料和时限的要求；说明哪些结论有证据、哪些需要转人工"
)
# 证据不足场景：编号格式合法，但本组库里没有（沿用单元20/21 那条"未找到不算失败"）。
GOAL_GAP = "查一下 EC-FAQ-99 的标准答复，并说明处理时要关注什么"

# ---------------------------------------------------------------- ② 离线剧本

PLAN_TEXT = (
    "（离线剧本）规划分三步：1) 已由平台工具取回 EC-FAQ-01 的标准答复，执行员直接引用，"
    "不得改写；2) 再按 DOC-02-01 的规则片段说明材料与时限，天数照抄原文；"
    "3) 结尾分开写「有证据的结论」与「需转人工的事项」，不承诺退款结果。"
)
ANSWER_TEXT = (
    "（离线剧本）结论：EC-FAQ-01 的标准答复是——课程样例中签收后 7 天内且未使用可申请"
    "退货评估，最终由人工复核（来源 DOC-02-01）。\n"
    "依据：DOC-02-01-普通商品退货规则.md 写明普通商品签收后 7 天内且未使用可申请退货评估；"
    "售后受理材料清单要求先确认订单编号、商品名称、签收时间和售后诉求。\n"
    "需转人工：是否最终通过、超时限或争议件，本组资料未给出判定标准，不做推断。"
)
ANSWER_GAP = (
    "（离线剧本）EC-FAQ-99 在本组资料库中没有对应记录（工具返回「未找到」），"
    "检索也没有命中可引用的规则片段；资料不足，建议转人工核实后再答复，不猜测答复内容。"
)
REVIEW_PASS = {
    "passed": True,
    "reason": "答复只用了 EC-FAQ-01 与 DOC-02-01 的事实，引用编号与文件名齐全，并把判定权交给人工",
}
REVIEW_FAIL = {
    "passed": False,
    "reason": "答复写了「7 天内一定通过退款」，依据里没有这条结论；退款结果只能由人工复核，属于编造承诺",
}
REVIEW_PASS_GAP = {
    "passed": True,
    "reason": "答复如实说明未找到并建议转人工，没有编造答复内容",
}

# 多轮往复用：第二版计划只补一句"按复核意见删掉承诺"，第二版草稿真的删掉。
PLAN_TEXT_AGAIN = (
    "（离线剧本）按复核意见修订计划：保留 EC-FAQ-01 与 DOC-02-01 的事实引用，"
    "删掉任何关于退款结果的承诺，结尾只写「是否通过由人工复核」。"
)
ANSWER_FIXED = (
    "（离线剧本）结论：EC-FAQ-01 的标准答复是签收后 7 天内且未使用可申请退货评估，"
    "是否通过由人工复核（来源 DOC-02-01）。\n"
    "依据：DOC-02-01-普通商品退货规则.md；材料清单见 DOC-02-U12-01。\n"
    "需转人工：是否最终通过、超时限或争议件，资料未给判定标准，不做推断、不承诺退款。"
)

# 成本演示值：估算 token 的预算。故意设得比"发起者 + 规划员 + 执行员"略小，
# 这样第二次发言完就撞上限——停下来的是钱，不是轮次（max_round 还没满）。
BUDGET_SMALL = 900


# ---------------------------------------------------------------- ③ 场景表


@dataclass(frozen=True)
class ScenarioSpec:
    """一个演示场景：跑几次、花多少钱、用哪份剧本、要不要故障注入。"""

    key: str
    label: str
    goal: str
    rounds: int
    token_budget: int
    script: ChatScript
    inject: bool
    why: str


SCENARIOS: dict[str, ScenarioSpec] = {
    "deliver": ScenarioSpec(
        "deliver",
        "正常协作：三位依次发言 → 复核通过 → 交付",
        GOAL_GROUP,
        DEFAULT_ROUNDS,
        DEFAULT_TOKEN_BUDGET,
        ChatScript([PLAN_TEXT], [ANSWER_TEXT], [REVIEW_PASS]),
        False,
        "最完整的一次协作：3 个角色各发言一次，复核通过才把草稿变成答复",
    ),
    "revise": ScenarioSpec(
        "revise",
        "复核不通过 → 群聊继续 → 再改一版 → 二次复核通过",
        GOAL_GROUP,
        6,
        DEFAULT_TOKEN_BUDGET,
        ChatScript(
            [PLAN_TEXT, PLAN_TEXT_AGAIN],
            [ANSWER_TEXT, ANSWER_FIXED],
            [REVIEW_FAIL, REVIEW_PASS],
        ),
        True,
        "群聊不是一次就收尾：不通过就再来一轮，直到有人通过或轮次用完",
    ),
    "tight": ScenarioSpec(
        "tight",
        "只给 2 轮：复核角色还没轮到 → 未交付",
        GOAL_GROUP,
        2,
        DEFAULT_TOKEN_BUDGET,
        ChatScript([PLAN_TEXT], [ANSWER_TEXT], []),
        True,
        "轮次上限生效：消息条数到 max_round 就停，不偷偷补一轮给复核",
    ),
    "budget": ScenarioSpec(
        "budget",
        "成本上限：预算花完就停 → 未交付",
        GOAL_GROUP,
        DEFAULT_ROUNDS,
        BUDGET_SMALL,
        ChatScript([PLAN_TEXT], [ANSWER_TEXT], []),
        True,
        "成本上限独立于轮次：max_round 还没满，但估算 token 已超预算",
    ),
    "reject": ScenarioSpec(
        "reject",
        "复核不通过：草稿不交付 → 转人工",
        GOAL_GROUP,
        DEFAULT_ROUNDS,
        DEFAULT_TOKEN_BUDGET,
        ChatScript([PLAN_TEXT], [ANSWER_TEXT], [REVIEW_FAIL]),
        True,
        "复核有放行权：它说不通过，答复就留空，草稿只留在消息流里",
    ),
    "gap": ScenarioSpec(
        "gap",
        "证据不足：如实说明 → 复核照样通过",
        GOAL_GAP,
        DEFAULT_ROUNDS,
        DEFAULT_TOKEN_BUDGET,
        ChatScript([PLAN_TEXT], [ANSWER_GAP], [REVIEW_PASS_GAP]),
        True,
        "查不到 ≠ 失败：说清资料不足并建议转人工，评审可以通过",
    ),
}
SCENARIO_ORDER = ("deliver", "revise", "tight", "budget", "reject", "gap")

# 轮次 / 成本守门的用例：全部在让任何角色开口之前被拦下。
GUARD_CASES: list[tuple[str, Any, Any]] = [
    ("轮次 0", 0, DEFAULT_TOKEN_BUDGET),
    ("轮次 7（上限 6）", MAX_ROUNDS + 1, DEFAULT_TOKEN_BUDGET),
    ("轮次不是整数", "3", DEFAULT_TOKEN_BUDGET),
    ("轮次是布尔", True, DEFAULT_TOKEN_BUDGET),
    ("预算 0", DEFAULT_ROUNDS, 0),
    ("预算负数", DEFAULT_ROUNDS, -100),
    ("预算不是整数", DEFAULT_ROUNDS, "900"),
]

CONCEPTS = [
    ("群聊", "多个角色在同一段消息流里轮流发言", "GroupChat：messages 按顺序累积"),
    ("发言顺序", "谁发言由管理器选，不靠模型自觉", "round_robin：按发言条数轮"),
    ("消息流", "后发言的人看得到前面所有人的原话", "render_transcript() 打印全文"),
    ("轮次上限", "消息条数硬上限（含发起者那条）", f"max_round = 轮次 + 1，上限 {MAX_ROUNDS} 轮"),
    ("成本上限", "估算 token 超预算就不再开口", f"演示值 {BUDGET_SMALL}：“发起者+两位”就到顶"),
    ("复核确认", "执行员写草稿，评审员才有放行权", "passed=True 才把草稿变成答复"),
]

QUESTION_LIST = [
    ("几个角色，各自管什么？", "3 个角色 + 发起者；角色表列了职责/输入/输出/不许做什么（第1节）"),
    ("消息怎么看？", "每条消息都留发送者、发言序号、字符数与正文，全文打印（第3节）"),
    ("凭什么停？", "max_round 管轮次、token 预算管成本，两条都是 Python 判断（第4节）"),
    ("谁说了算？", "评审员有放行权：它不发言或不通过，答复就是空的（第2、4节）"),
    ("查不到怎么办？", "如实说资料不足并建议转人工，复核照样可以通过（第4节）"),
    ("为什么不用 AutoGen 本体？", "任务卡允许等价实现；chat_crew.py 文件头写了取舍（第1节）"),
]


# ---------------------------------------------------------------- ④ 数据结构


@dataclass
class ScenarioResult:
    """一个场景的产出：场景定义、协作结果、异常。"""

    key: str
    label: str
    goal: str
    spec: ScenarioSpec
    collaboration: Collaboration | None = None
    error: str = ""


@dataclass
class GroupLabRun:
    """一次编排的全部产出；渲染与验收只读这份数据，不重跑。"""

    scenarios: list[ScenarioResult] = field(default_factory=list)
    invalid_rows: list[tuple[str, str]] = field(default_factory=list)
    guard_rows: list[tuple[str, str]] = field(default_factory=list)
    kb_records: int = 0
    faq_rows: int = 0
    model_name: str = ""
    provider: str = "mock"
    key_configured: bool = False
    standards: list[tuple[str, str, str]] = field(default_factory=list)


# ---------------------------------------------------------------- ⑤ 装配


def build_stack(
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
    model=None,
):
    """准备好"能跑一次群聊"的两件套：单元15 的注册表 + 模型。

    资料与工具全部沿用单元15（`tool_lab.build_kit` 那一套）：本单元换的是**协作方式**，
    不是数据口径。
    """
    _agent, registry, _store, kb_count, faq_rows = build_kit(database_path, faq_db_path)
    chat_model = model if model is not None else build_chat_model()
    return registry, chat_model, kb_count, faq_rows


def _scenario_model(spec: ScenarioSpec, model=None):
    """挑模型：故障注入场景一律用剧本；其余按 `LLM_PROVIDER` 走。

    为什么 tight/budget/reject/gap 要注入：真实模型不会故意少说一轮、也不会自己写
    一份不合格的草稿，那"轮次用完""复核不通过"就演示不到——而且每次运行的数字都不一样，
    证据没法复核。
    """
    if spec.inject:
        return build_chat_model(spec.script, provider="mock")
    if model is not None:
        return model
    return build_chat_model(spec.script)


# ---------------------------------------------------------------- ⑥ 运行


def run_scenario(
    key: str,
    goal: str | None = None,
    registry=None,
    model=None,
    inject: bool | None = None,
) -> ScenarioResult:
    """跑一个场景；注册表可复用（同一批资料，只换任务与剧本）。"""
    spec = SCENARIOS[key]
    if inject is not None and inject != spec.inject:
        spec = ScenarioSpec(
            spec.key, spec.label, spec.goal, spec.rounds, spec.token_budget,
            spec.script, inject, spec.why,
        )
    result = ScenarioResult(key=key, label=spec.label, goal=goal or spec.goal, spec=spec)
    if registry is None:
        registry, _model, _kb, _rows = build_stack()
    try:
        result.collaboration = run_collaboration(
            _scenario_model(spec, model), registry, result.goal,
            rounds=spec.rounds, token_budget=spec.token_budget,
        )
    except (ValueError, RuntimeError) as error:
        result.error = f"{type(error).__name__}：{error}"
    return result


def run_invalid(registry=None) -> list[tuple[str, str]]:
    """任务守门：五种坏任务分别撞一次，全部停在让任何角色开口之前。"""
    if registry is None:
        registry, _model, _kb, _rows = build_stack()
    rows: list[tuple[str, str]] = []
    for label, raw in INVALID_CASES:
        model = build_chat_model(ChatScript([PLAN_TEXT], [ANSWER_TEXT], [REVIEW_PASS]), "mock")
        try:
            run_collaboration(model, registry, raw)
            rows.append((f"{label}（{_shown(raw)}）", "竟然跑起来了（异常！）"))
        except ValueError as error:
            rows.append((f"{label}（{_shown(raw)}）", f"已拒绝：{error}；模型调用 {model.calls} 次"))
    return rows


def run_guards(registry=None) -> list[tuple[str, str]]:
    """轮次与成本守门：越界的参数一律当场拒绝，且一次模型调用都不发。"""
    if registry is None:
        registry, _model, _kb, _rows = build_stack()
    rows: list[tuple[str, str]] = []
    for label, rounds, budget in GUARD_CASES:
        model = build_chat_model(ChatScript([PLAN_TEXT], [ANSWER_TEXT], [REVIEW_PASS]), "mock")
        try:
            run_collaboration(model, registry, GOAL_GROUP, rounds=rounds, token_budget=budget)
            rows.append((label, "竟然跑起来了（异常！）"))
        except ValueError as error:
            rows.append((label, f"已拒绝：{error}；模型调用 {model.calls} 次"))
    return rows


def run_class(
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
    model=None,
    offline: bool = False,
) -> GroupLabRun:
    """按任务卡顺序跑完四节课：正常交付 → 轮次上限 → 成本上限 → 复核不通过 → 证据不足。

    `offline=True` 时五个场景全用剧本（测试要可复现时用；真实调用不该出现在单测里）。
    """
    registry, _chat_model, kb_count, faq_rows = build_stack(database_path, faq_db_path, model)
    run = GroupLabRun(
        kb_records=kb_count,
        faq_rows=faq_rows,
        provider=(os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower(),
        key_configured=bool(os.getenv("DEEPSEEK_API_KEY")),
    )
    for key in SCENARIO_ORDER:
        result = run_scenario(
            key, registry=registry, model=model, inject=True if offline else None
        )
        run.scenarios.append(result)
        if result.collaboration is not None and not run.model_name:
            run.model_name = result.collaboration.model_name
    run.invalid_rows = run_invalid(registry)
    run.guard_rows = run_guards(registry)
    run.standards = check_standards(run)
    return run


def check_standards(run: GroupLabRun) -> list[tuple[str, str, str]]:
    """四条完成标准逐条给结论；结论只从跑出来的数据里取，不手写数字。"""
    by_key = {result.key: result for result in run.scenarios}
    deliver = by_key.get("deliver")
    revise = by_key.get("revise")
    tight = by_key.get("tight")
    budget = by_key.get("budget")
    reject = by_key.get("reject")
    gap = by_key.get("gap")

    rows: list[tuple[str, str, str]] = []

    # 标准一：至少两个角色参与。
    if deliver and deliver.collaboration:
        collab = deliver.collaboration
        names = [role_name(role) for role in collab.participants]
        ok = len(collab.participants) >= 2
        rows.append((
            "至少两个角色参与",
            f"正常协作里实际发言的角色 {len(collab.participants)} 个（{'、'.join(names)}）；"
            f"角色表另有发起者（不发言、不调模型）",
            "通过" if ok else "不通过",
        ))
    else:
        rows.append(("至少两个角色参与", "正常协作场景未跑起来", "不通过"))

    # 标准二：消息内容可查看——每条都有发送者与非空正文，且全文可打印。
    if deliver and deliver.collaboration:
        collab = deliver.collaboration
        readable = all(message.sender and message.content.strip() for message in collab.messages)
        rows.append((
            "消息内容可查看",
            f"消息 {len(collab.messages)} 条，每条都留发送者/发言序号/字符数/正文；"
            f"共 {sum(message.chars for message in collab.messages)} 字，"
            f"全文打印 {len(render_result(collab).splitlines())} 行",
            "通过" if readable else "不通过",
        ))
    else:
        rows.append(("消息内容可查看", "正常协作场景未跑起来", "不通过"))

    # 标准三：轮次和成本受控——两条上限各自生效，且都不在半路多问一句。
    tight_ok = bool(
        tight and tight.collaboration
        and tight.collaboration.stop_reason == STOP_MAX_ROUND
        and len(tight.collaboration.messages) == tight.collaboration.max_round
        and not tight.collaboration.reviewed
    )
    budget_ok = bool(
        budget and budget.collaboration
        and budget.collaboration.stop_reason == STOP_BUDGET
        and budget.collaboration.tokens_used > budget.collaboration.token_budget
        and len(budget.collaboration.messages) < budget.collaboration.max_round
    )
    calls_ok = bool(
        deliver and deliver.collaboration
        and deliver.collaboration.model_calls == len(deliver.collaboration.agent_messages)
        and deliver.collaboration.manager_model_calls == 0
    )
    rows.append((
        "轮次和成本受控",
        (
            f"轮次：申请 {tight.collaboration.rounds_requested} 轮 → 消息 {len(tight.collaboration.messages)}"
            f" 条 = max_round {tight.collaboration.max_round}，复核未轮到"
            if tight and tight.collaboration else "轮次场景未跑起来"
        ) + "；" + (
            f"成本：{budget.collaboration.tokens_used} token > 预算 "
            f"{budget.collaboration.token_budget}，消息 {len(budget.collaboration.messages)} 条 <"
            f" max_round {budget.collaboration.max_round}"
            if budget and budget.collaboration else "成本场景未跑起来"
        ) + "；" + (
            f"模型调用 {deliver.collaboration.model_calls} 次 = 发言条数，管理器 0 次"
            if deliver and deliver.collaboration else "调用数未取到"
        ),
        "通过" if (tight_ok and budget_ok and calls_ok) else "不通过",
    ))

    # 标准四：结果由复核角色确认——通过才交付；不通过/没轮到一律不交付。
    deliver_ok = bool(
        deliver and deliver.collaboration
        and deliver.collaboration.stop_reason == STOP_REVIEWED
        and deliver.collaboration.delivered
        and deliver.collaboration.review.get("passed") is True
    )
    reject_ok = bool(
        reject and reject.collaboration
        and reject.collaboration.stop_reason == STOP_REJECTED
        and not reject.collaboration.answer
        and reject.collaboration.review.get("passed") is False
    )
    rows.append((
        "结果由复核角色确认",
        (
            f"通过：状态「{deliver.collaboration.status}」，答复取自执行员最后一版发言"
            if deliver and deliver.collaboration else "交付场景未跑起来"
        ) + "；" + (
            f"不通过：状态「{reject.collaboration.status}」，答复{'空' if not reject.collaboration.answer else '非空'}，"
            f"草稿仍留在消息流第 {len(reject.collaboration.agent_messages)} 条"
            if reject and reject.collaboration else "复核不通过场景未跑起来"
        ) + "；" + (
            f"复核没轮到：状态「{tight.collaboration.status}」"
            if tight and tight.collaboration else "轮次场景未跑起来"
        ),
        "通过" if (deliver_ok and reject_ok and tight_ok) else "不通过",
    ))

    # 附带一：多轮往复也受轮次约束——第二版草稿才是答复，且消息条数不超过 max_round。
    if revise and revise.collaboration:
        collab = revise.collaboration
        drafts = [m for m in collab.agent_messages if m.sender == "executor"]
        ok = (
            collab.stop_reason == STOP_REVIEWED
            and len(drafts) == 2
            and collab.answer == drafts[-1].content
            and len(collab.messages) <= collab.max_round
            and len(collab.review_history) == 2
        )
        rows.append((
            "多轮往复也受轮次约束",
            f"复核 {len(collab.review_history)} 次（第1次不通过 → 第2次通过）、草稿 {len(drafts)} 版；"
            f"消息 {len(collab.messages)} 条（上限 {collab.max_round}）、模型调用 "
            f"{collab.model_calls} 次；答复取自第 {len(drafts)} 版草稿",
            "通过" if ok else "不通过",
        ))

    # 附带二：证据不足时如实说明，不编造（复核照样可以通过）。
    if gap and gap.collaboration:
        collab = gap.collaboration
        rows.append((
            "证据不足 ≠ 失败（如实说明）",
            f"工具返回「未找到」；状态「{collab.status}」；"
            f"答复首句：{' '.join(collab.answer.split())[:40] or '（空）'}",
            "通过" if collab.delivered and "未找到" in collab.answer else "不通过",
        ))
    return rows


# ---------------------------------------------------------------- ⑦ 渲染


def render_header(run: GroupLabRun) -> str:
    provider = "DeepSeek 真实调用" if run.provider == "deepseek" else "离线脚本模式"
    return "\n".join([
        "单元22 AutoGen 多智能体实战（电商售后服务智能体项目组）",
        f"模型：{run.model_name or '—'}（{provider}；密钥已配置：{run.key_configured}）",
        f"资料：知识库 {run.kb_records} 条；FAQ 库 {run.faq_rows} 行（沿用单元15，未改动）",
        f"协作：3 个角色 + 发起者；轮次上限 {MAX_ROUNDS} 轮，成本按估算 token 计"
        f"（{DEFAULT_TOKEN_BUDGET} 为常规预算，{BUDGET_SMALL} 为成本演示值）",
    ])


def render_scenario(result: ScenarioResult) -> str:
    title = f"【{result.label}】"
    if result.spec.inject:
        title += "（故障注入：发言与复核结论用剧本，真实模型不受影响）"
    if result.collaboration is None:
        return f"{title}\n任务：{result.goal}\n未运行：{result.error}"
    return "\n".join([
        title,
        f"为什么这么设计：{result.spec.why}",
        "",
        render_result(result.collaboration),
    ])


def render_summary_table(run: GroupLabRun) -> str:
    lines = [
        "六条路径横向对照",
        "场景\t轮次\tmax_round\t消息\t发言\t参与角色\ttoken/预算\t模型调用\t终止原因\t交付",
    ]
    for result in run.scenarios:
        collab = result.collaboration
        if collab is None:
            lines.append(f"{result.label}\t—\t—\t—\t—\t—\t—\t—\t未运行\t{result.error}")
            continue
        lines.append(
            f"{result.label}\t{collab.rounds_requested}\t{collab.max_round}\t"
            f"{len(collab.messages)}\t{len(collab.agent_messages)}\t"
            f"{len(collab.participants)}\t{collab.tokens_used}/{collab.token_budget}\t"
            f"{collab.model_calls}\t{collab.stop_reason}\t{'是' if collab.delivered else '否'}"
        )
    return "\n".join(lines)


def render_state_table(run: GroupLabRun) -> str:
    """终止原因 → 状态的对照：四种停法四种状态，没有"笼统的结束"。"""
    lines = ["终止原因与状态（四种停法分开写）", "场景\t终止原因\t状态\t答复"]
    for result in run.scenarios:
        collab = result.collaboration
        if collab is None:
            continue
        lines.append(
            f"{result.label}\t{collab.stop_reason}\t{collab.status}\t"
            f"{'有' if collab.answer else '空'}"
        )
    return "\n".join(lines)


def render_standards(run: GroupLabRun) -> str:
    lines = ["完成标准逐条验收", "标准\t实测\t结论"]
    for name, evidence, verdict in run.standards:
        lines.append(f"{name}\t{evidence}\t{verdict}")
    return "\n".join(lines)


def render_invalid(rows: list[tuple[str, str]]) -> str:
    lines = ["无效任务（守门在让任何角色开口之前）", "输入\t结果"]
    lines.extend(f"{label}\t{verdict}" for label, verdict in rows)
    return "\n".join(lines)


def render_guards(rows: list[tuple[str, str]]) -> str:
    lines = ["轮次与成本守门（越界当场拒绝，模型调用 0 次）", "参数\t结果"]
    lines.extend(f"{label}\t{verdict}" for label, verdict in rows)
    return "\n".join(lines)


def render_question_list() -> str:
    lines = ["第1节要交的项目问题清单", "问题\t在哪一节验证"]
    lines.extend(f"{question}\t{where}" for question, where in QUESTION_LIST)
    return "\n".join(lines)


def render_concept_table() -> str:
    lines = ["概念记录", "概念\t含义\t在本项目里落在哪"]
    lines.extend(f"{name}\t{meaning}\t{here}" for name, meaning, here in CONCEPTS)
    lines.append("")
    lines.append(render_concepts())
    return "\n".join(lines)


STATUS_LABELS = {
    STOP_REVIEWED: STATUS_DELIVERED,
    STOP_REJECTED: STATUS_REJECTED,
    STOP_MAX_ROUND: STATUS_MAX_ROUND,
    STOP_BUDGET: STATUS_BUDGET,
}


def _main() -> None:
    run = run_class()
    print(render_header(run))
    print()
    print(render_flow())
    print()
    print(render_agents())
    print()
    print(render_question_list())
    print()
    for result in run.scenarios:
        print(render_scenario(result))
        print()
    print(render_summary_table(run))
    print()
    print(render_state_table(run))
    print()
    print(render_standards(run))
    print()
    print(render_invalid(run.invalid_rows))
    print()
    print(render_guards(run.guard_rows))
    print()
    print(render_concept_table())
    print()
    print("执行员写的是草稿：复核角色不发言或不通过，答复就是空的，绝不「看起来答完了」。")
    print("轮次与成本都是 Python 判断：停下来的时候，状态一定是「未交付」。")


if __name__ == "__main__":
    _main()
