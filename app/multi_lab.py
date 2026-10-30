"""单元21 的编排层：装本组工具 → 五角色协作 → 评审回路 → 交付或转人工。

按任务卡的四节课把命令串成一条：

    第1节  看懂多智能体架构：角色、共享黑板、协调者、评审回路
    第2节  给本组选一个多步骤任务，定下五个角色与交接顺序，跑出第一版结果
    第3节  实现规划、执行、证据核验、答复、评审五个角色（规划/执行沿用单元20）
    第4节  演示正常交付；再演示评审拒绝 → 修订一次 → 通过，以及二次拒绝 →
           **停止自动交付**并提示人工复核

    数据侧（一个字没换）：本组 3md + 1pdf + 1csv → Chroma，FAQ CSV → SQLite
    工具侧：app/toolkit.py（单元15 注册表）**保留不动**
    规划/执行：app/plan_solver.py（单元20）**保留不动**，只新增 compose_answer=False 开关
    新   增：app/multi_agent.py（五角色 + 黑板 + 协调者）、本文件（编排）

    任务卡的故障提示用的是"工单编号"，本组没有工单，对应的是 **FAQ 编号**：
    `BAD` 走格式校验 → 重规划 → 仍失败 → 执行阶段终止（连草稿都没有，评审不启动）。

    uv run --extra embedding --extra chroma --extra langchain python -m app.multi_lab
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.multi_agent import (
    MAX_REVISIONS,
    ROLE_EVIDENCE,
    ROLE_EXECUTOR,
    ROLE_PLANNER,
    ROLE_RESPONDER,
    ROLE_REVIEWER,
    STATUS_ABORTED,
    STATUS_DELIVERED,
    STATUS_HUMAN,
    Blackboard,
    CrewScript,
    ScriptedCrewModel,
    coordinate,
    render_concepts,
    render_flow,
    render_handoff,
    render_result,
    render_roles,
)
from app.plan_lab import (
    GOAL_MINIMAL,
    GOAL_MULTI,
    INVALID_CASES,
    STEPS_BAD,
    STEPS_MINIMAL,
    STEPS_MISSING,
    STEPS_MULTI,
    _shown,
)
from app.plan_solver import build_plan_model
from app.tool_lab import build_kit

# ---------------------------------------------------------------- ① 本组业务目标

# 第2节的多步骤任务：查标准答复 → 检索退货规则 → 按资料时限算截止日期（正好 3 步）。
GOAL_DELIVER = GOAL_MULTI
# 第4节的评审场景：同一个业务问题，看评审那一环怎么卡住草稿。
GOAL_REVIEW = "买家 2026年12月1日签收的普通商品要退货，请给出处理建议并说明依据"
# 证据不足场景：编号格式合法，但库里没有（沿用单元20 那条"未找到不算失败"的口径）。
GOAL_GAP = "查一下 EC-FAQ-99 的标准答复，并说明处理时要关注什么"
# 执行终止场景：计划里的编号被改成 BAD（相当于任务卡说的"在测试副本里改"）。
GOAL_ABORT = GOAL_MINIMAL

# ---------------------------------------------------------------- ② 离线剧本

# 答复智能体的三版草稿（离线脚本模式下不生成新事实，只演示评审卡的是什么）。
DRAFT_OK = (
    "（离线脚本模式）EC-FAQ-01 标准答复：普通商品签收后 7 天内可申请退货；"
    "依据 DOC-02-01-普通商品退货规则.md，2026-12-01 签收的退货评估截止日期为 2026-12-08。"
)
DRAFT_FLAWED = "普通商品可以在签收后 7 天内申请退货，请尽快提交申请。"
DRAFT_FIXED = (
    "依据 DOC-02-01-普通商品退货规则.md：2026-12-01 签收，7 天时限内申请退货，"
    "评估截止 2026-12-08；材料要求见 EC-FAQ-01 标准答复。资料未写明的时限不做推断。"
)
DRAFT_STILL_BAD = "已补充说明：退货时限以平台资料为准，请提交材料后等待审核。"
DRAFT_GAP = (
    "EC-FAQ-99 在本组资料库里没有对应记录（工具观察返回「未找到」），资料不足，"
    "建议转人工核实后再答复。"
)

# 评审智能体的结论序列：按调用顺序取，这样才能演"第一次不通过 → 修订 → 通过"。
REVIEW_PASS = {"passed": True, "reason": "答复只在观察范围内作答，并注明了依据文件与编号"}
REVIEW_FAIL_SOURCE = {
    "passed": False,
    "reason": "答复给出了 7 天时限，却没有引用任何依据文件，无法核对来源",
}
REVIEW_FAIL_AGAIN = {
    "passed": False,
    "reason": "修订后仍未引用具体依据文件与编号，事实依旧无法核对",
}
REVIEW_PASS_GAP = {
    "passed": True,
    "reason": "答复如实说明资料不足并建议转人工，没有编造观察之外的事实",
}

# 场景表：key → （标签，目标，计划剧本，修订计划，草稿序列，评审序列，是否故障注入）。
# 故障注入 = 强制用离线剧本：真实模型会把 BAD 编号改对、也不会故意写出不合格的草稿，
# 那就演示不到"评审拒绝 → 修订 → 停止交付"这三条了。
SCENARIOS: dict[str, tuple[str, str, list[dict], list[dict] | None, list[str], list[dict], bool]] = {
    "deliver": ("正常协作 → 评审通过 → 交付", GOAL_DELIVER, STEPS_MULTI, None,
                [DRAFT_OK], [REVIEW_PASS], False),
    "revise": ("评审拒绝 → 修订一次 → 通过", GOAL_REVIEW, STEPS_MINIMAL, None,
               [DRAFT_FLAWED, DRAFT_FIXED], [REVIEW_FAIL_SOURCE, REVIEW_PASS], True),
    "reject": ("两次都不通过 → 停止自动交付", GOAL_REVIEW, STEPS_MINIMAL, None,
               [DRAFT_FLAWED, DRAFT_STILL_BAD], [REVIEW_FAIL_SOURCE, REVIEW_FAIL_AGAIN], True),
    "gap": ("证据不足 → 如实说明 → 转人工提示", GOAL_GAP, STEPS_MISSING, None,
            [DRAFT_GAP], [REVIEW_PASS_GAP], True),
    "abort": ("执行阶段终止 → 连草稿都没有", GOAL_ABORT, STEPS_BAD, None,
              [], [], True),
}
SCENARIO_ORDER = ("deliver", "revise", "reject", "gap", "abort")
# 故障注入的场景（渲染时要标注，免得被当成真实模型的表现）。
INJECT_KEYS = tuple(key for key, item in SCENARIOS.items() if item[6])

# 第1节要交的概念记录与项目问题清单。
CONCEPTS = [
    ("智能体", "有明确输入输出的一个角色", "本单元五个：规划 / 执行 / 证据 / 答复 / 评审"),
    ("协调者", "调度与放行判断，自己不调模型", "coordinate()：谁先谁后、能不能交付由它定"),
    ("共享黑板", "角色之间唯一的交接面", "Blackboard.post()，每条都留发送者"),
    ("受控修订", "评审不通过只许改一次", f"MAX_REVISIONS = {MAX_REVISIONS}"),
    ("停止自动交付", "修订后仍不通过就转人工", "答复留空，草稿与意见留在黑板上"),
]

QUESTION_LIST = [
    ("几个智能体，各自管什么？", "5 个专职角色 + 协调者；角色表列了职责/输入/输出/不许做什么（第1节）"),
    ("它们怎么交接？", "共享黑板：每条消息都有发送者、内容与说明，可逐条追溯（第2、3节）"),
    ("答复谁写的、谁放行的？", "答复智能体只写草稿，评审智能体独立判，协调者才交付（第3节）"),
    ("评审不通过怎么办？", "把理由交回答复智能体，只修订一次（第4节）"),
    ("修订后还是不通过呢？", "停止自动交付：答复留空、状态转人工、草稿与意见保留（第4节）"),
    ("执行阶段就失败了呢？", "连草稿都没有，证据/答复/评审三个角色根本不启动（第4节）"),
]


# ---------------------------------------------------------------- ③ 数据结构


@dataclass
class CrewScenario:
    """一个场景的产出：目标、黑板、异常。"""

    key: str
    label: str
    goal: str
    board: Blackboard | None = None
    error: str = ""


@dataclass
class CrewLabRun:
    """一次编排的全部产出；渲染与验收只从这份数据里取，不重跑。"""

    scenarios: list[CrewScenario] = field(default_factory=list)
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
    """准备好"能跑一次协作"的两件套：单元15 的注册表 + 模型。

    工具与资料**全部沿用单元15**（`tool_lab.build_kit` 建的那一套）：本单元换的是
    "谁来干活、谁来复核"，不是换数据口径。
    """
    _agent, registry, _store, kb_count, faq_rows = build_kit(database_path, faq_db_path)
    chat_model = model if model is not None else build_crew_model()
    return registry, chat_model, kb_count, faq_rows


def build_crew_model(script: CrewScript | None = None, provider: str | None = None):
    """按环境变量挑模型：`deepseek` 走真实 ChatDeepSeek，`mock` 走离线剧本。

    真实模式刻意不做兜底：缺包或缺密钥就当场报错，让课堂上看见原因，而不是
    悄悄换成离线答案（单元05 静默降级那个坑代价很大）。
    """
    chosen = (provider or os.getenv("LLM_PROVIDER", "mock")).strip().lower()
    if chosen == "deepseek":
        return build_plan_model(provider="deepseek")  # 复用单元16 的工厂：同一份密钥
    if chosen != "mock":
        raise ValueError("LLM_PROVIDER 只能是 mock 或 deepseek")
    script = script or CrewScript()
    return ScriptedCrewModel(
        steps=list(script.steps),
        revised=list(script.revised) if script.revised is not None else None,
        drafts=list(script.drafts),
        reviews=list(script.reviews),
    )


def _script_of(key: str) -> CrewScript:
    _label, _goal, steps, revised, drafts, reviews, _inject = SCENARIOS[key]
    return CrewScript(steps=list(steps), revised=list(revised) if revised else None,
                      drafts=list(drafts), reviews=list(reviews))


def _scenario_model(key: str, model=None, inject: bool = False):
    """挑模型：`inject=True` 一律用剧本；否则真实模式共用 ChatDeepSeek。"""
    if inject:
        return build_crew_model(_script_of(key), provider="mock")
    if model is not None:
        return model
    return build_crew_model(_script_of(key))


# ---------------------------------------------------------------- ⑤ 运行


def run_scenario(
    key: str,
    goal: str | None = None,
    registry=None,
    model=None,
    max_revisions: int = MAX_REVISIONS,
    inject: bool | None = None,
) -> CrewScenario:
    """跑一个场景；注册表可复用（同一批资料，只换目标与剧本）。

    `inject` 不传时按场景表里写好的走；测试里要稳定复现就显式传 `inject=True`，
    免得 `.env` 把它变成真实调用。
    """
    label, default_goal, _steps, _revised, _drafts, _reviews, default_inject = SCENARIOS[key]
    use_inject = default_inject if inject is None else inject
    scenario = CrewScenario(key=key, label=label, goal=goal or default_goal)
    if registry is None:
        registry, _model, _kb, _rows = build_stack()
    try:
        scenario.board = coordinate(
            _scenario_model(key, model, use_inject), registry, scenario.goal,
            max_revisions=max_revisions,
        )
    except (ValueError, RuntimeError) as error:
        scenario.error = f"{type(error).__name__}：{error}"
    return scenario


def run_invalid(registry=None) -> list[tuple[str, str]]:
    """第4节：五种坏目标分别撞一次守门，全部停在问任何角色之前。"""
    if registry is None:
        registry, _model, _kb, _rows = build_stack()
    rows: list[tuple[str, str]] = []
    for label, raw in INVALID_CASES:
        model = build_crew_model(_script_of("deliver"), provider="mock")
        try:
            coordinate(model, registry, raw)
            rows.append((f"{label}（{_shown(raw)}）", "竟然跑起来了（异常！）"))
        except ValueError as error:
            rows.append((
                f"{label}（{_shown(raw)}）",
                f"已拒绝：{error}；模型调用 {model.calls} 次",
            ))
    return rows


def run_class(
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
    model=None,
    offline: bool = False,
) -> CrewLabRun:
    """按任务卡顺序跑完四节课：正常交付 → 修订 → 停止交付 → 证据不足 → 执行终止。

    `offline=True` 时五个场景全用剧本（测试要可复现时用；真实调用不该出现在单测里）。
    """
    registry, _chat_model, kb_count, faq_rows = build_stack(database_path, faq_db_path, model)
    run = CrewLabRun(
        kb_records=kb_count,
        faq_rows=faq_rows,
        provider=(os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower(),
        key_configured=bool(os.getenv("DEEPSEEK_API_KEY")),
    )
    for key in SCENARIO_ORDER:
        scenario = run_scenario(
            key, registry=registry, model=model, inject=True if offline else None
        )
        run.scenarios.append(scenario)
        if scenario.board is not None and not run.model_name:
            run.model_name = scenario.board.model_name
    run.invalid_rows = run_invalid(registry)
    run.standards = check_standards(run)
    return run


def check_standards(run: CrewLabRun) -> list[tuple[str, str, str]]:
    """三条完成标准逐条给结论；结论只从跑出来的数据里取，不手写。"""
    by_key = {scenario.key: scenario for scenario in run.scenarios}
    deliver = by_key.get("deliver")
    revise = by_key.get("revise")
    reject = by_key.get("reject")
    abort = by_key.get("abort")
    gap = by_key.get("gap")

    rows: list[tuple[str, str, str]] = []

    # 标准一：至少 3 个智能体有明确输入输出与职责——角色表 + 实际发过言的角色。
    speaking = set()
    if deliver and deliver.board:
        speaking = {m.sender for m in deliver.board.messages}
    needed = {ROLE_PLANNER, ROLE_EXECUTOR, ROLE_EVIDENCE, ROLE_RESPONDER, ROLE_REVIEWER}
    ok = needed.issubset(speaking)
    rows.append((
        "至少 3 个智能体有明确输入、输出和职责",
        f"角色表 {len(_role_rows())} 个专职角色 + 协调者；"
        f"正常协作里实际发过言的角色 {len(speaking)} 个："
        f"{'、'.join(sorted(speaking)) or '无'}",
        "通过" if ok else "不通过",
    ))

    # 标准二：共享状态能追溯——黑板条数、发送者、四类产物都在。
    if deliver and deliver.board:
        board = deliver.board
        has_all = bool(board.plan) and bool(board.observations) and bool(board.evidence) \
            and bool(board.drafts) and bool(board.review_history)
        rows.append((
            "共享状态能追溯",
            f"黑板 {len(board.messages)} 条交接（每条都有发送者）；"
            f"计划 {len(board.plan)} 步 / 观察 {len(board.observations)} 条 / "
            f"草稿 {len(board.drafts)} 版 / 评审 {len(board.review_history)} 次",
            "通过" if has_all else "不通过",
        ))
    else:
        rows.append(("共享状态能追溯", "正常协作场景未跑起来", "不通过"))

    # 标准三：评审未通过时有受控修订——revise 场景恰好修订 1 次后交付。
    if revise and revise.board:
        board = revise.board
        ok = (
            board.revisions == 1
            and len(board.drafts) == 2
            and len(board.review_history) == 2
            and board.delivered
            and board.status == STATUS_DELIVERED
        )
        rows.append((
            "评审未通过时有受控修订",
            f"第1次评审：{'通过' if board.review_history[0].get('passed') else '不通过'} → "
            f"修订 {board.revisions} 次（上限 {MAX_REVISIONS}）→ "
            f"第2次评审：{'通过' if len(board.review_history) > 1 and board.review_history[1].get('passed') else '不通过'}；"
            f"状态：{board.status}",
            "通过" if ok else "不通过",
        ))
    else:
        rows.append(("评审未通过时有受控修订", "修订场景未跑起来", "不通过"))

    # 标准三的另一半：仍未通过不得自动交付。
    if reject and reject.board:
        board = reject.board
        ok = (
            not board.delivered
            and not board.answer
            and board.status == STATUS_HUMAN
            and len(board.review_history) == 2
            and board.revisions == MAX_REVISIONS
        )
        rows.append((
            "仍未通过不得自动交付",
            f"评审 {len(board.review_history)} 次均未通过；答复：{board.answer or '（空）'}；"
            f"状态：{board.status}；草稿 {len(board.drafts)} 版留在黑板上供人工复核",
            "通过" if ok else "不通过",
        ))
    else:
        rows.append(("仍未通过不得自动交付", "二次拒绝场景未跑起来", "不通过"))

    # 附带一：证据不足时如实说明，不编造。
    if gap and gap.board:
        board = gap.board
        rows.append((
            "证据不足 ≠ 失败（如实说明）",
            f"证据核验 sufficient={board.evidence_sufficient}；"
            f"缺：{board.evidence.get('missing', '—')}；状态：{board.status}",
            "通过" if not board.evidence_sufficient else "不通过",
        ))

    # 附带二：执行阶段终止时，后面三个角色根本不启动。
    if abort and abort.board:
        board = abort.board
        ok = (
            board.status == STATUS_ABORTED
            and not board.drafts
            and not board.review_history
            and not board.answer
        )
        rows.append((
            "执行阶段终止 → 不进入评审",
            f"状态：{board.status}；草稿 {len(board.drafts)} 版、评审 "
            f"{len(board.review_history)} 次、答复：{board.answer or '（空）'}；"
            f"异常：{board.error[:60]}",
            "通过" if ok else "不通过",
        ))
    return rows


def _role_rows() -> list[Any]:
    from app.multi_agent import ROLES

    return list(ROLES)


# ---------------------------------------------------------------- ⑥ 渲染


def render_header(run: CrewLabRun) -> str:
    provider = "DeepSeek 真实调用" if run.provider == "deepseek" else "离线脚本模式"
    return "\n".join([
        "单元21 多智能体系统架构（电商售后服务智能体项目组）",
        f"模型：{run.model_name or '—'}（{provider}；密钥已配置：{run.key_configured}）",
        f"资料：知识库 {run.kb_records} 条；FAQ 库 {run.faq_rows} 行（沿用单元15，未改动）",
        f"工具：单元15 的三项只读工具；评审后最多修订 {MAX_REVISIONS} 次",
    ])


def render_scenario(scenario: CrewScenario) -> str:
    title = f"【{scenario.label}】"
    if scenario.key in INJECT_KEYS:
        title += "（故障注入：草稿与评审结论用剧本，真实模型不受影响）"
    if scenario.board is None:
        return f"{title}\n目标：{scenario.goal}\n未运行：{scenario.error}"
    return "\n".join([
        f"{title}目标：{scenario.goal}",
        "",
        render_result(scenario.board),
        "",
        render_handoff(scenario.board),
    ])


def render_summary_table(run: CrewLabRun) -> str:
    lines = ["五条路径横向对照", "场景\t目标\t计划步数\t观察\t证据\t草稿版数\t评审次数\t修订\t状态\t交付"]
    for scenario in run.scenarios:
        if scenario.board is None:
            lines.append(f"{scenario.label}\t{scenario.goal[:16]}…\t—\t—\t—\t—\t—\t—\t未运行\t{scenario.error}")
            continue
        board = scenario.board
        lines.append(
            f"{scenario.label}\t{scenario.goal[:16]}…\t{len(board.plan)}\t"
            f"{len(board.observations)}\t{'充分' if board.evidence_sufficient else '不足'}\t"
            f"{len(board.drafts)}\t{len(board.review_history)}\t{board.revisions}\t"
            f"{board.status}\t{'是' if board.delivered else '否'}"
        )
    return "\n".join(lines)


def render_standards(run: CrewLabRun) -> str:
    lines = ["完成标准逐条验收", "标准\t实测\t结论"]
    for name, evidence, verdict in run.standards:
        lines.append(f"{name}\t{evidence}\t{verdict}")
    return "\n".join(lines)


def render_invalid(rows: list[tuple[str, str]]) -> str:
    lines = ["无效输入（守门在问任何角色之前）", "输入\t结果"]
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


def _main() -> None:
    run = run_class()
    print(render_header(run))
    print()
    print(render_flow())
    print()
    print(render_roles())
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
    print("答复智能体只有草稿，评审智能体才有放行权：草稿写完要等评审结论。")
    print("二次不通过就停：答复留空、状态转人工，草稿与评审意见留在黑板上给人工复核。")


if __name__ == "__main__":
    _main()
