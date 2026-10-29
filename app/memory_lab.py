"""单元19 的编排层：装本组工具与执行器 → 接三种记忆 → 四节课场景 → 逐条验收。

分工与前面几个单元一样，本文件只负责把任务卡的四节课串成一条命令：

    第1节  分清对话记忆库与业务知识库，画出本组的读—答—写流程
    第2节  用本组业务对象（EC-FAQ-01）跑两轮连续问答，验证「刚才那项」能定位
    第3节  三种记忆模式横向对比，并验证长期记忆在"重启"后仍能召回
    第4节  会话隔离 + 旧状态重新查工具，按四条完成标准逐条给结论

    数据侧（一个字没换）：本组 3md + 1pdf + 1csv → Chroma，FAQ CSV → SQLite
    工具侧：app/toolkit.py（单元15 注册表）**保留不动**
    执行器：app/lc_agent.py（单元17）**保留不动**，本单元给它多传一份历史消息
    新   增：app/memory_chain.py（三种记忆、会话隔离、指代解析）

    uv run --extra embedding --extra chroma --extra langchain python -m app.memory_lab
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.chroma_store import resolve_unit19_memory_db_path
from app.lc_agent import (
    DEFAULT_MAX_STEPS,
    ScriptedToolModel,
    answer_step,
    build_agent_model,
    build_executor,
    tool_step,
)
from app.memory_chain import (
    DEFAULT_MODE,
    DEFAULT_SESSION,
    MODE_BUFFER,
    MODE_HYBRID,
    MODE_SUMMARY,
    MODE_VECTOR,
    MODE_LABELS,
    AgentMemoryManager,
    TurnRecord,
    render_memory_flow,
    render_modes,
    render_snapshot,
    recalled_lines,
    render_turn,
    render_turns,
    run_turn,
)
from app.tool_lab import build_kit
from app.toolkit import TOOL_FAQ

# 第2节：本组真实业务对象——FAQ 编号是这三条里唯一能"被指代"的东西。
TURN1_QUESTION = "EC-FAQ-01 的标准答复是什么？"
# 第2轮故意不写编号，只说「刚才那项」：能不能接上，全看记忆。
TURN2_QUESTION = "刚才那项，买家要在签收后多久内申请退货评估？"
TURN3_QUESTION = "我刚才问的是哪个编号？"
MULTI_QUESTIONS = (TURN1_QUESTION, TURN2_QUESTION, TURN3_QUESTION)
# 三种模式对比只跑前两轮：一轮问编号、一轮问"刚才那项"，够看出差异。
PAIR_QUESTIONS = (TURN1_QUESTION, TURN2_QUESTION)

# 第4节：两个会话各跑一轮，证明互相拿不到对方的东西。
SESSION_A = "student-a"
SESSION_B = "student-b"

# 离线剧本（LLM_PROVIDER=mock 时用）。第2轮照旧调用工具——这就是"业务状态重新查工具"
# 这条标准在离线模式下也能演示的原因；第3轮刻意不调工具，用来看记忆里剩下什么。
PLAN_MULTI: list[dict] = [
    tool_step(TOOL_FAQ, question_id="EC-FAQ-01"),
    {},  # 第1轮答复：由工具原文摘录
    tool_step(TOOL_FAQ, question_id="EC-FAQ-01"),
    {},  # 第2轮答复：同样来自工具，不来自记忆
    answer_step(
        "（离线脚本模式）本轮没有调用工具：编号 EC-FAQ-01 来自本会话的对话记忆；"
        "涉及答复内容时仍须重新查工具，记忆只解决「哪一项」，不解决「答案是什么」。"
    ),
]

# 第1节要交的概念记录：几个容易混的词在本项目里分别指什么。
CONCEPTS = [
    ("对话记忆", "本轮之外、过去轮次说过的话", "只作语境；不是证据，不能当资料引用"),
    ("业务知识库", ".chroma_unit15/ + .sqlite_unit15/", "事实的唯一来源，回答必须落到这里"),
    ("长期记忆", ".chroma_unit19_memory/，按会话隔离", "跨运行仍在，用于跨轮指代与偏好"),
    ("短期历史",  "内存里的完整消息或摘要", "进程一退出就没了，这是它与长期记忆的分界"),
    ("读答写", "load → run_agent → save", "顺序不能倒：先回忆再作答，答完才落记忆"),
    ("会话隔离", "键前缀 + 库端 source 过滤", "两个学生同时问，谁也看不到谁的对话"),
]

# 第1节要交的项目问题清单：每条对应本单元的一个验证场景。
QUESTION_LIST = [
    ("「刚才那项」到底指什么？",
     f"指代解析落到具体编号并打印（第2节「{TURN2_QUESTION}」）"),
    ("两个学生同时用会不会串话？",
     f"会话 {SESSION_A} 与 {SESSION_B} 各自隔离，互不可见（第4节）"),
    ("程序关掉再开还记得吗？",
     "长期记忆落盘，新建管理器仍可召回；完整历史与摘要则消失（第3节）"),
    ("记过的东西能不能直接当答案？",
     "不能：第2轮仍然调用工具现查，记忆只提供对象（第4节）"),
]


@dataclass
class MemoryLabRun:
    """一次编排的全部产出；渲染与验收都只从这份数据里取，不重跑。"""

    mode: str
    session_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    mode_compare: list[TurnRecord] = field(default_factory=list)
    isolated: list[TurnRecord] = field(default_factory=list)
    restart_recall: list[dict] = field(default_factory=list)
    restart_count: int = 0
    kb_count: int = 0
    faq_rows: int = 0
    model_name: str = ""
    summary_model: str = ""
    standards: list[tuple[str, str, str]] = field(default_factory=list)
    # 渲染要读管理器里的摘要原文与各会话计数，一并带出来，避免为了显示再跑一遍。
    managers: dict = field(default_factory=dict, repr=False)
    manager: Any = field(default=None, repr=False)
    iso_manager: Any = field(default=None, repr=False)


def build_stack(
    model=None,
    max_steps: int = DEFAULT_MAX_STEPS,
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
):
    """准备好"能跑多轮问答"的三件套：执行器 + 注册表 + 资料库。"""
    _agent, registry, store, kb_count, faq_rows = build_kit(database_path, faq_db_path)
    decision = model if model is not None else build_agent_model(PLAN_MULTI)
    executor = build_executor(registry, decision, max_steps)
    return executor, registry, store, kb_count, faq_rows


def build_memory_manager(
    model=None,
    database_path: Path | None = None,
    summary_model=None,
) -> AgentMemoryManager:
    """造一个记忆管理器；摘要要不要调模型单独给，便于离线演示。"""
    return AgentMemoryManager(
        model=summary_model,
        database_path=database_path or resolve_unit19_memory_db_path(),
    )


def run_session(
    executor,
    manager: AgentMemoryManager,
    session_id: str = DEFAULT_SESSION,
    questions=PAIR_QUESTIONS,
    mode: str = DEFAULT_MODE,
    max_steps: int | None = None,
) -> list[TurnRecord]:
    """连着问几轮；记忆在 `run_turn` 内部按读—答—写的顺序自己流转。"""
    records: list[TurnRecord] = []
    for question in questions:
        records.append(run_turn(executor, manager, session_id, question, mode, max_steps))
    return records


def run_mode_compare(
    executor,
    questions=PAIR_QUESTIONS,
    database_path: Path | None = None,
    registry=None,
):
    """第3节：三种模式各跑同样的两轮，各自用自己的管理器（互不影响）。"""
    results: list[TurnRecord] = []
    for mode in (MODE_BUFFER, MODE_SUMMARY, MODE_VECTOR):
        manager = build_memory_manager(database_path=database_path)
        # 长期记忆是落盘的，不清就会把上一次演示的轮次一起召回来。
        manager.clear_session(f"mode-{mode}")
        current = executor
        if registry is not None and isinstance(executor.model, ScriptedToolModel):
            # 离线剧本模型的 plan 是"按调用次数往下走"的，主会话已经把它走完了。
            # 每个模式换一份新剧本，三种模式的工具行为才可比（真实模型不需要这步）。
            current = build_executor(registry, build_agent_model(PLAN_MULTI), executor.max_steps)
        results.extend(run_session(current, manager, f"mode-{mode}", questions, mode))
    return results


def run_isolation(executor, database_path: Path | None = None):
    """第4节：两个会话各问一次，第二个会话应该什么都回忆不到。"""
    manager = build_memory_manager(database_path=database_path)
    for session in (SESSION_A, SESSION_B):
        manager.clear_session(session)
    first = run_turn(executor, manager, SESSION_A, TURN1_QUESTION, MODE_HYBRID)
    # 换一个会话问同一句：库里已经有 A 的记忆了，B 若还能看到就是串话。
    second = run_turn(executor, manager, SESSION_B, TURN1_QUESTION, MODE_HYBRID)
    return [first, second], manager


def recall_after_restart(
    database_path: Path | None = None,
    session_id: str = SESSION_A,
    question: str = TURN2_QUESTION,
) -> tuple[int, list[dict]]:
    """模拟"程序重启"：只重建管理器（内存清空），看长期记忆还在不在。

    完整历史与摘要挂在这一步上消失，长期记忆因为落盘还能被召回——
    这个差异就是第3节要讲的，两种记忆不是强弱关系，是"存活范围"不同。
    """
    manager = build_memory_manager(database_path=database_path)
    return manager.long_term_count(session_id), manager.vector.recall(session_id, question)


def check_standards(run: MemoryLabRun) -> list[tuple[str, str, str]]:
    """第4节：四条完成标准逐条给结论，只依据已经跑出来的记录，不重新跑。"""
    second = run.turns[1] if len(run.turns) > 1 else None
    resolution = second.snapshot.resolution if second else None

    resolved_ok = resolution is not None and resolution.object_id == "EC-FAQ-01"
    rows: list[tuple[str, str, str]] = [
        (
            "能解析跨轮指代",
            (
                f"第2轮「{TURN2_QUESTION}」→ {resolution.object_id}"
                f"（来自第 {resolution.from_turn} 轮{resolution.source}，关键词「{resolution.keyword}」）"
                if resolution
                else "第2轮没有解析出对象"
            ),
            "通过" if resolved_ok else "未通过",
        ),
    ]

    if len(run.isolated) >= 2:
        other = run.isolated[1]
        clean = len(other.snapshot.history) == 0 and len(other.snapshot.recalled) == 0
        rows.append((
            "不同会话不串话",
            f"会话 {other.session_id} 带回来的历史 {len(other.snapshot.history)} 条、"
            f"长期记忆召回 {len(other.snapshot.recalled)} 条",
            "通过" if clean else "未通过",
        ))

    rows.append((
        "长期记忆重启后可召回",
        f"新建管理器后仍读到 {run.restart_count} 条、本次召回 {len(run.restart_recall)} 条",
        "通过" if run.restart_count > 0 and run.restart_recall else "未通过",
    ))

    requery = bool(second and second.tools)
    rows.append((
        "业务状态重新查工具",
        f"第2轮调用的工具：{'、'.join(second.tools) if second and second.tools else '（无）'}",
        "通过" if requery else "未通过",
    ))
    return rows


# ---------------- 渲染 ----------------

def _executor_model_name(executor) -> str:
    return type(executor.model).__name__


def render_header(executor, kb_count: int, faq_rows: int, manager: AgentMemoryManager) -> str:
    lines = [
        "=== 单元19：对话记忆系统（完整历史 / 摘要 / 向量长期记忆）===",
        f"资料：{kb_count} 条（.chroma_unit15/）；FAQ 库：{faq_rows} 行（.sqlite_unit15/）",
        f"执行器模型：{_executor_model_name(executor)}；摘要引擎：{manager.summary_engine.engine_name}",
        "长期记忆库：.chroma_unit19_memory/（与知识库严格分开）",
    ]
    return "\n".join(lines)


def render_concepts() -> str:
    lines = ["第1节 概念记录（本组的两类库与读答写顺序）：", "概念\t在本项目里是什么\t边界"]
    for name, meaning, boundary in CONCEPTS:
        lines.append(f"{name}\t{meaning}\t{boundary}")
    lines.append("")
    lines.append("第1节 本组要验证的项目问题：")
    for question, evidence in QUESTION_LIST:
        lines.append(f"- {question} → {evidence}")
    return "\n".join(lines)


def render_mode_compare(records: list[TurnRecord], managers: dict) -> str:
    """第3节横向对照：同一组问题，三种模式各自带进模型了什么。"""
    lines = [
        "三种模式横向对比（同一组问题各跑两轮，第2轮都问「刚才那项」）：",
        "模式\t第2轮带进模型的对话\t第2轮长期记忆召回\t第2轮是否解析出编号\t第2轮是否重新查工具",
    ]
    by_mode: dict[str, list[TurnRecord]] = {}
    for record in records:
        by_mode.setdefault(record.mode, []).append(record)
    for mode in (MODE_BUFFER, MODE_SUMMARY, MODE_VECTOR):
        turns = by_mode.get(mode, [])
        second = turns[1] if len(turns) > 1 else None
        if second is None:
            lines.append(f"{MODE_LABELS[mode]}\t（未运行）\t-\t-\t-")
            continue
        resolution = second.snapshot.resolution
        lines.append(
            f"{MODE_LABELS[mode]}\t{len(second.snapshot.history)} 条\t"
            f"{len(second.snapshot.recalled)} 条\t"
            f"{resolution.object_id if resolution else '（未解析）'}\t"
            f"{'是（' + '、'.join(second.tools) + '）' if second.tools else '否'}"
        )
    lines.append("")
    lines.append("摘要模式的摘要内容（检查编号有没有被压缩掉）：")
    for mode, manager in managers.items():
        if mode != MODE_SUMMARY:
            continue
        session = f"mode-{mode}"
        lines.append(f"  {session}：{manager.summary(session) or '（空）'}")
    return "\n".join(lines)


def render_restart(count: int, recalled: list[dict]) -> str:
    lines = [
        "第3节 长期记忆跨运行召回（新建管理器 = 内存清空，等价于重启程序）：",
        f"  库里该会话的记忆条数：{count}",
        f"  按第2轮问句召回：{len(recalled)} 条",
    ]
    for text in recalled_lines(recalled):
        lines.append(f"    - {text}")
    lines.append("  对照：完整历史与摘要在这一步会归零（它们只在内存里）。")
    return "\n".join(lines)


def render_isolation(records: list[TurnRecord], manager: AgentMemoryManager) -> str:
    lines = ["第4节 会话隔离（两个会话各问一次）："]
    for record in records:
        lines.append(f"  会话 {record.session_id}｜本轮带回来历史 {len(record.snapshot.history)} 条、"
                     f"长期记忆召回 {len(record.snapshot.recalled)} 条")
    lines.append(
        f"  库里 {SESSION_A} 的记忆 {manager.long_term_count(SESSION_A)} 条、"
        f"{SESSION_B} 的记忆 {manager.long_term_count(SESSION_B)} 条（各自隔离，过滤在库端完成）"
    )
    return "\n".join(lines)


def render_standards(rows: list[tuple[str, str, str]]) -> str:
    lines = ["第4节 完成标准逐条验收：", "标准\t实测\t结论"]
    for name, observed, verdict in rows:
        lines.append(f"{name}\t{observed}\t{verdict}")
    return "\n".join(lines)


def render_run(run: MemoryLabRun) -> str:
    blocks = [
        render_header_plain(run),
        "",
        "【第1节 记忆流程与本组两个库】",
        render_memory_flow(),
        "",
        render_concepts(),
        "",
        "【第2节 两轮连续问答（完整历史 + 长期记忆）】",
        render_turns(run.turns),
        "",
        "  第2轮回忆到的内容：",
    ]
    if len(run.turns) > 1:
        for line in render_snapshot(run.turns[1].snapshot).split("\n"):
            blocks.append(f"    {line}")
    blocks += [
        "",
        "【第3节 三种模式对比】",
        render_modes(),
        "",
        render_mode_compare(run.mode_compare, run.managers),
        "",
        render_restart(run.restart_count, run.restart_recall),
        "",
        "【第4节 会话隔离与完成标准】",
        render_isolation(run.isolated, run.iso_manager),
        "",
        render_standards(run.standards),
    ]
    return "\n".join(blocks)


def render_header_plain(run: MemoryLabRun) -> str:
    return (
        f"=== 单元19：对话记忆系统（模式 {run.mode}｜会话 {run.session_id}）===\n"
        f"资料 {run.kb_count} 条（.chroma_unit15/）；FAQ 库 {run.faq_rows} 行（.sqlite_unit15/）；"
        f"执行器模型 {run.model_name}；摘要引擎 {run.summary_model}"
    )


def run_class(
    mode: str = DEFAULT_MODE,
    database_path: Path | None = None,
    model=None,
) -> MemoryLabRun:
    """按任务卡四节课顺序跑一遍，返回全部产出（渲染与验收都基于它）。"""
    executor, registry, _store, kb_count, faq_rows = build_stack(
        model=model, database_path=database_path
    )
    manager = build_memory_manager(database_path=database_path)
    manager.clear_session(DEFAULT_SESSION)
    turns = run_session(executor, manager, DEFAULT_SESSION, MULTI_QUESTIONS, mode)

    managers: dict[str, AgentMemoryManager] = {}
    mode_compare: list[TurnRecord] = []
    for item_mode in (MODE_BUFFER, MODE_SUMMARY, MODE_VECTOR):
        item_manager = build_memory_manager(database_path=database_path)
        managers[item_mode] = item_manager
        current = executor
        if isinstance(executor.model, ScriptedToolModel):
            # 离线剧本模型按"调用次数"往下走，主会话已经把它走完了；每个模式
            # 换一份新剧本，三者的工具行为才可比（真实模型不需要这一步）。
            current = build_executor(registry, build_agent_model(PLAN_MULTI), executor.max_steps)
        mode_compare.extend(
            run_session(current, item_manager, f"mode-{item_mode}", PAIR_QUESTIONS, item_mode)
        )

    # 隔离环节也换一份新剧本：否则离线模式下 plan 早已走完，两个会话都变成"没调工具"，
    # 记忆里存的内容就看不出"这一轮到底查了什么"。
    isolated, iso_manager = run_isolation(
        build_executor(registry, build_agent_model(PLAN_MULTI), executor.max_steps)
        if isinstance(executor.model, ScriptedToolModel)
        else executor,
        database_path=database_path,
    )
    run = MemoryLabRun(
        mode=mode,
        session_id=DEFAULT_SESSION,
        turns=turns,
        mode_compare=mode_compare,
        isolated=isolated,
        kb_count=kb_count,
        faq_rows=faq_rows,
        model_name=_executor_model_name(executor),
        summary_model=manager.summary_engine.engine_name,
        managers=managers,
        manager=manager,
        iso_manager=iso_manager,
    )
    # 重启召回放在最后：它只是读，不往任何会话里写东西。
    run.restart_count, run.restart_recall = recall_after_restart(database_path)
    run.standards = check_standards(run)
    return run


if __name__ == "__main__":  # pragma: no cover - 手动跑演示时用
    print(render_run(run_class()))


__all__ = [
    "CONCEPTS",
    "MULTI_QUESTIONS",
    "PAIR_QUESTIONS",
    "PLAN_MULTI",
    "QUESTION_LIST",
    "SESSION_A",
    "SESSION_B",
    "TURN1_QUESTION",
    "TURN2_QUESTION",
    "TURN3_QUESTION",
    "MemoryLabRun",
    "build_memory_manager",
    "build_stack",
    "check_standards",
    "recall_after_restart",
    "render_concepts",
    "render_isolation",
    "render_mode_compare",
    "render_modes",
    "render_restart",
    "render_run",
    "render_standards",
    "run_class",
    "run_isolation",
    "run_mode_compare",
    "run_session",
]
