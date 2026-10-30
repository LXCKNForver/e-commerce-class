"""单元21：多智能体系统架构 —— 五个专职角色 + 共享黑板 + 协调者。

单元20 解决的问题是"先出计划、再逐步执行"，但它仍是一条**单人流水线**：同一个
模型既拆计划、又写答复，写完就交付，没人复核。单元21 把它拆成一支队伍：

    用户目标
      │  协调者（coordinator：本模块的 `coordinate()`，不调模型，只做调度与放行判断）
      ▼
    ① 规划智能体 planner   → 1~3 步 JSON 计划（只提工具与参数，不执行）
      ▼ 写进黑板
    ② 执行智能体 executor  → 逐步调真实工具，产出观察（**不写答复**）
      ▼ 写进黑板
    ③ 证据核验智能体 evidence → {sufficient, reason, missing}（只判断够不够，不管措辞）
      ▼ 写进黑板
    ④ 答复智能体 responder → 一版面向用户的草稿（**没有自行放行的权限**）
      ▼ 写进黑板
    ⑤ 评审智能体 reviewer  → {passed, reason}（独立复核，不改写观察）
      │        ├ 通过 → 协调者把草稿变成答复，交付
      │        └ 不通过 → 把理由交回答复智能体，**只许修订一次**
      │                     ├ 修订后通过 → 交付
      │                     └ 修订后仍不通过 → **停止自动交付**，转人工复核

四件事必须讲清楚（也是完成标准的落点）：

1. **每个角色的输入与输出都是显式的**。规划只看目标，执行只看计划，证据只看观察，
   答复只看观察 + 证据意见（+ 评审意见），评审只看目标 + 观察 + 证据 + 草稿。
   谁都不许越过上一步直接碰原始资料或自己放行。
2. **共享黑板是唯一的交接面**。角色之间不互相调用，全部通过 `Blackboard.post()`
   留痕；每条消息都有发送者、内容与说明，所以"这份答复是怎么来的"能一步步追回去。
   黑板只存**产物**，不存模型内部推理。
3. **评审不通过只有一次受控修订**。反馈交回答复智能体重写一次措辞（不得改工具
   事实），再评一次。
4. **仍未通过就停止自动交付**。答复留空、状态标记为转人工，草稿与评审意见留在
   黑板上给人工复核——"看起来答完了"比"明确停下来"危险得多。

规划与执行两个角色**直接复用单元20**（`make_plan` / `execute_plan`），本单元只给
`execute_plan` 加了一个 `compose_answer=False` 开关：答复不再是执行器的事。

    uv run --extra embedding --extra chroma --extra langchain python -m app.multi_lab
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.plan_solver import (
    FEEDBACK_MARK,
    MAX_REPLANS,
    PLANNER_MARK,
    PlanAborted,
    PlanStep,
    StepFailure,
    StepObservation,
    TaskPlan,
    execute_plan,
    make_plan,
    validate_goal,
)

# ---------------------------------------------------------------- ① 常量

# 五个角色的键：黑板消息、角色表、渲染都用它，不另起一套字符串。
ROLE_PLANNER = "planner"
ROLE_EXECUTOR = "executor"
ROLE_EVIDENCE = "evidence"
ROLE_RESPONDER = "responder"
ROLE_REVIEWER = "reviewer"
# 协调者不是"第六个模型角色"：它不调模型，只做调度、修订计数与是否放行的判断。
ROLE_COORDINATOR = "coordinator"

ROLE_ORDER = (ROLE_PLANNER, ROLE_EXECUTOR, ROLE_EVIDENCE, ROLE_RESPONDER, ROLE_REVIEWER)

# 任务卡原话：评审未通过时**有受控修订**，仍未通过不得自动交付。
# 写成常量：修订次数、渲染、验收三处都读它，不各写一份。
MAX_REVISIONS = 1
# 执行阶段沿用单元20 的规矩：计划失败至多重规划一次。
MAX_PLAN_REPLANS = MAX_REPLANS

# 三种结束状态：交付了、转人工了、执行阶段就终止了（连草稿都没有）。
STATUS_DELIVERED = "已自动交付（评审通过）"
STATUS_HUMAN = "评审未通过，已停止自动交付，转人工复核"
STATUS_ABORTED = "执行阶段终止（未产出草稿，未进入评审）"

HUMAN_REVIEW_NOTE = "自动评审未通过：答复未交付，已保留执行证据与评审意见，请人工复核。"

# 四个角色的提示词标记：离线剧本模型靠它们认出"此刻扮演谁"。
# 规划与答复两个标记直接沿用单元20，剧本模型可以两个单元共用同一套写法。
EVIDENCE_MARK = "证据核验员"
REVIEWER_MARK = "独立评审员"
# 交接数据统一放在这个标记之后：离线剧本模型据此取出 JSON，不必猜哪一段是数据。
PAYLOAD_MARK = "交接数据："

OFFLINE_MODEL_NAME = "scripted-crew-model"
MESSAGE_PREVIEW = 120


class CrewProtocolError(ValueError):
    """某个角色没按约定输出（不是 JSON / 缺字段）：不猜、不降级，直接报错。"""


# ---------------------------------------------------------------- ② 数据结构


@dataclass(frozen=True)
class AgentRole:
    """一个角色的契约：职责、输入、输出、不许做什么。

    任务卡要求"至少 3 个智能体具有明确输入、输出和职责"，这张表就是那个证据；
    不许做什么同样重要——评审不能自己写答复，答复不能自己放行。
    """

    key: str
    name: str
    duty: str
    inputs: str
    outputs: str
    forbidden: str


ROLES: tuple[AgentRole, ...] = (
    AgentRole(
        ROLE_PLANNER,
        "规划智能体",
        "把用户目标拆成 1~3 步可执行的工具调用",
        "用户目标（+ 失败后重规划的反馈）",
        "JSON 计划：每步只有工具名、参数、这一步解决什么",
        "不执行工具、不碰路径/SQL/代码、不写最终答复",
    ),
    AgentRole(
        ROLE_EXECUTOR,
        "执行智能体",
        "按计划逐步调用注册表里的真实工具并记录观察",
        "TaskPlan + 单元15 的 ToolRegistry",
        "逐步观察：第几步、工具、参数、真实返回值、耗时",
        "不改写观察、不自己作答（答复归答复智能体）",
    ),
    AgentRole(
        ROLE_EVIDENCE,
        "证据核验智能体",
        "判断观察是否足以支撑用户所问",
        "用户目标 + 全部工具观察",
        'JSON：{"sufficient": 布尔, "reason": 理由, "missing": 缺什么}',
        "不写答复、不补业务事实、不凭常识放行",
    ),
    AgentRole(
        ROLE_RESPONDER,
        "答复智能体",
        "依据观察写出面向用户的答复草稿",
        "观察 + 证据意见（+ 评审意见，修订时）",
        "一版答复草稿（文本）",
        "不得更改工具事实、**不得自行放行**、证据不足要明说",
    ),
    AgentRole(
        ROLE_REVIEWER,
        "评审智能体",
        "独立复核草稿是否回应问题、忠于证据、说明缺口",
        "目标 + 观察 + 证据意见 + 当前草稿",
        'JSON：{"passed": 布尔, "reason": 理由}',
        "不得改写工具观察、不得代写答复、不因「差不多」放行",
    ),
)

ROLE_NAMES: dict[str, str] = {role.key: role.name for role in ROLES}
ROLE_NAMES[ROLE_COORDINATOR] = "协调者"


@dataclass
class RoleMessage:
    """黑板上的一条留痕：谁发的、发了什么、这次交接是什么。"""

    seq: int
    sender: str
    content: str
    note: str = ""

    @property
    def sender_name(self) -> str:
        return ROLE_NAMES.get(self.sender, self.sender)

    def preview(self, limit: int = MESSAGE_PREVIEW) -> str:
        text = " ".join(self.content.split())
        return text if len(text) <= limit else text[:limit] + "…"


@dataclass
class Blackboard:
    """共享黑板：所有角色只往这里写、只从这里读。

    存的是**产物**（计划 / 观察 / 证据意见 / 草稿 / 评审意见），不是模型内部推理——
    后者看不见也测不了，留痕只留能对账的东西。

    这份状态活在内存里，不落库（与单元19 的长期记忆相反，与单元20 的执行状态相同）：
    一次协作结束，黑板就是这次协作的完整记录。
    """

    task: str = ""
    plan: list[dict[str, Any]] = field(default_factory=list)
    plan_versions: list[list[dict[str, Any]]] = field(default_factory=list)
    execution: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    draft: str = ""
    drafts: list[str] = field(default_factory=list)
    review: dict[str, Any] = field(default_factory=dict)
    review_history: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    status: str = ""
    delivered: bool = False
    revisions: int = 0
    model_calls: int = 0
    elapsed_ms: int = 0
    error: str = ""
    model_name: str = ""
    messages: list[RoleMessage] = field(default_factory=list)

    def post(self, sender: str, content: Any, note: str = "") -> RoleMessage:
        """每次交接都留一条：发送者、内容、说明。后面追责只看这份列表。"""
        rendered = content if isinstance(content, str) else json.dumps(
            content, ensure_ascii=False
        )
        message = RoleMessage(seq=len(self.messages) + 1, sender=sender,
                              content=rendered, note=note)
        self.messages.append(message)
        return message

    # ---- 下面几个属性给渲染与验收用，避免各处重复解析 execution 字典 ----

    @property
    def observations(self) -> list[dict[str, Any]]:
        rows = self.execution.get("steps") if isinstance(self.execution, dict) else None
        return [row for row in rows if isinstance(row, dict)] if rows else []

    @property
    def failures(self) -> list[dict[str, Any]]:
        rows = self.execution.get("failures") if isinstance(self.execution, dict) else None
        return [row for row in rows if isinstance(row, dict)] if rows else []

    @property
    def replanned(self) -> bool:
        return bool(self.execution.get("replanned")) if isinstance(self.execution, dict) else False

    @property
    def senders(self) -> list[str]:
        """这次协作里实际发过言的角色（去重、按出现顺序）。"""
        seen: list[str] = []
        for message in self.messages:
            if message.sender not in seen:
                seen.append(message.sender)
        return seen

    @property
    def evidence_sufficient(self) -> bool:
        return bool(self.evidence.get("sufficient")) if isinstance(self.evidence, dict) else False


# ---------------------------------------------------------------- ③ 五个角色


def _system(text: str) -> SystemMessage:
    return SystemMessage(content=text)


def _human(text: str) -> HumanMessage:
    return HumanMessage(content=text)


def _payload(**fields: Any) -> str:
    """把交接数据拼成一段 JSON：标记在前，离线剧本模型据此取出。"""
    return f"{PAYLOAD_MARK}\n{json.dumps(fields, ensure_ascii=False)}"


def _json_object(content: Any, role: str) -> dict[str, Any]:
    """解析角色输出；协议不符就报错，绝不把自由文本当结构化结论。"""
    raw = str(content).strip()
    if raw.startswith("```"):
        raw = raw.strip()
        raw = raw[3:]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise CrewProtocolError(f"{role}输出不是 JSON：{raw[:160]}") from exc
    if not isinstance(parsed, dict):
        raise CrewProtocolError(f"{role}输出必须是 JSON 对象：{raw[:160]}")
    return parsed


def planner_agent(
    model: Any,
    registry: Any,
    task: str,
    feedback: str = "",
    max_steps: int = 3,
) -> TaskPlan:
    """规划智能体：复用单元20 的 `make_plan`（白名单内的 JSON 计划）。"""
    return make_plan(model, registry, task, feedback, max_steps=max_steps)


def executor_agent(model: Any, registry: Any, plan: TaskPlan) -> dict[str, Any]:
    """执行智能体：复用单元20 的 `execute_plan`，但**关掉总结**。

    返回原始观察与失败记录；答复由答复智能体写。这样"谁写的答复"只有一个答案。
    """
    return execute_plan(model, registry, plan, compose_answer=False)


def analyze_evidence(model: Any, task: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """证据核验智能体：只判断观察够不够，不负责措辞。

    输入：模型、用户目标、执行智能体留下的观察列表。
    输出：{"sufficient": 布尔, "reason": 理由, "missing": 缺什么}。
    """
    response = model.invoke([
        _system(
            f"你是电商售后智能体的{EVIDENCE_MARK}。只看真实工具观察，不凭常识补业务事实。"
            "重点检查：未找到、检索 0 条命中、缺来源或缺关键字段。"
            '只输出 JSON：{"sufficient":布尔值,"reason":"理由","missing":"缺什么；无则写无"}。'
        ),
        _human(_payload(task=task, steps=steps)),
    ])
    result = _json_object(getattr(response, "content", response), ROLE_NAMES[ROLE_EVIDENCE])
    if not isinstance(result.get("sufficient"), bool) or not all(
        isinstance(result.get(key), str) for key in ("reason", "missing")
    ):
        raise CrewProtocolError(
            f"{ROLE_NAMES[ROLE_EVIDENCE]}缺少 sufficient、reason 或 missing 字段：{result}"
        )
    return result


def compose_answer(
    model: Any,
    task: str,
    steps: list[dict[str, Any]],
    evidence: dict[str, Any],
    feedback: str = "",
) -> str:
    """答复智能体：依据观察作答；评审不通过时按意见修订一次。

    输入：目标、观察、证据意见、可选的评审反馈。输出：一版草稿（文本）。
    它**没有放行权限**：草稿写进黑板后必须等评审结论。
    """
    response = model.invoke([
        _system(
            "你是电商售后智能体的答复员。只引用工具观察中的事实，保留编号、日期与文件名。"
            "证据不足就明说资料不足并建议转人工，不许虚构政策、时限或承诺。"
            "若有评审意见，据此修订表述，但**不得更改工具观察到的事实**。"
        ),
        _human(_payload(
            task=task, steps=steps, evidence_check=evidence, review_feedback=feedback or "无",
        )),
    ])
    draft = str(getattr(response, "content", response)).strip()
    if not draft:
        raise CrewProtocolError("答复智能体返回空内容")
    return draft


def review_answer(
    model: Any,
    task: str,
    steps: list[dict[str, Any]],
    evidence: dict[str, Any],
    answer: str,
) -> dict[str, Any]:
    """评审智能体：独立检查草稿。输入四件套，输出 {passed, reason}。"""
    response = model.invoke([
        _system(
            f"你是电商售后智能体的{REVIEWER_MARK}。对照目标、工具观察和证据核验意见"
            "检查这版答复：是否回应了问题、是否只用了观察里的事实、是否说明了缺口。"
            "不得改写工具观察，也不得代写答复。"
            '只输出 JSON：{"passed":布尔值,"reason":"理由"}。'
        ),
        _human(_payload(task=task, steps=steps, evidence_check=evidence, draft=answer)),
    ])
    result = _json_object(getattr(response, "content", response), ROLE_NAMES[ROLE_REVIEWER])
    if not isinstance(result.get("passed"), bool) or not isinstance(result.get("reason"), str):
        raise CrewProtocolError(
            f"{ROLE_NAMES[ROLE_REVIEWER]}缺少 passed 或 reason 字段：{result}"
        )
    return result


# ---------------------------------------------------------------- ④ 协调者


def _plan_rows(plan: TaskPlan) -> list[dict[str, Any]]:
    return [
        {"tool": step.tool, "arguments": dict(step.arguments), "reason": step.reason}
        for step in plan.steps
    ]


def _observation_rows(steps: list[Any]) -> list[dict[str, Any]]:
    """把 StepObservation 转成可 JSON 化的字典（黑板只存产物）。"""
    rows: list[dict[str, Any]] = []
    for item in steps:
        if isinstance(item, StepObservation):
            rows.append({
                "step": item.step,
                "tool": item.tool,
                "arguments": dict(item.arguments),
                "observation": item.observation,
                "elapsed_ms": item.elapsed_ms,
                "attempt": item.attempt,
            })
        elif isinstance(item, dict):
            rows.append(dict(item))
    return rows


def _failure_rows(failures: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in failures:
        if isinstance(item, StepFailure):
            rows.append({
                "step": item.step,
                "tool": item.tool,
                "arguments": dict(item.arguments),
                "error": item.error,
                "attempt": item.attempt,
                "after_replan": item.after_replan,
            })
        elif isinstance(item, dict):
            rows.append(dict(item))
    return rows


def coordinate(
    model: Any,
    registry: Any,
    task: str,
    max_revisions: int = MAX_REVISIONS,
    max_replans: int = MAX_PLAN_REPLANS,
) -> Blackboard:
    """协调五个角色跑完一次协作；评审拒绝只修订一次，二次拒绝转人工。

    输入：模型、单元15 的注册表、用户目标。输出：写满的 Blackboard。
    守门（目标不合格）在问任何角色之前；执行阶段终止时后面三个角色根本不启动。
    """
    clean_task = validate_goal(task)
    board = Blackboard(task=clean_task, model_name=_model_name(model))
    started = time.perf_counter()

    # ① 规划：只出计划，一步工具都不执行。
    plan = planner_agent(model, registry, clean_task)
    board.plan = _plan_rows(plan)
    board.plan_versions.append(list(board.plan))
    board.model_calls += 1
    board.post(ROLE_PLANNER, board.plan, f"第1版计划：{len(board.plan)} 步（未执行）")

    # ② 执行：真实工具在这里才被调用；失败沿用单元20 的"至多重规划一次"。
    try:
        outcome = executor_agent(model, registry, plan)
    except PlanAborted as exc:
        # 未解决的失败：没有观察就没有草稿，评审环节整个跳过（不假装审过）。
        board.status = STATUS_ABORTED
        board.error = str(exc)
        # 规划那一次在 ① 已经数过；这里只补"失败后又重新规划"的那几次。
        board.model_calls += plan.replans
        board.post(ROLE_COORDINATOR, str(exc), "执行阶段终止，不进入证据/答复/评审")
        board.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return board
    # 同上：只补重规划那几次（规划智能体本身那一次在 ① 已计入）。
    board.model_calls += plan.replans
    board.execution = {
        "steps": _observation_rows(list(outcome.get("steps", []))),
        "failures": _failure_rows(list(outcome.get("failures", []))),
        "replanned": bool(outcome.get("replanned")),
        "attempts": outcome.get("attempts", 1),
    }
    if board.replanned:
        # 计划被改过：黑板同步最新那一版，免得展示的仍是第一次规划。
        board.plan = _plan_rows(plan)
        board.plan_versions.append(list(board.plan))
        board.post(ROLE_PLANNER, board.plan, "重规划后的计划（第2版）")
    board.post(
        ROLE_EXECUTOR,
        {"steps": len(board.observations), "failures": len(board.failures),
         "replanned": board.replanned},
        f"逐步观察：{len(board.observations)} 条成功、{len(board.failures)} 次失败",
    )

    rows = board.observations

    # ③ 证据核验：只判断够不够。
    board.evidence = analyze_evidence(model, clean_task, rows)
    board.model_calls += 1
    board.post(
        ROLE_EVIDENCE, board.evidence,
        f"证据{'充分' if board.evidence_sufficient else '不足'}：{board.evidence.get('missing', '')}",
    )

    # ④ 答复：先交草稿，它自己不能放行。
    board.draft = compose_answer(model, clean_task, rows, board.evidence)
    board.drafts.append(board.draft)
    board.model_calls += 1
    board.post(ROLE_RESPONDER, board.draft, "第1版草稿（等评审，不得自行放行）")

    # ⑤ 评审：最多评 max_revisions + 1 次，中间最多修订 max_revisions 次。
    for attempt in range(max_revisions + 1):
        board.review = review_answer(model, clean_task, rows, board.evidence, board.draft)
        board.model_calls += 1
        board.review_history.append(dict(board.review))
        board.post(
            ROLE_REVIEWER, board.review,
            f"第{attempt + 1}次评审：{'通过' if board.review['passed'] else '不通过'}",
        )
        if board.review["passed"]:
            board.answer = board.draft
            board.delivered = True
            board.status = STATUS_DELIVERED
            board.post(ROLE_COORDINATOR, "评审通过，草稿成为正式答复", "自动交付")
            break
        if attempt < max_revisions:
            # 受控修订：把评审理由交回答复智能体，只改表述、不改工具事实。
            board.revisions += 1
            board.draft = compose_answer(
                model, clean_task, rows, board.evidence, feedback=str(board.review["reason"]),
            )
            board.drafts.append(board.draft)
            board.model_calls += 1
            board.post(ROLE_RESPONDER, board.draft, f"按评审意见修订（第{attempt + 2}版草稿）")

    if not board.delivered:
        # 仍未通过：答复留空，草稿与评审意见留在黑板上给人工复核。
        board.answer = ""
        board.status = STATUS_HUMAN
        reason = str(board.review.get("reason", "")) if board.review else ""
        board.post(
            ROLE_COORDINATOR,
            f"{HUMAN_REVIEW_NOTE}评审意见：{reason}",
            f"停止自动交付（已修订 {board.revisions} 次，上限 {max_revisions}）",
        )
    board.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return board


# ---------------------------------------------------------------- ⑤ 离线剧本模型


@dataclass
class CrewScript:
    """一份离线剧本：计划、答复草稿序列、评审结论序列。

    为什么要"序列"：评审要能演"第一次不通过 → 修订 → 第二次通过"，
    所以评审结论按调用顺序取，而不是固定一个值。
    """

    steps: list[dict[str, Any]] = field(default_factory=list)
    revised: list[dict[str, Any]] | None = None
    drafts: list[str] = field(default_factory=list)
    reviews: list[dict[str, Any]] = field(default_factory=list)


class ScriptedCrewModel(BaseChatModel):
    """离线剧本模型：占的就是 ChatDeepSeek 那一格，没有密钥也能跑完整个协作。

    它按提示词里的角色标记分派（**看全部消息**，因为角色写在 system、数据写在 human）：

    * `{PLANNER_MARK}` → 输出 JSON 计划；带"上次执行反馈"且不是"无"时输出修订计划；
    * `{EVIDENCE_MARK}` → 按规则核验观察（未找到 / 0 命中 → 不充分），不编判断；
    * `{REVIEWER_MARK}` → 按剧本顺序取评审结论；取完默认通过；
    * 其余（答复员）→ 按剧本顺序取草稿；取完默认把观察原文摘录出来。

    证据核验在离线模式下用规则而不是剧本，理由是：它判断的是"观察里有没有东西"，
    这种判断有唯一正确答案，写死成剧本反而会在换场景时说谎。
    """

    model_name: str = OFFLINE_MODEL_NAME
    steps: list[dict[str, Any]] = []
    revised: list[dict[str, Any]] | None = None
    drafts: list[str] = []
    reviews: list[dict[str, Any]] = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return OFFLINE_MODEL_NAME

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        object.__setattr__(self, "calls", int(self.calls) + 1)
        joined = _joined_text(messages)
        if REVIEWER_MARK in joined:
            content = json.dumps(self._take_review(), ensure_ascii=False)
        elif EVIDENCE_MARK in joined:
            content = json.dumps(_scripted_evidence(_payload_json(joined)), ensure_ascii=False)
        elif PLANNER_MARK in joined:
            is_revision = FEEDBACK_MARK in joined and f"{FEEDBACK_MARK}无" not in joined
            plan = self.revised if (is_revision and self.revised is not None) else self.steps
            content = json.dumps({"steps": [dict(item) for item in plan]}, ensure_ascii=False)
        else:
            content = self._take_draft(joined)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    # ---- 两个按调用顺序取的剧本队列 ----

    def _take_review(self) -> dict[str, Any]:
        if self.reviews:
            return dict(self.reviews.pop(0))
        return {"passed": True, "reason": "（离线剧本）未安排评审结论，默认通过：草稿只摘录观察原文"}

    def _take_draft(self, prompt: str) -> str:
        if self.drafts:
            return self.drafts.pop(0)
        return _scripted_draft(_payload_json(prompt))


def _scripted_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """离线证据核验：按规则判断"观察里有没有东西"，不编造结论。"""
    steps = payload.get("steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list) or not steps:
        return {
            "sufficient": False,
            "reason": "（离线规则）本次没有任何工具观察",
            "missing": "全部事实",
        }
    missing: list[str] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        index = item.get("step", "?")
        observation = item.get("observation")
        if not isinstance(observation, dict) or not observation:
            missing.append(f"第{index}步没有有效观察")
            continue
        if observation.get("status") == "未找到":
            key = observation.get("question_id") or observation.get("reimbursement_id") or ""
            missing.append(f"第{index}步未找到：{key}")
        elif isinstance(observation.get("hits"), list) and not observation["hits"]:
            missing.append(f"第{index}步检索 0 条命中")
    if missing:
        return {
            "sufficient": False,
            "reason": "（离线规则）观察里有未找到或空结果，不足以支撑完整答复",
            "missing": "；".join(missing),
        }
    return {
        "sufficient": True,
        "reason": f"（离线规则）{len(steps)} 条观察都返回了真实内容",
        "missing": "无",
    }


def _scripted_draft(payload: dict[str, Any]) -> str:
    """离线草稿：只把观察原文摘出来，不补、不算、不编。"""
    steps = payload.get("steps") if isinstance(payload, dict) else None
    lines = ["（离线脚本模式：以下只摘录工具返回的原文，不生成新事实，最终以人工复核为准）"]
    if not isinstance(steps, list) or not steps:
        lines.append("本次没有取得任何工具观察，资料不足，建议转人工核实。")
        return "\n".join(lines)
    for item in steps:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- 第{item.get('step', '?')}步 {item.get('tool', '')}："
            f"{_brief(item.get('observation'))}"
        )
    evidence = payload.get("evidence_check") if isinstance(payload, dict) else None
    if isinstance(evidence, dict) and not evidence.get("sufficient"):
        lines.append(f"证据核验意见：{evidence.get('missing', '')}——请人工复核后再答复。")
    return "\n".join(lines)


def _brief(observation: Any, limit: int = 120) -> str:
    if not isinstance(observation, dict):
        return " ".join(str(observation).split())[:limit]
    if observation.get("status") == "未找到":
        key = observation.get("question_id") or observation.get("reimbursement_id") or ""
        return f"未找到：{key}"
    if "answer" in observation:
        return " ".join(f"{observation.get('question_id', '')}：{observation['answer']}".split())[:limit]
    if "deadline" in observation:
        return " ".join(
            f"签收 {observation.get('sign_date', '')} + {observation.get('window_days', '')} 天"
            f" → 截止 {observation['deadline']}（依据 {observation.get('basis_doc', '')}）".split()
        )[:limit]
    if isinstance(observation.get("hits"), list):
        hits = observation["hits"]
        return f"{len(hits)} 条命中：" + " ".join(
            str(hits[0].get("text", "")).split() if hits and isinstance(hits[0], dict) else ""
        )[:limit]
    return " ".join(json.dumps(observation, ensure_ascii=False).split())[:limit]


def _payload_json(text: str) -> dict[str, Any]:
    """取出交接数据那段 JSON；取不到就当空字典，离线剧本不因此崩掉。"""
    raw = text.split(PAYLOAD_MARK, 1)[-1].strip() if PAYLOAD_MARK in text else ""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _joined_text(messages: Any) -> str:
    """角色标记写在 system 里、数据写在 human 里，判断角色必须看全部消息。"""
    parts: list[str] = []
    for message in _iter_messages(messages):
        content = getattr(message, "content", message)
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _iter_messages(messages: Any):
    if isinstance(messages, (list, tuple)):
        return list(messages)
    to_messages = getattr(messages, "to_messages", None)
    if callable(to_messages):
        return list(to_messages())
    return [messages]


def _model_name(model: Any) -> str:
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


# ---------------------------------------------------------------- ⑥ 渲染


def render_roles() -> str:
    """任务卡要交的第一件东西：角色表（职责 / 输入 / 输出 / 不许做什么）。"""
    lines = ["五个专职角色（任务卡要求至少 3 个）", "角色\t职责\t输入\t输出\t不许做什么"]
    for role in ROLES:
        lines.append(f"{role.name}\t{role.duty}\t{role.inputs}\t{role.outputs}\t{role.forbidden}")
    lines.append(
        f"{ROLE_NAMES[ROLE_COORDINATOR]}\t调度与放行判断（不调模型）\t各角色写进黑板的产物\t"
        "交付 / 转人工的结论\t不许替任何角色写产物、不许绕过评审放行"
    )
    return "\n".join(lines)


def render_flow() -> str:
    """架构图：协调者 + 五个角色 + 共享黑板 + 评审回路。"""
    return "\n".join([
        "多智能体架构（本组电商售后）",
        "",
        "  用户目标",
        "    │  协调者 coordinate()：只调度，不调模型",
        "    ▼",
        "  ① 规划智能体 → 1~3 步 JSON 计划 ─────┐",
        "    ▼                                  │",
        "  ② 执行智能体 → 逐步观察（真实工具）─┤ 写进共享黑板",
        "    ▼                                  │ （每条都留发送者）",
        "  ③ 证据核验智能体 → {充分, 理由, 缺什么}┤",
        "    ▼                                  │",
        "  ④ 答复智能体 → 一版草稿（不可自行放行）┘",
        "    ▼",
        "  ⑤ 评审智能体 → {通过, 理由}",
        "    │       ├ 通过 → 协调者交付（草稿 → 答复）",
        "    │       └ 不通过 → 把理由交回 ④，**只许修订一次**",
        "    │                    ├ 修订后通过 → 交付",
        "    │                    └ 修订后仍不通过 → **停止自动交付**，转人工复核",
        "    └──────────── 评审回路最多走一轮 ────────────┘",
        "",
        "执行阶段若终止（计划修订后仍失败），③④⑤ 三个角色根本不启动。",
    ])


def render_handoff(board: Blackboard) -> str:
    """共享黑板的交接记录：序号 / 发送者 / 这次交接是什么 / 内容。"""
    lines = ["共享黑板交接记录（可逐步追溯）", "序号\t发送者\t说明\t内容"]
    for message in board.messages:
        lines.append(
            f"{message.seq}\t{message.sender_name}\t{message.note}\t{message.preview()}"
        )
    return "\n".join(lines)


def render_plan(board: Blackboard) -> str:
    if not board.plan:
        return "执行计划：（空）"
    lines = ["执行计划（规划智能体产出，执行前）", "步\t工具\t参数\t这一步解决什么"]
    for index, step in enumerate(board.plan, start=1):
        payload = ", ".join(f"{key}={value}" for key, value in step.get("arguments", {}).items())
        lines.append(f"{index}\t{step.get('tool', '')}\t{payload or '（无）'}\t"
                     f"{step.get('reason', '—')}")
    if len(board.plan_versions) > 1:
        lines.append(f"注：共 {len(board.plan_versions)} 版计划（执行阶段重规划过一次）")
    return "\n".join(lines)


def render_observations(board: Blackboard) -> str:
    """观察与失败按「第几轮 → 第几步」排序。

    重规划后新计划从第 1 步重跑，分开两个列表打印会把第2轮的成功排在第1轮的
    失败前面，看不出失败的先后；合起来排序才读得出"第1轮第3步错了，所以重来"。
    """
    lines = ["工具观察（执行智能体产出）", "轮\t步\t工具\t参数\t观察\t耗时ms"]
    events: list[tuple[int, int, int, dict[str, Any]]] = [
        (int(row.get("attempt", 1)), int(row.get("step", 0)), 0, row)
        for row in board.observations
    ] + [
        (int(row.get("attempt", 1)), int(row.get("step", 0)), 1, row)
        for row in board.failures
    ]
    for attempt, step, kind, row in sorted(events, key=lambda item: (item[0], item[1])):
        payload = ", ".join(f"{key}={value}" for key, value in row.get("arguments", {}).items())
        detail = _brief(row.get("observation")) if kind == 0 else f"失败：{row.get('error', '')}"
        cost = row.get("elapsed_ms", "—") if kind == 0 else "—"
        lines.append(
            f"第{attempt}轮\t{step}\t{row.get('tool', '')}\t{payload or '（无）'}\t"
            f"{detail}\t{cost}"
        )
    if not events:
        lines.append("（本次没有执行任何步骤）")
    return "\n".join(lines)


def render_result(board: Blackboard) -> str:
    """一次协作的完整结果：目标 → 计划 → 观察 → 证据 → 草稿 → 评审 → 结论。"""
    lines = [
        f"目标：{board.task}",
        "",
        render_plan(board),
        "",
        render_observations(board),
        "",
        # 证据核验没跑过就如实写"未发生"，别拿空字典渲染成 sufficient=False。
        (
            "证据核验：（未发生，执行阶段已终止）"
            if not board.evidence
            else "证据核验："
            f"sufficient={board.evidence_sufficient}；理由：{board.evidence.get('reason', '—')}；"
            f"缺：{board.evidence.get('missing', '—')}"
        ),
        "",
    ]
    for index, draft in enumerate(board.drafts, start=1):
        tag = "第1版草稿" if index == 1 else f"第{index}版草稿（按评审意见修订）"
        lines.append(f"{tag}：{' '.join(draft.split())[:MESSAGE_PREVIEW]}")
    if not board.drafts:
        lines.append("草稿：（无，执行阶段就终止了）")
    lines.append("")
    for index, review in enumerate(board.review_history, start=1):
        lines.append(
            f"第{index}次评审：passed={review.get('passed')}；理由：{review.get('reason', '')}"
        )
    if not board.review_history:
        lines.append("评审：（未发生）")
    lines.append("")
    lines.append(f"状态：{board.status}")
    lines.append(f"交付的答复：{board.answer or '（空：未自动交付）'}")
    if board.error:
        lines.append(f"异常记录：{board.error}")
    lines.append(
        f"模型调用 {board.model_calls} 次（规划 {1 + (1 if board.replanned else 0)}"
        f" + 证据 1 + 答复 {len(board.drafts)} + 评审 {len(board.review_history)}）；"
        f"耗时 {board.elapsed_ms} ms"
    )
    return "\n".join(lines)


CONCEPTS: list[tuple[str, str, str]] = [
    ("智能体", "有明确输入输出的一个角色", "本单元五个：规划/执行/证据/答复/评审"),
    ("协调者", "调度与放行，不调模型", "coordinate()：谁先谁后、能不能交付由它定"),
    ("共享黑板", "角色之间唯一的交接面", "Blackboard.post() 每条都留发送者"),
    ("角色交接", "上一步的产物是下一步的输入", "计划→观察→证据意见→草稿→评审结论"),
    ("受控修订", "评审不通过只改一次", f"MAX_REVISIONS = {MAX_REVISIONS}"),
    ("停止自动交付", "修订后仍不通过就转人工", "答复留空，草稿与意见留给人工复核"),
]


def render_concepts() -> str:
    lines = ["概念记录（第1节）", "概念\t含义\t在本项目里落在哪"]
    lines.extend(f"{name}\t{meaning}\t{here}" for name, meaning, here in CONCEPTS)
    return "\n".join(lines)


__all__ = [
    "Blackboard",
    "CrewProtocolError",
    "CrewScript",
    "HUMAN_REVIEW_NOTE",
    "MAX_REVISIONS",
    "ROLES",
    "ROLE_COORDINATOR",
    "ROLE_EVIDENCE",
    "ROLE_EXECUTOR",
    "ROLE_NAMES",
    "ROLE_ORDER",
    "ROLE_PLANNER",
    "ROLE_RESPONDER",
    "ROLE_REVIEWER",
    "RoleMessage",
    "STATUS_ABORTED",
    "STATUS_DELIVERED",
    "STATUS_HUMAN",
    "ScriptedCrewModel",
    "analyze_evidence",
    "compose_answer",
    "coordinate",
    "executor_agent",
    "planner_agent",
    "render_concepts",
    "render_flow",
    "render_handoff",
    "render_observations",
    "render_plan",
    "render_result",
    "render_roles",
    "review_answer",
]
