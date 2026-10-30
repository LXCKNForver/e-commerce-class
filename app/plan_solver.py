"""单元20：Plan-and-Solve —— 模型只出计划，程序逐步执行，失败后至多重规划一次。

前面三个单元各自解决了"谁来编排"的问题：

* 单元17 让模型每一步自己决定调哪个工具（边想边做，步数靠上限兜底）；
* 单元18 把顺序与分支写死在程序里（模型只在摘要与润色两步说话）；
* 单元19 给执行器加记忆（跨轮只解决「哪一项」，事实仍现查）。

单元20 是第三种：**先把目标拆成一张计划表，再照着表一步一步执行**。区别在
"计划"这个中间产物是显式的——能打印、能对账、能在课堂上逐条核对：

    目标（一句话业务问题）
      │ ① 规划（模型第 1 次调用：只输出 JSON 计划，一个工具都不执行）
      ▼
    TaskPlan：1~3 个 PlanStep（工具名 + 参数 + 这一步解决什么）
      │ ② 逐步执行（Python 按 cursor 推进，仍过单元15 的白名单与格式校验）
      │      ├ 成功 → 记一条观察（第几步、工具、参数、真实返回值、耗时）
      │      └ 失败 → ③ 至多重规划一次（把失败原因与已有观察交回规划器）
      │              ├ 修订后跑完 → 标记 replanned=True，照常出答复
      │              └ 修订后仍失败 → 终止并报错（不出答复）
      ▼
    ④ 答复（模型最后一次调用，只许用观察里出现过的事实）

五条要记住的（也是课堂要讲清的）：

1. **计划是数据，不是指令**。模型给的只有"工具名 + 参数"，路径、SQL、代码一概
   没有；执行一律走单元15 的 `ToolRegistry.call()`，清单外的工具照样跑不了。
2. **执行状态只在本次运行内保存**：`TaskPlan` 活在内存里，不落库——正好和单元19
   的长期记忆相反（那是跨运行，这是本次运行），两者别混为一谈。
3. **重规划只有一次机会**。失败时把"第几步、什么错、已经拿到了什么"交回规划器；
   修订后的计划仍失败就直接终止，绝不无限重试（否则一次坏参数能跑一整节课）。
4. **没解决的失败不许报成完成**：状态是 aborted、答复为空，连最后一次总结模型都
   不调用——"看起来答完了"比"明确报错"危险得多。
5. **格式不合法 ≠ 查不到**。编号写成 BAD 会被格式校验拒绝（触发重规划）；编号
   格式合法但库里没有（如 EC-FAQ-99）只是"未找到"，是正常观察，不触发重规划。

    uv run --extra embedding --extra chroma --extra langchain python -m app.plan_lab
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.lc_chain import build_chat_model
from app.toolkit import ALLOWED_TOOLS, ToolRejected

# ---------------------------------------------------------------- ① 常量

# 规划器能挑的工具：与单元15 的白名单是同一份（不许自造工具名）。
# 计划里过了这一关，执行时还要再过一次 `registry.call` 的参数格式校验——
# 两道门各管一件事：第一道管"能不能提"，第二道管"参数对不对"。
PLANNER_ALLOWED: frozenset[str] = frozenset(ALLOWED_TOOLS)

MIN_PLAN_STEPS = 1
MAX_PLAN_STEPS = 3
# 任务卡的原话：失败后**至多**重规划一次。写成常量，执行器与渲染都读它。
MAX_REPLANS = 1

MIN_GOAL_CHARS = 4
MAX_GOAL_CHARS = 200

# 三种结束状态：跑完了、修订后仍失败、计划本身没生成（守门或解析）。
STATUS_DONE = "计划执行完成"
STATUS_ABORTED = "修订后仍失败，已终止（未产出答复）"
STATUS_NO_PLAN = "未生成可执行计划"

# 两种模型角色写进提示词里：离线剧本模型靠这两个标记认出"此刻该干什么"。
PLANNER_MARK = "任务规划员"
ANSWERER_MARK = "答复员"
# 首次规划时反馈是空的；执行器据此区分"第一次规划"与"失败后的重规划"。
EMPTY_FEEDBACK = "无"
FEEDBACK_MARK = "上次执行反馈："

OBSERVE_PREVIEW = 160
OFFLINE_MODEL_NAME = "scripted-plan-model"


class PlanAborted(RuntimeError):
    """修订后的计划仍然失败：终止本次运行，不产出答复。"""


# ---------------------------------------------------------------- ② 数据结构


@dataclass(frozen=True)
class PlanStep:
    """计划里的一步：工具名 + 参数 + 这一步解决什么。

    注意这里**没有** execute：计划只描述"要做什么"，做的过程在 `execute_plan` 里，
    而且只能走注册表。模型给不出函数、路径或 SQL——它也写不进这个 dataclass。
    """

    tool: str
    arguments: dict[str, str] = field(default_factory=dict)
    reason: str = ""

    def describe(self) -> str:
        payload = ", ".join(f"{key}={value}" for key, value in self.arguments.items())
        return f"{self.tool}({payload})" if payload else f"{self.tool}()"


@dataclass
class StepObservation:
    """一条真实观察：第几步、哪个工具、传了什么、返回了什么、花了多久。"""

    step: int
    tool: str
    arguments: dict[str, str]
    observation: dict[str, Any]
    elapsed_ms: int
    attempt: int = 1

    @property
    def ok(self) -> bool:
        return True

    def preview(self, limit: int = OBSERVE_PREVIEW) -> str:
        return _observation_text(self.observation, limit)


@dataclass
class StepFailure:
    """一次失败的尝试：为什么失败、是不是已经重规划过。"""

    step: int
    tool: str
    arguments: dict[str, str]
    error: str
    attempt: int = 1
    after_replan: bool = False


@dataclass
class TaskPlan:
    """计划 + 本次运行的执行状态。

    cursor 指向下一步；observations 只增不减（失败前的成功步骤留作审计证据）；
    replans 是已经用掉的重规划次数，到 MAX_REPLANS 就不再问模型要计划了。

    这份状态**只在内存里**：任务卡明写"执行状态只在本次运行内保存，不要求落库"。
    """

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    cursor: int = 0
    observations: list[StepObservation] = field(default_factory=list)
    failures: list[StepFailure] = field(default_factory=list)
    replans: int = 0
    attempt: int = 1
    # 每一轮规划的版本都留一份：课堂上要对照"修订前 vs 修订后"改了哪一步。
    versions: list[list[PlanStep]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.versions:
            self.versions = [list(self.steps)]

    @property
    def replanned(self) -> bool:
        return self.replans > 0

    @property
    def pending(self) -> int:
        return max(len(self.steps) - self.cursor, 0)


@dataclass
class PlanRun:
    """一次 Plan-and-Solve 的全部产出；渲染与验收只从这份数据里取，不重跑。"""

    goal: str
    status: str
    answer: str = ""
    plan: TaskPlan | None = None
    replanned: bool = False
    replans: int = 0
    model_calls: int = 0
    elapsed_ms: int = 0
    error: str = ""
    model_name: str = ""

    @property
    def steps(self) -> list[StepObservation]:
        return list(self.plan.observations) if self.plan else []

    @property
    def failures(self) -> list[StepFailure]:
        return list(self.plan.failures) if self.plan else []

    @property
    def finished(self) -> bool:
        """只有真的跑完才算完成；aborted 与 no_plan 一律不算。"""
        return self.status == STATUS_DONE


# ---------------------------------------------------------------- ③ 守门与规划


def validate_goal(goal: Any) -> str:
    """输入守门：目标不合格就在问模型之前拦下（不浪费一次规划调用）。"""
    if not isinstance(goal, str):
        raise ValueError(f"目标必须是字符串，实际是：{type(goal).__name__}")
    clean = " ".join(goal.split())
    if not clean:
        raise ValueError("目标不能为空")
    if len(clean) < MIN_GOAL_CHARS:
        raise ValueError(f"目标至少 {MIN_GOAL_CHARS} 个字，实际只有 {len(clean)} 个")
    if len(clean) > MAX_GOAL_CHARS:
        raise ValueError(
            f"目标最多 {MAX_GOAL_CHARS} 个字，实际 {len(clean)} 个；过长会撑爆计划窗口"
        )
    return clean


def _strip_fence(raw: str) -> str:
    """模型爱用 ```json 包住 JSON：先剥掉围栏再解析，别把围栏当内容。"""
    text = raw.strip()
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()


def make_plan(
    model: Any,
    registry: Any,
    goal: str,
    feedback: str = "",
    max_steps: int = MAX_PLAN_STEPS,
) -> TaskPlan:
    """输入目标；模型输出 JSON 子任务；返回**白名单内**的计划。

    输入：model 已配置的聊天模型；registry 提供工具说明；goal 用户目标；
    feedback 首次为空，步骤失败后装入"哪一步、什么错、已有什么"。
    输出：TaskPlan，含 1~3 个 PlanStep，**尚不包含任何工具执行结果**。

    这一阶段只调用模型、不执行工具：模型给的路径、SQL、代码不会被运行，
    因为它们根本进不了 `PlanStep`（只有 tool / arguments / reason 三个字段）。
    """
    clean_goal = validate_goal(goal)
    schema = (
        '只输出 JSON 对象，格式：{"steps":[{"tool":"工具名",'
        '"arguments":{"参数名":"参数值"},"reason":"这一步解决什么"}]}。'
        f"最多 {max_steps} 步，至少 {MIN_PLAN_STEPS} 步。"
        "可用工具与参数（工具名必须一字不差）：\n"
        f"{registry.descriptions()}\n"
        "只选完成目标必需的工具；不生成最终回答、代码、路径或虚构编号。"
    )
    response = model.invoke([
        _system(f"你是电商售后智能体的{PLANNER_MARK}。" + schema),
        _human(f"目标：{clean_goal}\n{FEEDBACK_MARK}{feedback or EMPTY_FEEDBACK}"),
    ])
    raw = _strip_fence(str(getattr(response, "content", response)))
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"规划结果不是约定的 JSON：{raw[:160]}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"规划结果必须是 JSON 对象：{raw[:160]}")
    items = payload.get("steps")
    if not isinstance(items, list) or not MIN_PLAN_STEPS <= len(items) <= max_steps:
        actual = len(items) if isinstance(items, list) else type(items).__name__
        raise ValueError(
            f"计划必须包含 {MIN_PLAN_STEPS} 到 {max_steps} 个步骤，实际 {actual}"
        )
    steps: list[PlanStep] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第{index}步不是 JSON 对象")
        tool = item.get("tool")
        if tool not in PLANNER_ALLOWED:
            raise ValueError(
                f"第{index}步选择了非白名单工具：{tool!r}；只有 {sorted(PLANNER_ALLOWED)} 可用"
            )
        raw_arguments = item.get("arguments", {})
        if not isinstance(raw_arguments, dict):
            raise ValueError(f"第{index}步的参数不是 JSON 对象")
        arguments: dict[str, str] = {}
        for key, value in raw_arguments.items():
            # JSON 里没有"字符串"这一说：模型给数字 7 和给 "7" 是一回事。
            # 宽的是类型不是值——格式对不对由 registry.call 的正则说了算。
            if not isinstance(key, str):
                raise ValueError(f"第{index}步的参数名必须是字符串：{key!r}")
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError(
                    f"第{index}步参数 {key} 类型不正确：{type(value).__name__}"
                )
            arguments[key] = str(value).strip()
        reason = item.get("reason")
        steps.append(
            PlanStep(
                tool=str(tool),
                arguments=arguments,
                reason=str(reason)[:60] if isinstance(reason, (str, int)) else "",
            )
        )
    return TaskPlan(goal=clean_goal, steps=steps)


# ---------------------------------------------------------------- ④ 执行与重规划


def execute_plan(
    model: Any,
    registry: Any,
    plan: TaskPlan,
    max_replans: int = MAX_REPLANS,
    compose_answer: bool = True,
) -> dict[str, Any]:
    """按计划逐步调用真实工具；失败时把原因交回规划器，至多重规划一次。

    输入：模型、单元15 的注册表、一份 TaskPlan。
    输出：答复、观察列表、失败列表、是否重规划、模型调用次数。

    与 `make_plan` 的分工不同：这里才真正访问 SQLite / Chroma / 资料原文。
    全部步骤跑完后才让模型把观察整理成答复——**失败终止时连这一步都不做**，
    所以"没解决"不会伪装成"答完了"。

    `compose_answer=False` 是单元21 用的开关：多智能体架构里**答复是专职
    角色的事**，执行智能体只交出观察，不许顺手把答案也写了。默认 True，
    单元20 的行为一行不变。
    """
    # 用 while 而不是 for：失败后整个计划会被替换，要从新计划的第一步重新走。
    while plan.cursor < len(plan.steps):
        step = plan.steps[plan.cursor]
        started = time.perf_counter()
        try:
            # 关键断点：工具在这里才真正执行。registry.call 先过白名单与格式正则，
            # 再进真实业务函数；返回值就是后面答复的唯一依据。
            observation = registry.call(step.tool, step.arguments)
        except (ToolRejected, ValueError, KeyError, FileNotFoundError) as exc:
            elapsed_ms = _elapsed_ms(started)
            plan.failures.append(
                StepFailure(
                    step=plan.cursor + 1,
                    tool=step.tool,
                    arguments=dict(step.arguments),
                    error=str(exc),
                    attempt=plan.attempt,
                    after_replan=plan.replans > 0,
                )
            )
            if plan.replans >= max_replans:
                # 修订后的计划还是失败：终止。不重试、不猜、不产出答复。
                raise PlanAborted(f"修订后第{plan.cursor + 1}步仍失败：{exc}") from exc
            # 把"哪一步、什么错、已经拿到了什么"交回规划器，让它自己改。
            # 之前成功的观察留在列表里当审计证据，新计划另起一轮（attempt + 1）。
            revised = make_plan(model, registry, plan.goal, _feedback(step, exc, plan))
            plan.steps = revised.steps
            plan.versions.append(list(revised.steps))
            plan.cursor = 0
            plan.replans += 1
            plan.attempt += 1
            continue
        plan.observations.append(
            StepObservation(
                step=plan.cursor + 1,
                tool=step.tool,
                arguments=dict(step.arguments),
                observation=observation,
                elapsed_ms=_elapsed_ms(started),
                attempt=plan.attempt,
            )
        )
        plan.cursor += 1
    answer = _summarize(model, plan) if compose_answer else ""
    return {
        "answer": answer,
        "steps": plan.observations,
        "failures": plan.failures,
        "replanned": plan.replanned,
        # 模型调用 = 1 次规划 + N 次重规划 +（0 或 1）次总结，是可数的，不是口头承诺。
        "model_calls": 1 + plan.replans + (1 if compose_answer else 0),
        "attempts": plan.attempt,
    }


def solve(
    goal: str,
    registry: Any,
    model: Any,
    max_steps: int = MAX_PLAN_STEPS,
    max_replans: int = MAX_REPLANS,
) -> PlanRun:
    """一次完整的 Plan-and-Solve：规划 → 逐步执行 →（必要时重规划一次）→ 答复。

    输入：业务目标、单元15 的注册表、模型。输出：PlanRun（含状态与全部证据）。
    守门失败与"模型没给出合法计划"直接抛出（在问工具之前）；
    只有"修订后仍失败"被收进 `PlanRun.status = aborted`，答复留空。
    """
    started = time.perf_counter()
    plan = make_plan(model, registry, goal, max_steps=max_steps)
    run = PlanRun(
        goal=plan.goal,
        status=STATUS_DONE,
        plan=plan,
        model_name=_model_name(model),
    )
    try:
        outcome = execute_plan(model, registry, plan, max_replans=max_replans)
    except PlanAborted as exc:
        # 未解决的失败：如实上报，绝不落到"完成"里。
        run.status = STATUS_ABORTED
        run.error = str(exc)
        run.answer = ""
        run.replanned = plan.replanned
        run.replans = plan.replans
        run.model_calls = 1 + plan.replans
        run.elapsed_ms = _elapsed_ms(started)
        return run
    run.answer = str(outcome["answer"])
    run.replanned = bool(outcome["replanned"])
    run.replans = plan.replans
    run.model_calls = int(outcome["model_calls"])
    run.elapsed_ms = _elapsed_ms(started)
    return run


def _feedback(step: PlanStep, error: Exception, plan: TaskPlan) -> str:
    """给规划器的反馈：哪一步、什么错、已经拿到了什么（观察原文截断）。"""
    done = [
        {"tool": item.tool, "arguments": item.arguments, "observation": item.preview(120)}
        for item in plan.observations
    ]
    return (
        f"第{plan.cursor + 1}步 {step.describe()} 执行失败：{error}。"
        f"已成功取得的观察：{json.dumps(done, ensure_ascii=False) if done else '无'}。"
        "请改掉出错的那一步（多半是参数格式不对），保留其余必要步骤。"
    )


def _summarize(model: Any, plan: TaskPlan) -> str:
    """最后一次模型调用：只许用观察里出现过的事实，不许补造政策或编号。"""
    observations = [
        {
            "step": item.step,
            "tool": item.tool,
            "arguments": item.arguments,
            "observation": item.observation,
        }
        for item in plan.observations
    ]
    response = model.invoke([
        _system(
            f"你是电商售后智能体的{ANSWERER_MARK}。只能依据上面工具返回的观察回答；"
            "保留编号、日期与文件名。观察里没有的时限、金额、政策一律不许补充；"
            "证据不足就明说资料不足并建议转人工。"
        ),
        _human(
            f"用户目标：{plan.goal}\n"
            f"工具观察：{json.dumps(observations, ensure_ascii=False)}"
        ),
    ])
    return str(getattr(response, "content", response)).strip()


def _system(text: str) -> SystemMessage:
    return SystemMessage(content=text)


def _human(text: str) -> HumanMessage:
    return HumanMessage(content=text)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


# ---------------------------------------------------------------- ⑤ 离线剧本模型


class ScriptedPlanModel(BaseChatModel):
    """离线剧本模型：占的就是 ChatDeepSeek 那一格，没有密钥也能跑完整条链路。

    它认两种角色（靠提示词里的标记判断，而不是靠调用顺序）：

    * `{PLANNER_MARK}` → 输出 JSON 计划；提示词里带"上次执行反馈"且不是"无"时，
      输出 `revised`（修订计划）。`revised` 不单独给就原样重发首次计划——
      这正好用来演示"修订后仍失败则终止"。
    * `{ANSWERER_MARK}` → 只把工具返回的原文摘出来，不生成任何新事实。
    """

    model_name: str = OFFLINE_MODEL_NAME
    steps: list[dict[str, Any]] = []
    revised: list[dict[str, Any]] | None = None
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
        # 角色写在 system 消息里，参数写在 human 消息里：判断角色要看全部消息，
        # 生成答复才只看最后一条（观察 JSON 在那儿）。
        text = _joined_text(messages)
        if PLANNER_MARK in text:
            # 首次规划的反馈固定是"无"；带了实际错误内容的那一次才是重规划。
            is_revision = FEEDBACK_MARK in text and f"{FEEDBACK_MARK}{EMPTY_FEEDBACK}" not in text
            plan = self.revised if (is_revision and self.revised is not None) else self.steps
            content = json.dumps(
                {"steps": [dict(item) for item in plan]}, ensure_ascii=False
            )
        else:
            content = _scripted_answer(text)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def build_plan_model(
    steps: list[dict[str, Any]] | None = None,
    revised: list[dict[str, Any]] | None = None,
    provider: str | None = None,
):
    """按环境变量挑模型：`deepseek` 走真实 ChatDeepSeek，`mock` 走离线剧本。

    真实模式刻意不做兜底：缺包或缺密钥就当场报错，让课堂上看见原因，
    而不是悄悄换成离线答案（单元05 静默降级那个坑代价很大）。
    """
    chosen = (provider or os.getenv("LLM_PROVIDER", "mock")).strip().lower()
    if chosen == "deepseek":
        return build_chat_model("deepseek")  # 复用单元16 的工厂：同一份密钥、同一模型名
    if chosen != "mock":
        raise ValueError("LLM_PROVIDER 只能是 mock 或 deepseek")
    return ScriptedPlanModel(steps=steps or [], revised=revised)


def plan_step(tool: str, reason: str = "", **arguments: Any) -> dict[str, Any]:
    """剧本里的一步：{"tool": ..., "arguments": {...}, "reason": ...}。"""
    return {"tool": tool, "arguments": dict(arguments), "reason": reason}


def _scripted_answer(prompt: str) -> str:
    """离线答复：把工具观察里的原文摘出来，不补、不算、不编。"""
    raw = prompt.split("工具观察：", 1)[-1].strip() if "工具观察：" in prompt else ""
    try:
        observations = json.loads(raw) if raw else []
    except ValueError:
        observations = []
    if not observations:
        return "（离线脚本模式）本次没有取得任何工具观察，资料不足，建议转人工核实。"
    lines = ["（离线脚本模式：以下为工具返回的原文摘录，不生成新事实，最终以人工复核为准）"]
    for item in observations:
        observation = item.get("observation") if isinstance(item, dict) else None
        lines.append(f"- 第{item.get('step', '?')}步 {item.get('tool', '')}："
                     f"{_observation_text(observation or {}, 160)}")
    return "\n".join(lines)


def _last_text(messages: Any) -> str:
    text = ""
    for message in _iter_messages(messages):
        content = getattr(message, "content", message)
        if isinstance(content, str):
            text = content
    return text


def _joined_text(messages: Any) -> str:
    """把所有消息拼成一段：system 里写角色，human 里写参数，两边都要看。"""
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
    if isinstance(model, ScriptedPlanModel):
        return f"{OFFLINE_MODEL_NAME}（离线脚本，不联网）"
    name = getattr(model, "model_name", None) or getattr(model, "model", "")
    return f"ChatDeepSeek {name}（真实调用）" if name else type(model).__name__


# ---------------------------------------------------------------- ⑥ 渲染


def _observation_text(observation: Any, limit: int = OBSERVE_PREVIEW) -> str:
    """把一条观察压成一行可读文本；不同工具挑各自的关键字段。"""
    if not isinstance(observation, dict):
        return " ".join(str(observation).split())[:limit]
    if observation.get("status") == "未找到":
        key = observation.get("question_id") or observation.get("reimbursement_id") or ""
        text = f"未找到：{key}（{observation.get('storage', '')}）"
    elif "answer" in observation:
        text = " ".join(
            f"{observation.get('question_id', '')}：{observation['answer']}".split()
        )
    elif "deadline" in observation:
        text = " ".join(
            f"签收 {observation.get('sign_date', '')} + {observation.get('window_days', '')} 天 "
            f"→ 截止 {observation['deadline']}（依据 {observation.get('basis_doc', '')}）".split()
        )
    elif "hits" in observation:
        hits = observation["hits"]
        sources = ", ".join(
            dict.fromkeys(str(hit.get("source", "")) for hit in hits if isinstance(hit, dict))
        )
        preview = " ".join(str(hits[0].get("text", "")).split())[:60] if hits else ""
        text = f"{len(hits)} 条命中（来源：{sources or '无'}）：{preview}"
    else:
        text = " ".join(json.dumps(observation, ensure_ascii=False).split())
    # 不管走哪一支，最后统一截断：来源清单本身就能把一行撑爆。
    return text[:limit]


def render_plan(plan: TaskPlan) -> str:
    """第2节课要交的东西：模型拆出来的计划（只是一张表，还没执行）。

    重规划过就把每一轮计划都列出来——"改回正确编号再查一次"这种改动要看得见。
    """
    if not plan.versions or not any(plan.versions):
        return "执行计划：（空）"
    lines = ["执行计划（模型只提工具与参数，尚未执行任何一步）"]
    for attempt, steps in enumerate(plan.versions, start=1):
        title = "第1轮计划" if len(plan.versions) == 1 else f"第{attempt}轮计划"
        if attempt > 1:
            title += "（失败后重规划）"
        lines.append(f"{title}：")
        lines.append("步\t工具\t参数\t这一步解决什么")
        for index, step in enumerate(steps, start=1):
            payload = ", ".join(f"{key}={value}" for key, value in step.arguments.items())
            lines.append(f"{index}\t{step.tool}\t{payload or '（无）'}\t{step.reason or '—'}")
    lines.append(
        f"当前计划 {len(plan.steps)} 步（上限 {MAX_PLAN_STEPS}）；工具全部来自单元15 的白名单"
        f"（{sorted(PLANNER_ALLOWED)}），参数格式到执行时才校验。"
    )
    return "\n".join(lines)


def render_trajectory(run: PlanRun) -> str:
    """第3节课要交的轨迹：每一步的调用、真实返回值或失败原因、耗时。

    按"第几轮 → 第几步"排序：重规划后新计划从第 1 步重新跑，不排序就会把
    第2轮的成功步骤排在第1轮的失败之前，看不出失败的先后。
    """
    lines = ["执行轨迹（含失败后重规划的那一次）", "轮\t步\t工具\t参数\t状态\t观察或错误\t耗时ms"]
    events: list[tuple[int, int, int, Any]] = [
        (item.attempt, item.step, 0, item) for item in run.steps
    ] + [(item.attempt, item.step, 1, item) for item in run.failures]
    for _attempt, _step, kind, item in sorted(events, key=lambda row: (row[0], row[1])):
        payload = ", ".join(f"{key}={value}" for key, value in item.arguments.items())
        if kind == 0:
            lines.append(
                f"第{item.attempt}轮\t{item.step}\t{item.tool}\t{payload or '（无）'}\t成功\t"
                f"{item.preview()}\t{item.elapsed_ms}"
            )
        else:
            lines.append(
                f"第{item.attempt}轮\t{item.step}\t{item.tool}\t{payload or '（无）'}\t"
                f"失败\t{item.error}\t—"
            )
    if not events:
        lines.append("（本次运行没有执行任何步骤）")
    lines.append(
        f"合计：成功 {len(run.steps)} 步，失败 {len(run.failures)} 次，"
        f"重规划 {run.replans} 次（上限 {MAX_REPLANS}）。"
    )
    return "\n".join(lines)


def render_result(run: PlanRun) -> str:
    """一次运行的证据：目标、模型、状态、是否重规划、答复、轨迹。"""
    lines = [
        f"目标：{run.goal}",
        f"模型：{run.model_name or '—'}",
        f"状态：{run.status}；是否重规划：{'是' if run.replanned else '否'}；"
        f"模型调用：{run.model_calls} 次（1 次规划 + {run.replans} 次重规划"
        f" + {'1' if run.finished else '0'} 次总结）",
    ]
    if run.error:
        lines.append(f"终止原因：{run.error}")
    lines.append(f"答复：{run.answer or '（未产出答复）'}")
    if run.plan is not None:
        lines.append("")
        lines.append(render_plan(run.plan))
    lines.append("")
    lines.append(render_trajectory(run))
    return "\n".join(lines)


def render_concepts() -> str:
    """第1节课要交的概念记录：几个容易混的词在本项目里分别指什么。"""
    lines = ["Plan-and-Solve 概念对照（本项目里的实际含义）", "概念\t是什么\t在本项目里落在哪"]
    for name, meaning, here in CONCEPTS:
        lines.append(f"{name}\t{meaning}\t{here}")
    return "\n".join(lines)


CONCEPTS: list[tuple[str, str, str]] = [
    ("规划（Plan）", "把目标拆成 1~3 个可执行步骤", "make_plan()：模型只输出 JSON，不执行工具"),
    ("求解（Solve）", "按计划逐步调用真实工具", "execute_plan()：走单元15 的注册表"),
    ("观察", "工具返回的真实结果", "StepObservation：第几步、参数、返回值、耗时"),
    ("重规划", "失败后把错误交回规划器改一次", "replans 上限 1；改完从第一步重跑"),
    ("终止", "修订后仍失败就报错", "PlanAborted：状态 aborted、答复为空"),
    ("计划 vs 记忆", "计划只在本次运行；记忆跨轮", "TaskPlan 不落库（与单元19 相反）"),
]


__all__ = [
    "ANSWERER_MARK",
    "CONCEPTS",
    "EMPTY_FEEDBACK",
    "MAX_GOAL_CHARS",
    "MAX_PLAN_STEPS",
    "MAX_REPLANS",
    "MIN_GOAL_CHARS",
    "MIN_PLAN_STEPS",
    "OFFLINE_MODEL_NAME",
    "PLANNER_ALLOWED",
    "PLANNER_MARK",
    "PlanAborted",
    "PlanRun",
    "PlanStep",
    "ScriptedPlanModel",
    "STATUS_ABORTED",
    "STATUS_DONE",
    "STATUS_NO_PLAN",
    "StepFailure",
    "StepObservation",
    "TaskPlan",
    "build_plan_model",
    "execute_plan",
    "make_plan",
    "plan_step",
    "render_concepts",
    "render_plan",
    "render_result",
    "render_trajectory",
    "solve",
    "validate_goal",
]
