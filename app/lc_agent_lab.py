"""单元17 的编排层：装本组工具 → 接进 LangChain 执行器 → 跑四节课场景 → 对照旧循环。

分工与前面几个单元一样，本文件只负责把任务卡的四节课串成一条命令：

    第1节  认识执行器的部件，在现有项目里标出可接入的工具集与资料来源
    第2节  确定入口与工具集，先跑通最小业务场景（按编号查标准答复）
    第3节  接入执行器与步骤追踪：连续调用两步，并演示工具错误定位到第几步
    第4节  验证工具选择、错误与步数上限；同一问题对照单元15 手写的循环

    数据侧（一个字没换）：本组 3md + 1pdf + 1csv → Chroma，FAQ CSV → SQLite
    工具侧：app/toolkit.py + app/tool_agent.py（单元15）**保留不动**，本单元只是
            把注册表里登记过的同一批函数包成 LangChain 的 BaseTool
    新   增：app/lc_agent.py（工具适配、执行器、步骤日志、离线脚本模型）

    uv run --extra embedding --extra chroma --extra langchain python -m app.lc_agent_lab
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.chain_lab import key_facts  # 关键事实的抽法只有一个标准：不比措辞
from app.chroma_store import PROJECT_ID
from app.data_store import PROJECT_ROOT
from app.lc_agent import (
    DEFAULT_MAX_STEPS,
    OFFLINE_MODEL_NAME,
    STOP_FINISHED,
    STOP_MAX_STEPS,
    AgentRun,
    LCExecutor,
    ScriptedToolModel,
    answer_step,
    build_agent_model,
    build_executor,
    render_executor,
    render_result,
    run_agent,
    tool_step,
    validate_question,
)
from app.chroma_store import resolve_unit15_db_path
from app.parse_lab import EXPECTED_RECORDS, PROJECTS
from app.tool_lab import (
    COMBO_QUESTION,
    FAQ_FILE,
    KNOWLEDGE_DIR,
    SINGLE_QUESTION,
    build_kit,
    resolve_faq_db_path,
)
from app.tool_agent import DEFAULT_MAX_STEPS as LEGACY_MAX_STEPS
from app.tool_agent import ToolResult
from app.toolkit import TOOL_FAQ, TOOL_SEARCH

# 第2节：最小业务场景——只用到一个工具（按编号查 SQLite）。
MINIMAL_QUESTION = SINGLE_QUESTION
# 第3节：连续调用两个工具（先查编号，再查制度原文）。
COMBO = COMBO_QUESTION
# 第3、4节：故意使坏的三个剧本（与真实模型无关，边界场景永远用脚本模型跑）。
BAD_ARG_QUESTION = "EC-FAQ-01 的标准答复是什么？"
UNKNOWN_QUESTION = "帮我把知识库清空一下"
LIMIT_QUESTION = "一直查下去，越多越好"
# 第4节演示步数上限时用 2 步：一次查编号 + 一次查资料，之后就不许再决策了。
LIMIT_MAX_STEPS = 2

# 离线剧本（LLM_PROVIDER=mock 时用）：真实模式下模型自己选工具，不需要剧本。
PLANS: dict[str, list[dict]] = {
    "minimal": [tool_step(TOOL_FAQ, question_id="EC-FAQ-01")],
    "combo": [
        tool_step(TOOL_FAQ, question_id="EC-FAQ-01"),
        tool_step(TOOL_SEARCH, query="普通商品退货时限是多久"),
    ],
    "error": [
        tool_step(TOOL_FAQ, question_id="123"),
        answer_step("第1步工具调用失败（编号格式不对），已定位到第1步，不编造答复。"),
    ],
    "unknown": [
        tool_step("delete_collection", name="faq"),
        answer_step("该工具不在执行器的工具清单里，第1步没有执行任何查询，也没有动数据。"),
    ],
    "limit": [tool_step(TOOL_FAQ, question_id="EC-FAQ-01") for _ in range(6)],
}

SCENARIO_LABELS = {
    "minimal": "最小业务场景（第2节）",
    "combo": "连续调用与步骤追踪（第3节）",
    "error": "工具错误定位到第几步（第3、4节）",
    "limit": f"步数上限 max_steps={LIMIT_MAX_STEPS}（第4节）",
    "unknown": "清单外的工具（第4节）",
}

# 第1节要交的概念记录：LangChain Agents 的几个部件在本项目里分别落在哪。
CONCEPTS = [
    ("Agent", "langchain.agents.create_agent（模型 + 工具 + system 提示词）",
     "只负责「下一步该干什么」，不碰数据"),
    ("工具", "单元15 注册表的三项只读能力，包成 BaseTool",
     "唯一能碰到数据的入口，名字与参数都来自注册表"),
    ("执行器", "create_agent 返回的状态图（决策 → 执行 → 观察 → 再决策）",
     "替我们跑循环，本单元不再手写 while"),
    ("步骤", "messages：AIMessage(tool_calls) + ToolMessage(观察)",
     "每一步的提议与结果都留在消息里，可回放"),
    ("上限", "ModelCallLimitMiddleware（run_limit = max_steps）",
     "模型决策次数触顶就收尾，且不产出业务答案"),
    ("日志", "ToolCallLog：输入、输出/失败原因、耗时",
     "工具错误能定位到第几步，而不是一句「调用失败」"),
]

# 第1节要交的项目问题清单：每条都对应本单元的一个验证场景。
QUESTION_LIST = [
    ("模型能不能绕过工具直接编答案？",
     "system 提示词限定只依据工具返回；工具清单里没有的名字执行器不认"
     f"（第4节「清单外的工具」，max_steps={DEFAULT_MAX_STEPS}）"),
    ("工具出错时能不能说清是哪一步？",
     "步骤表给出步号、工具、参数、失败原因（第3节「工具错误定位到第几步」）"),
    ("步数上限是不是真的生效？",
     f"max_steps={LIMIT_MAX_STEPS} 时执行器触顶收尾，回答为空（第4节）"),
    ("换成框架的执行器，行为会不会变？",
     "同一问题对照单元15 手写的循环，比用到的工具与关键事实（第4节）"),
]


@dataclass
class AgentScenario:
    """一个场景 = 一个问题 + 一次真实执行的结果。"""

    key: str
    label: str
    question: str
    max_steps: int
    plan: list[dict] = field(default_factory=list)
    run: AgentRun | None = None
    error: str = ""


@dataclass
class Comparison:
    """同题对照：单元17 执行器 vs 单元15 手写的循环。"""

    question: str
    executor_run: AgentRun
    legacy: ToolResult
    executor_facts: list[str] = field(default_factory=list)
    legacy_facts: list[str] = field(default_factory=list)
    verdict: str = ""


@dataclass
class ExecutorLabRun:
    """一次完整演练的输出；字段全部来自真实执行，渲染函数只读不写。"""

    knowledge_dir: Path
    kb_records: int
    faq_file: Path
    faq_db_path: Path
    faq_rows: int
    project_id: str
    collection_name: str
    database_path: Path
    model_name: str
    provider: str
    key_configured: bool
    max_steps: int
    executor: LCExecutor | None = None
    registry: object | None = None
    scenarios: list[AgentScenario] = field(default_factory=list)
    comparison: Comparison | None = None
    invalid_rows: list[tuple[str, str]] = field(default_factory=list)

    def scenario(self, key: str) -> AgentScenario | None:
        for item in self.scenarios:
            if item.key == key:
                return item
        return None


# ---------------------------------------------------------------- ① 装配


def build_stack(
    model=None,
    max_steps: int = DEFAULT_MAX_STEPS,
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
    decision_model=None,
    plan: list[dict] | None = None,
):
    """准备好"能跑一次执行器"的四件套：向量库 + FAQ 库 + 工具 + 执行器。

    `build_kit()` 是单元15 的装配函数，本单元直接复用它——同一批资料、
    同一份注册表，换的只是"谁来跑这个循环"。
    """
    legacy_agent, registry, store, kb_count, faq_rows = build_kit(
        database_path, faq_db_path, decision_model
    )
    chat_model = model if model is not None else build_agent_model(plan)
    executor = build_executor(registry, chat_model, max_steps)
    return executor, legacy_agent, registry, store, kb_count, faq_rows


def _is_scripted(executor: LCExecutor) -> bool:
    return isinstance(executor.model, ScriptedToolModel)


def _with(executor: LCExecutor, plan: list[dict] | None = None, max_steps: int | None = None):
    """换剧本或换上限时重建执行器：工具与日志沿用同一份，资料口径不会变。"""
    if plan is None and (max_steps is None or max_steps == executor.max_steps):
        return executor
    model = ScriptedToolModel(plan=plan or []) if plan is not None else executor.model
    return build_executor(
        executor.registry, model, max_steps or executor.max_steps, executor.system_prompt, executor.log
    )


# ---------------------------------------------------------------- ② 运行


def run_scenario(
    key: str,
    question: str,
    executor: LCExecutor,
    plan: list[dict] | None = None,
    max_steps: int | None = None,
    scripted: bool = False,
) -> AgentScenario:
    """跑一个场景。

    `scripted=True` 的场景（工具报错、清单外的工具、步数上限）永远用剧本模型跑：
    这些是"故意使坏的模型"，要的是可复现的异常路径，与真实模型好不好无关。
    """
    actual_plan = plan
    if actual_plan is None and scripted is False and _is_scripted(executor):
        actual_plan = PLANS.get(key)
    target = _with(executor, actual_plan, max_steps)
    scenario = AgentScenario(
        key=key,
        label=SCENARIO_LABELS.get(key, key),
        question=question,
        max_steps=target.max_steps,
        plan=list(actual_plan or []),
    )
    try:
        scenario.run = run_agent(target, question)
    except ValueError as error:
        scenario.error = f"输入未通过校验：{error}"
    except RuntimeError as error:  # pragma: no cover - 依赖缺失时才走这里
        scenario.error = f"执行器未运行：{error}"
    return scenario


def run_invalid() -> list[tuple[str, str]]:
    """无效输入：守门在跑执行器之前，模型与工具都不会被触发。"""
    cases = [
        ("空字符串", ""),
        ("只有空白", "   "),
        ("长度不足", "退"),
        ("不是字符串", 123),
    ]
    rows: list[tuple[str, str]] = []
    for label, raw in cases:
        try:
            validate_question(raw)
            rows.append((f"{label}（{raw!r}）", "竟然通过了校验（异常！）"))
        except ValueError as error:
            rows.append((f"{label}（{raw!r}）", f"已拒绝：{error}"))
    return rows


def compare_with_legacy(
    question: str, executor: LCExecutor, legacy_agent, plan: list[dict] | None = None
) -> Comparison:
    """同一问题、同一份资料、同一批工具：执行器 vs 单元15 手写的循环。"""
    actual_plan = plan
    if actual_plan is None and _is_scripted(executor):
        actual_plan = PLANS["minimal"]
    target = _with(executor, actual_plan)
    run = run_agent(target, question)
    legacy = legacy_agent.run(question, LEGACY_MAX_STEPS)

    executor_facts = key_facts(run.answer, *[step.observation for step in run.steps])
    legacy_facts = key_facts(legacy.answer, *legacy.sources)
    common = [item for item in executor_facts if item in legacy_facts]

    if not run.finished:
        verdict = "执行器触顶未完成：步数上限生效，不能拿它当答案用"
    elif common:
        verdict = f"关键事实一致（共同依据：{'、'.join(common)}）"
    elif set(step.tool for step in run.steps) & set(legacy.distinct_tools):
        verdict = "用到同一个工具，但两边答案里没有相同的数字/编号：措辞差异可接受，数字差异要人工核对"
    else:
        verdict = "两条路径调用的工具不同：需要人工确认哪条才是本问题的依据"
    return Comparison(
        question=question,
        executor_run=run,
        legacy=legacy,
        executor_facts=executor_facts,
        legacy_facts=legacy_facts,
        verdict=verdict,
    )


def run_class(
    model=None,
    max_steps: int = DEFAULT_MAX_STEPS,
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
    decision_model=None,
    minimal_question: str = MINIMAL_QUESTION,
    combo_question: str = COMBO,
) -> ExecutorLabRun:
    """按任务卡顺序跑完四节课；model / decision_model 可显式传入以便离线测试。"""
    executor, legacy_agent, registry, store, kb_count, faq_rows = build_stack(
        model, max_steps, database_path, faq_db_path, decision_model
    )
    run = ExecutorLabRun(
        knowledge_dir=KNOWLEDGE_DIR,
        kb_records=kb_count,
        faq_file=FAQ_FILE,
        faq_db_path=faq_db_path or resolve_faq_db_path(),
        faq_rows=faq_rows,
        project_id=PROJECT_ID,
        collection_name=store.collection.name,
        database_path=Path(database_path) if database_path else resolve_unit15_db_path(),
        model_name=_model_name(executor.model),
        provider=(os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower(),
        key_configured=bool(os.getenv("DEEPSEEK_API_KEY")),
        max_steps=executor.max_steps,
        executor=executor,
        registry=registry,
    )
    run.scenarios.append(run_scenario("minimal", minimal_question, executor))
    run.scenarios.append(run_scenario("combo", combo_question, executor))
    run.scenarios.append(
        run_scenario("error", BAD_ARG_QUESTION, executor, plan=PLANS["error"], scripted=True)
    )
    run.scenarios.append(
        run_scenario(
            "limit", LIMIT_QUESTION, executor, plan=PLANS["limit"],
            max_steps=LIMIT_MAX_STEPS, scripted=True,
        )
    )
    run.scenarios.append(
        run_scenario("unknown", UNKNOWN_QUESTION, executor, plan=PLANS["unknown"], scripted=True)
    )
    run.invalid_rows = run_invalid()
    run.comparison = compare_with_legacy(minimal_question, executor, legacy_agent)
    return run


def _model_name(model: object) -> str:
    if isinstance(model, ScriptedToolModel):
        return f"{OFFLINE_MODEL_NAME}（离线脚本，不联网）"
    name = getattr(model, "model_name", None) or getattr(model, "model", "")
    return f"ChatDeepSeek {name}（真实调用）" if name else type(model).__name__


# ---------------------------------------------------------------- ③ 渲染


def render_header(run: ExecutorLabRun) -> str:
    """第1节要确认的三件事：资料、项目标识、模型与密钥。"""
    expected = (
        "与单元12 一致" if run.kb_records == EXPECTED_RECORDS
        else f"单元12 是 {EXPECTED_RECORDS} 条：不一致"
    )
    return "\n".join([
        f"知识资料目录：{_relative(run.knowledge_dir)}（{run.kb_records} 条，{expected}）",
        f"FAQ 原始文件：{_relative(run.faq_file)}（同步进 {_relative(run.faq_db_path)}，"
        f"库内 {run.faq_rows} 行）",
        f"项目标识：{run.project_id}；向量集合：{run.collection_name}（{_relative(run.database_path)}）",
        f"模型：{run.model_name}；密钥已配置：{run.key_configured}；"
        f"步数上限：{run.max_steps}（模型决策次数）",
    ])


def render_concepts() -> str:
    """第1节要交的概念记录：每个部件在本项目里落在哪、管什么。"""
    lines = ["LangChain Agents 的部件与本项目的对应（第1节）：", "部件\t本组落点\t作用"]
    for name, location, duty in CONCEPTS:
        lines.append(f"{name}\t{location}\t{duty}")
    lines.append("")
    lines.append("项目问题清单（每条都在后面的场景里验证）：")
    for question, how in QUESTION_LIST:
        lines.append(f"  · {question} → {how}")
    return "\n".join(lines)


def render_tool_binding(registry, executor: LCExecutor) -> str:
    """第2节要交的工具清单：注册表里的三项，全部接进了执行器。"""
    lines = ["接入执行器的工具（名字、说明、参数全部来自单元15 的注册表）：",
             "工具\t参数\t数据来源\t接线"]
    sources = {
        TOOL_SEARCH: "Chroma 向量库（单元09 检索）",
        TOOL_FAQ: "SQLite 只读库（本组 FAQ CSV）",
    }
    for name in registry.names():
        spec = registry.spec(name)
        lines.append(
            f"{spec.name}\t{', '.join(spec.parameters)}\t"
            f"{sources.get(name, '资料原文 + Python 日期加法')}\t"
            f"{'已接入' if name in executor.tool_names else '未接入（异常！）'}"
        )
    lines.append(
        "工具描述里没有本地路径、没有 SQL、没有可执行代码——模型只能挑名字、填参数，"
        "真正执行的仍是注册时登记的那个 Python 函数。"
    )
    return "\n".join(lines)


def render_scenario(scenario: AgentScenario) -> str:
    if scenario.run is None:
        return f"--- {scenario.label}\n问题：{scenario.question}\n{scenario.error}"
    return "\n".join([f"--- {scenario.label}", render_result(scenario.run)])


def render_comparison(comparison: Comparison) -> str:
    """第4节要交的对照：同一问题、同一批工具，比工具与硬事实。"""
    run = comparison.executor_run
    legacy = comparison.legacy
    return "\n".join([
        "同题对照（同一问题、同一份资料、同一批工具）：",
        "项\t单元17 执行器\t单元15 手写循环",
        f"步数上限\t{run.max_steps}\t{legacy.max_steps}",
        f"调用的工具\t{'、'.join(step.tool for step in run.steps) or '（无）'}\t"
        + ("、".join(legacy.distinct_tools) or "（无）"),
        f"停止原因\t{run.stop_reason}\t{legacy.stop_reason}",
        f"关键事实\t{'、'.join(comparison.executor_facts) or '（无）'}\t"
        + ("、".join(comparison.legacy_facts) or "（无）"),
        "",
        f"结论：{comparison.verdict}",
        "说明：对照的是调用的工具与数字/编号这类硬事实；模型措辞不要求一致。",
    ])


def render_invalid(rows: list[tuple[str, str]]) -> str:
    lines = ["无效输入（守门在跑执行器之前，模型与工具都不会被触发）：", "输入\t结果"]
    for shown, outcome in rows:
        lines.append(f"{shown}\t{outcome}")
    return "\n".join(lines)


def render_run(run: ExecutorLabRun) -> str:
    """一次完整演练的输出，与 main.py 的 /lcagent demo 逐字一致。"""
    blocks = ["=== 单元17：LangChain Agents（执行器 + 工具接入 + 步骤日志）===", render_header(run), ""]
    if run.executor is not None:
        blocks.extend([render_executor(run.executor), ""])
        if run.registry is not None:
            blocks.extend([render_tool_binding(run.registry, run.executor), ""])
        blocks.extend([render_concepts(), ""])
    for key in ("minimal", "combo", "error", "limit", "unknown"):
        scenario = run.scenario(key)
        if scenario is not None:
            blocks.extend([render_scenario(scenario), ""])
    if run.invalid_rows:
        blocks.extend([render_invalid(run.invalid_rows), ""])
    if run.comparison is not None:
        blocks.extend([render_comparison(run.comparison), ""])
    blocks.append(
        "换了执行器，没换工具也没换资料：三项只读能力还是单元15 注册表里那三个，"
        "资料还是单元12 解析的那 7 条；变的只是「循环由谁跑」——从手写 while 换成了框架。"
    )
    return "\n".join(blocks)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "BAD_ARG_QUESTION",
    "COMBO",
    "CONCEPTS",
    "LIMIT_MAX_STEPS",
    "LIMIT_QUESTION",
    "MINIMAL_QUESTION",
    "PLANS",
    "QUESTION_LIST",
    "UNKNOWN_QUESTION",
    "AgentScenario",
    "Comparison",
    "ExecutorLabRun",
    "build_stack",
    "compare_with_legacy",
    "render_comparison",
    "render_concepts",
    "render_header",
    "render_invalid",
    "render_run",
    "render_scenario",
    "render_tool_binding",
    "run_class",
    "run_invalid",
    "run_scenario",
]


if __name__ == "__main__":  # pragma: no cover - 手动跑演示时用
    print(render_run(run_class()))
