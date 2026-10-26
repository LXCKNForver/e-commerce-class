"""单元14 的编排层：装资料 → 建 Agent → 跑三个课堂场景 → 渲染轨迹对比。

分工与前面几个单元一样：

- 解析   `app/loaders.py`（单元12 的成果，md/pdf/csv → 同一种知识记录）
- 入库   `app/chroma_store.py`（单元14 自己的库 `.chroma_unit14/`）
- 检索   `app/retriever.py`（单元09 的检索服务，Chroma + 单元06 的中文向量模型）
- 循环   `app/agent.py`（本单元新增：决策 → 执行 → 观察 → 再决策）
- 本文件 只负责把任务卡的四段时间串成一条命令，并把结果渲染出来。

四段时间对应的动作：

    0–8 分    确认资料目录、项目标识、密钥配置（render_header）
    8–18 分   单事项问题：检索一次就够，看来源、答案、finished
    18–30 分  双事项问题：观察第二轮是否再次调用 search_knowledge
    30–40 分  max-steps 2 重跑复杂问题：区分 finished 与 max_steps，做轨迹对比

    uv run --extra embedding --extra chroma --extra parse python -m app.agent_lab
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.agent import (
    DEFAULT_MAX_STEPS,
    DeepSeekDecisionModel,
    ScriptedDecisionModel,
    build_decision_model,
    render_result,
    render_trace,
    ReActAgent,
)
from app.chroma_store import (
    PROJECT_ID,
    UNIT14_COLLECTION,
    UNIT14_COLLECTION_METADATA,
    ChromaVectorStore,
    resolve_unit14_db_path,
)
from app.data_store import PROJECT_ROOT
from app.loaders import load_materials
from app.parse_lab import EXPECTED_RECORDS, PROJECTS
from app.retriever import RetrievalService

# 任务卡写明了：电商售后组直接使用单元12 对应的实践项目资料。
PROJECT_KEY = "ecom"
KNOWLEDGE_DIR = PROJECTS[PROJECT_KEY].folder
EXPECTED_KB_RECORDS = EXPECTED_RECORDS

# 8–18 分钟：单事项问题（对应 CSV 里的 EC-FAQ-01 与 DOC-02-01）。
SINGLE_QUESTION = "普通商品签收后还能申请退货评估吗？"
# 18–30 分钟：涉及两个事项的问题——退货时限 + 质量问题材料，期望看到两次检索。
DOUBLE_QUESTION = "普通商品签收后多久还能申请退货评估？另外商品有质量问题需要准备哪些材料？"
# 30–40 分钟：轮次上限演示就用较复杂的那个问题重跑。
LIMITED_MAX_STEPS = 2

# 提交内容要求说明"为什么模型只能提出动作、实际检索必须由 Python 执行"，这张表就是答案。
WHY_PYTHON_EXECUTES = [
    ("模型给不出向量", "向量由本地中文模型算出，模型 API 返回不了 512 维向量，也拿不到库里的距离"),
    ("隔离要靠 where", "project_id 过滤在 Chroma 端执行，交给模型等于放弃项目边界"),
    ("动作必须可信", "白名单拦住未知动作；模型“提议删库”也只是个没人执行的字符串"),
    ("引用必须真实", "finish 的来源由 Python 校验，没检索到过的一律拒绝放行"),
    ("停止条件要可控", "max_steps 由 Python 计数，到点了就是未完成，不允许再来一次"),
]

SCENARIO_LABELS = {
    "single": "单事项问题（8–18 分钟）",
    "double": "双事项问题（18–30 分钟）",
    "limited": "轮次受限重跑（30–40 分钟）",
}


@dataclass
class ScenarioRun:
    """一个场景 = 一个问题 + 一个轮次上限 + 一次真实运行的结果。"""

    key: str
    label: str
    question: str
    max_steps: int
    result: object  # AgentResult；延迟标注类型避免循环 import

    @property
    def finished(self) -> bool:
        return bool(getattr(self.result, "finished", False))

    @property
    def stop_reason(self) -> str:
        return str(getattr(self.result, "stop_reason", ""))

    @property
    def search_count(self) -> int:
        return int(getattr(self.result, "search_count", 0))


@dataclass
class ActRun:
    """一次完整演练的输出；字段全部来自真实运行，渲染函数只读不写。"""

    database_path: Path
    collection_name: str
    knowledge_dir: Path
    kb_records: int
    model_name: str
    provider: str
    key_configured: bool
    scenarios: list[ScenarioRun] = field(default_factory=list)

    def scenario(self, key: str) -> ScenarioRun | None:
        for item in self.scenarios:
            if item.key == key:
                return item
        return None


def decision_model_name(model) -> str:
    if isinstance(model, DeepSeekDecisionModel):
        return f"DeepSeek {model.model}（真实调用）"
    if isinstance(model, ScriptedDecisionModel):
        return "离线脚本策略（LLM_PROVIDER=mock）"
    return type(model).__name__


def build_agent_store(database_path: Path | None = None) -> tuple[ChromaVectorStore, int]:
    """解析本组资料并写进单元14 自己的库；返回库句柄与本项目条数。

    库目录是 `.chroma_unit14/`：与单元12 演示库、单元13 评估库分开，
    各自保住自己的复现现场（这单元的库会被 Agent 反复查询）。
    """
    records, failures = load_materials(KNOWLEDGE_DIR, PROJECT_ID)
    if not records:
        raise RuntimeError(
            f"没有可入库的文本，请检查资料目录：{KNOWLEDGE_DIR}；解析失败：{failures}"
        )
    store = ChromaVectorStore(
        Path(database_path) if database_path else resolve_unit14_db_path(),
        UNIT14_COLLECTION,
        UNIT14_COLLECTION_METADATA,
    )
    store.upsert_documents(records)
    return store, store.count(PROJECT_ID)


def build_agent(
    max_steps: int = DEFAULT_MAX_STEPS,
    database_path: Path | None = None,
    decision_model=None,
) -> tuple[ReActAgent, ChromaVectorStore, int]:
    """准备好"能跑一次问答"的 Agent：装库 + 选决策模型。"""
    store, count = build_agent_store(database_path)
    model = decision_model if decision_model is not None else build_decision_model()
    agent = ReActAgent(model, RetrievalService(store), PROJECT_ID)
    return agent, store, count


def run_scenario(
    key: str,
    question: str,
    max_steps: int,
    agent: ReActAgent | None = None,
    database_path: Path | None = None,
    decision_model=None,
) -> tuple[ScenarioRun, ReActAgent, ChromaVectorStore, int]:
    """跑一个场景；agent 可复用（同一个库、同一个模型，只换问题与轮次上限）。"""
    if agent is None:
        agent, store, count = build_agent(max_steps, database_path, decision_model)
    else:
        # 复用调用方准备好的 Agent：同一个库、同一个模型，只换问题与轮次上限。
        store = getattr(agent.retriever, "store", None)
        count = store.count(PROJECT_ID) if store is not None else 0

    result = agent.run(question, max_steps)
    scenario = ScenarioRun(
        key=key,
        label=SCENARIO_LABELS.get(key, key),
        question=question,
        max_steps=max_steps,
        result=result,
    )
    return scenario, agent, store, count


def run_class(
    decision_model=None,
    database_path: Path | None = None,
    single_question: str = SINGLE_QUESTION,
    double_question: str = DOUBLE_QUESTION,
) -> ActRun:
    """按任务卡顺序跑完三个场景，复用同一个库与同一个模型，只换问题与轮次上限。"""
    model = decision_model if decision_model is not None else build_decision_model()
    agent, store, count = build_agent(DEFAULT_MAX_STEPS, database_path, model)

    run = ActRun(
        database_path=Path(database_path) if database_path else resolve_unit14_db_path(),
        collection_name=store.collection.name,
        knowledge_dir=KNOWLEDGE_DIR,
        kb_records=count,
        model_name=decision_model_name(model),
        provider=(os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower(),
        key_configured=bool(os.getenv("DEEPSEEK_API_KEY")),
    )

    plan = [
        ("single", single_question, DEFAULT_MAX_STEPS),
        ("double", double_question, DEFAULT_MAX_STEPS),
        ("limited", double_question, LIMITED_MAX_STEPS),
    ]
    for key, question, max_steps in plan:
        scenario, _agent, _store, _count = run_scenario(key, question, max_steps, agent=agent)
        run.scenarios.append(scenario)
    return run


# ------------------------------------------------------------------ 渲染


def render_header(run: ActRun) -> str:
    """0–8 分钟要确认的三件事：资料路径、项目标识、密钥配置。"""
    return "\n".join(
        [
            f"资料目录：{_relative(run.knowledge_dir)}（{run.kb_records} 条"
            + (f"，完成标准要求 {EXPECTED_KB_RECORDS} 条：一致）"
               if run.kb_records == EXPECTED_KB_RECORDS else
               f"，完成标准要求 {EXPECTED_KB_RECORDS} 条：不一致）"),
            f"项目标识：{PROJECT_ID}",
            f"数据库目录：{_relative(run.database_path)}；集合：{run.collection_name}",
            f"决策模型：{run.model_name}；密钥已配置：{run.key_configured}",
        ]
    )


def render_scenario(scenario: ScenarioRun) -> str:
    """一个场景的完整证据：结果摘要 + 逐轮轨迹。"""
    blocks = [
        f"--- {scenario.label}：max_steps={scenario.max_steps}",
        render_result(scenario.result),
        "",
        render_trace(scenario.result),
    ]
    return "\n".join(blocks)


def render_comparison(run: ActRun) -> str:
    """任务卡 30–40 分钟要的轨迹对比：双事项 4 轮 vs 同一题限 2 轮。"""
    double = run.scenario("double")
    limited = run.scenario("limited")
    lines = ["轨迹对比：同一道题，只改最大轮次", "场景\t最大轮次\t实际轮次\t检索次数\t停止原因\t是否答复"]
    for scenario in (double, limited):
        if scenario is None:
            continue
        answered = "是" if scenario.finished else "否（未完成，不予采纳）"
        lines.append(
            f"{scenario.label}\t{scenario.max_steps}\t{scenario.result.steps}\t"
            f"{scenario.search_count}\t{scenario.stop_reason}\t{answered}"
        )
    if double is not None and limited is not None:
        lines.append("")
        lines.append(
            f"结论：max_steps 从 {double.max_steps} 降到 {limited.max_steps} 后，"
            f"停止原因从 {double.stop_reason} 变成 {limited.stop_reason}；"
            "后者必须如实标记为“尚未完成”，不得把半途观察到的片段当答案交付。"
        )
        lines.append("限制轮次不会让模型更谨慎，只会让它更早停下——这是成本约束，不是质量信号。")
    return "\n".join(lines)


def render_why_python() -> str:
    return "\n".join(
        ["为什么模型只能提出动作、检索必须由 Python 执行，共 "
         f"{len(WHY_PYTHON_EXECUTES)} 条："]
        + [f"  {index}. {title}——{detail}"
           for index, (title, detail) in enumerate(WHY_PYTHON_EXECUTES, start=1)]
    )


def render_run(run: ActRun) -> str:
    """一次完整演练的输出，与 main.py 的 /act demo 逐字一致。"""
    blocks = ["=== 单元14：ReAct Agent ===", render_header(run), ""]
    for scenario in run.scenarios:
        blocks.extend([render_scenario(scenario), ""])
    blocks.append(render_comparison(run))
    blocks.append("")
    blocks.append(render_why_python())
    return "\n".join(blocks)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "DOUBLE_QUESTION",
    "EXPECTED_KB_RECORDS",
    "KNOWLEDGE_DIR",
    "LIMITED_MAX_STEPS",
    "SINGLE_QUESTION",
    "build_agent",
    "build_agent_store",
    "render_comparison",
    "render_header",
    "render_result",
    "render_run",
    "render_scenario",
    "render_trace",
    "render_why_python",
    "run_class",
    "run_scenario",
]


if __name__ == "__main__":  # pragma: no cover - 手动跑演示时用
    print(render_run(run_class()))
