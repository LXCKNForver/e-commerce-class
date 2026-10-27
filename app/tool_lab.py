"""单元15 的编排层：装两份数据源 → 注册工具 → 跑四个课堂场景 → 渲染证据。

分工与前面几个单元一样，本文件只负责把任务卡的四节课串成一条命令：

    0–15 分   区分知识资料与结构化记录，确认数据源与工具清单
    15–30 分  注册 2—3 项只读工具，跑检索与 SQLite FAQ 查询
    30–45 分  组合问题：观察连续调用，检查来源记录
    45–60 分  验证单工具、连续调用、拒绝路径，说明计算工具是否适用

    数据侧：
      非结构化  本组资料 → Chroma（单元12 解析 + 单元09 检索）
      结构化    同一份 FAQ CSV → SQLite（app/faq_db.py，只读 + 参数化 SQL）
    能力侧：
      app/toolkit.py    三项工具 + 白名单 + 注册表（name/description/parameters/execute）
      app/tool_agent.py 多工具 ReAct 循环（决策 → 放行 → 执行 → 观察 → 再决策）

    uv run --extra embedding --extra chroma --extra parse python -m app.tool_lab
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.chroma_store import (
    PROJECT_ID,
    UNIT15_COLLECTION,
    UNIT15_COLLECTION_METADATA,
    ChromaVectorStore,
    resolve_unit15_db_path,
)
from app.data_store import PROJECT_ROOT
from app.faq_db import count_faq, prepare_faq_db
from app.loaders import load_materials
from app.parse_lab import EXPECTED_RECORDS, PROJECTS
from app.retriever import RetrievalService
from app.tool_agent import (
    DEFAULT_MAX_STEPS,
    DeepSeekToolModel,
    ScriptedRebelModel,
    ScriptedToolModel,
    ToolAgent,
    build_tool_decision_model,
    render_result,
    render_trace,
)
from app.toolkit import TOOL_FAQ, render_rejections, render_tool_list, build_registry

# 任务卡写明了：电商售后组直接使用单元12 对应的实践项目资料。
PROJECT_KEY = "ecom"
KNOWLEDGE_DIR = PROJECTS[PROJECT_KEY].folder
# 同一份原始 CSV 有两条去路：逐行进向量库、整表进 SQLite。
FAQ_FILE = KNOWLEDGE_DIR / "常见售后问题.csv"
EXPECTED_KB_RECORDS = EXPECTED_RECORDS
EXPECTED_FAQ_ROWS = 3
DEFAULT_FAQ_DB_PATH = PROJECT_ROOT / ".sqlite_unit15" / "faq.db"

# 15–30 分钟：单工具问题（只用到结构化查询）。
SINGLE_QUESTION = "EC-FAQ-01 的标准答复是什么？"
# 30–45 分钟：组合问题——编号答复 + 制度原文，期望连续调用两项不同工具。
COMBO_QUESTION = "EC-FAQ-01 的标准答复是什么？另外普通商品退货时限是多久？"
# 45–60 分钟（选做）：计算问题——先查到时限，再由 Python 算截止日期。
CALC_QUESTION = "2026年12月1日签收的普通商品，退货评估最晚到哪天？"
# 拒绝路径的两个剧本：一个撞工具白名单，一个撞参数格式。
REBEL_UNKNOWN = ("delete_collection", {"name": "faq"}, "按剧本请求删库")
REBEL_BAD_ARG = (TOOL_FAQ, {"question_id": "123"}, "按剧本乱填编号")

SCENARIO_LABELS = {
    "single": "单工具（15–30 分钟）",
    "combo": "连续调用（30–45 分钟）",
    "calc": "计算工具（45–60 分钟，选做）",
    "reject": "拒绝路径（45–60 分钟）",
}

# 第1节课要讲清的两类数据：一个是"靠相似"，一个是"按编号"。
DATA_SOURCES = [
    (
        "非结构化",
        "本组 3 篇 md + 1 份 pdf + 1 份 csv",
        "Chroma 向量库",
        "search_knowledge",
        "按相似度返回 Top-3 片段，适合制度流程类提问",
    ),
    (
        "结构化",
        f"{FAQ_FILE.name}（{EXPECTED_FAQ_ROWS} 行）",
        "SQLite 只读库",
        "lookup_faq",
        "按编号精确命中一行，答复是固定的标准答复",
    ),
]

# 第4节课要交的取舍说明：本组有真实数值，所以算得动。
CALC_NOTE = [
    ("本组有没有数值", "有：DOC-02-01 写着“签收后7天内”，CSV 里没有金额/时长字段"),
    ("算什么", "签收日期 + 7 天 = 退货评估截止日期（纯日期加法）"),
    ("天数从哪来", "必须由 search_knowledge 先查到资料原文，计算工具再核对原文里确有“N天”"),
    ("谁来算", "Python 的 datetime 做加法；模型只提供签收日期，禁止口算日期"),
    ("算不出来怎么办", "资料里查不到时限就直接回答资料不足，不做估算、不补默认值"),
    ("不适用会怎样", "若本组只有文本没有数值（如纯政策问答），就不做计算工具，照实说明即可"),
]


@dataclass
class ToolScenario:
    """一个场景 = 一个问题 + 一个轮次上限 + 一或两次真实运行的结果。"""

    key: str
    label: str
    question: str
    max_steps: int
    results: list[object] = field(default_factory=list)

    @property
    def result(self):
        return self.results[0] if self.results else None

    @property
    def finished(self) -> bool:
        return bool(self.result is not None and getattr(self.result, "finished", False))

    @property
    def stop_reasons(self) -> list[str]:
        return [str(getattr(item, "stop_reason", "")) for item in self.results]


@dataclass
class ToolKitRun:
    """一次完整演练的输出；字段全部来自真实运行，渲染函数只读不写。"""

    database_path: Path
    collection_name: str
    knowledge_dir: Path
    kb_records: int
    faq_file: Path
    faq_db_path: Path
    faq_rows: int
    model_name: str
    provider: str
    key_configured: bool
    registry: object | None = None  # 工具注册表（渲染工具清单与拒绝清单时用）
    scenarios: list[ToolScenario] = field(default_factory=list)

    def scenario(self, key: str) -> ToolScenario | None:
        for item in self.scenarios:
            if item.key == key:
                return item
        return None


def decision_model_name(model) -> str:
    if isinstance(model, DeepSeekToolModel):
        return f"DeepSeek {model.model}（真实调用）"
    if isinstance(model, ScriptedToolModel):
        return "离线脚本策略（LLM_PROVIDER=mock）"
    return type(model).__name__


def resolve_faq_db_path() -> Path:
    """单元15 的 FAQ 库路径：默认 `.sqlite_unit15/faq.db`，可用 SQLITE_UNIT15_DB_DIR 覆盖。"""
    override = os.environ.get("SQLITE_UNIT15_DB_DIR", "").strip()
    return Path(override) if override else DEFAULT_FAQ_DB_PATH


def build_faq_db(faq_db_path: Path | None = None) -> tuple[Path, int]:
    """把本组 FAQ CSV 同步进 SQLite（不改原文件），返回库路径与行数。"""
    target = Path(faq_db_path) if faq_db_path else resolve_faq_db_path()
    prepare_faq_db(FAQ_FILE, target)
    return target, count_faq(target)


def build_kit(
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
    decision_model=None,
):
    """准备好"能跑一次问答"的三件套：向量库 + FAQ 库 + 工具 Agent。"""
    records, failures = load_materials(KNOWLEDGE_DIR, PROJECT_ID)
    if not records:
        raise RuntimeError(
            f"没有可入库的资料，请检查资料目录：{KNOWLEDGE_DIR}；解析失败：{failures}"
        )
    store = ChromaVectorStore(
        Path(database_path) if database_path else resolve_unit15_db_path(),
        UNIT15_COLLECTION,
        UNIT15_COLLECTION_METADATA,
    )
    store.upsert_documents(records)
    target, rows = build_faq_db(faq_db_path)
    registry = build_registry(
        RetrievalService(store), PROJECT_ID, FAQ_FILE, target, KNOWLEDGE_DIR
    )
    model = decision_model if decision_model is not None else build_tool_decision_model(registry)
    agent = ToolAgent(model, registry)
    return agent, registry, store, store.count(PROJECT_ID), rows


def run_scenario(
    key: str,
    question: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    agent=None,
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
    decision_model=None,
) -> tuple[ToolScenario, object, object]:
    """跑一个场景；agent 可复用（同一批数据源，只换问题）。"""
    if agent is None:
        agent, _registry, store, _count, _rows = build_kit(
            database_path, faq_db_path, decision_model
        )
    else:
        store = None
    result = agent.run(question, max_steps)
    scenario = ToolScenario(
        key=key,
        label=SCENARIO_LABELS.get(key, key),
        question=question,
        max_steps=max_steps,
        results=[result],
    )
    return scenario, agent, store


def run_rejections(agent, max_steps: int = DEFAULT_MAX_STEPS) -> ToolScenario:
    """拒绝路径：两个"坏模型"各跑一次，看循环是不是停在白名单前。"""
    scenario = ToolScenario(
        key="reject",
        label=SCENARIO_LABELS["reject"],
        question="（无用户问题：两个剧本分别提出未授权请求）",
        max_steps=max_steps,
    )
    original = agent.decision_model
    for action, arguments, reason in (REBEL_UNKNOWN, REBEL_BAD_ARG):
        agent.decision_model = ScriptedRebelModel(action, arguments, reason)
        scenario.results.append(agent.run("请帮我处理一下这个售后问题", max_steps))
    agent.decision_model = original
    return scenario


def run_class(
    decision_model=None,
    database_path: Path | None = None,
    faq_db_path: Path | None = None,
    single_question: str = SINGLE_QUESTION,
    combo_question: str = COMBO_QUESTION,
    calc_question: str = CALC_QUESTION,
) -> ToolKitRun:
    """按任务卡顺序跑完四个场景，复用同一批数据源与同一个决策模型。"""
    agent, registry, store, kb_count, faq_rows = build_kit(
        database_path, faq_db_path, decision_model
    )
    model = agent.decision_model
    run = ToolKitRun(
        database_path=Path(database_path) if database_path else resolve_unit15_db_path(),
        collection_name=store.collection.name,
        knowledge_dir=KNOWLEDGE_DIR,
        kb_records=kb_count,
        faq_file=FAQ_FILE,
        faq_db_path=Path(faq_db_path) if faq_db_path else resolve_faq_db_path(),
        faq_rows=faq_rows,
        model_name=decision_model_name(model),
        provider=(os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower(),
        key_configured=bool(os.getenv("DEEPSEEK_API_KEY")),
        registry=registry,
    )

    for key, question in (("single", single_question), ("combo", combo_question), ("calc", calc_question)):
        scenario, _agent, _store = run_scenario(key, question, DEFAULT_MAX_STEPS, agent=agent)
        run.scenarios.append(scenario)
    run.scenarios.append(run_rejections(agent))
    return run


# ------------------------------------------------------------------ 渲染


def render_header(run: ToolKitRun) -> str:
    """0–15 分钟要确认的三件事：数据源、项目标识、模型与密钥。"""
    return "\n".join(
        [
            f"知识资料目录：{_relative(run.knowledge_dir)}（{run.kb_records} 条"
            + ("，与单元12 一致）" if run.kb_records == EXPECTED_KB_RECORDS
               else f"，单元12 是 {EXPECTED_KB_RECORDS} 条：不一致）"),
            f"FAQ 原始文件：{_relative(run.faq_file)}（同步进 {_relative(run.faq_db_path)}，"
            + f"库内 {run.faq_rows} 行"
            + ("，与原始 CSV 一致）" if run.faq_rows == EXPECTED_FAQ_ROWS
               else f"，原始 CSV 是 {EXPECTED_FAQ_ROWS} 行：不一致）"),
            f"项目标识：{PROJECT_ID}；向量集合：{run.collection_name}（{_relative(run.database_path)}）",
            f"决策模型：{run.model_name}；密钥已配置：{run.key_configured}",
        ]
    )


def render_data_sources() -> str:
    """第1节课要交的清单：两类数据源各自装在哪、由哪个工具读。"""
    lines = ["数据源与工具对应关系（同一份 CSV 有两条去路）：", "类型\t原始资料\t落点\t读取工具\t特点"]
    for kind, origin, storage, tool, note in DATA_SOURCES:
        lines.append(f"{kind}\t{origin}\t{storage}\t{tool}\t{note}")
    lines.append(
        "非结构化靠相似度猜最相关的片段，结构化按编号取唯一确定的一行；"
        "两路返回的都是带来源的数据，Agent 依据它们作答。"
    )
    return "\n".join(lines)


def render_scenario(scenario: ToolScenario) -> str:
    """一个场景的完整证据：结果摘要 + 逐轮轨迹。"""
    blocks = [f"--- {scenario.label}：max_steps={scenario.max_steps}"]
    for result in scenario.results:
        blocks.extend([render_result(result), "", render_trace(result), ""])
    return "\n".join(blocks).rstrip()


def render_calc_note() -> str:
    """第4节课要求的取舍说明：本组为什么能做计算工具、它是怎么保证不猜的。"""
    lines = ["计算工具是否适用（选做）："]
    for title, detail in CALC_NOTE:
        lines.append(f"  · {title}：{detail}")
    lines.append(
        "结论：本组有真实数值（7 天时限 + 签收日期），计算工具**适用**；"
        "若换成只有政策文本的小组，就不做，并在文档里写明不适用。"
    )
    return "\n".join(lines)


def render_run(run: ToolKitRun) -> str:
    """一次完整演练的输出，与 main.py 的 /tools demo 逐字一致。"""
    blocks = ["=== 单元15：Agent 工具集 ===", render_header(run), "", render_data_sources(), ""]
    for key in ("single", "combo", "reject"):
        scenario = run.scenario(key)
        if scenario is not None:
            blocks.extend([render_scenario(scenario), ""])
    registry = getattr(run, "registry", None)
    if registry is not None:
        blocks.extend([render_rejections(registry), "", render_tool_list(registry), ""])
    calc = run.scenario("calc")
    if calc is not None:
        blocks.extend([render_scenario(calc), ""])
    blocks.append(render_calc_note())
    return "\n".join(blocks)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "CALC_QUESTION",
    "COMBO_QUESTION",
    "DATA_SOURCES",
    "EXPECTED_FAQ_ROWS",
    "EXPECTED_KB_RECORDS",
    "FAQ_FILE",
    "KNOWLEDGE_DIR",
    "SINGLE_QUESTION",
    "ToolKitRun",
    "ToolScenario",
    "build_faq_db",
    "build_kit",
    "render_calc_note",
    "render_data_sources",
    "render_header",
    "render_result",
    "render_run",
    "render_scenario",
    "render_tool_list",
    "render_trace",
    "resolve_faq_db_path",
    "run_class",
    "run_rejections",
    "run_scenario",
]


if __name__ == "__main__":  # pragma: no cover - 手动跑演示时用
    print(render_run(run_class()))
