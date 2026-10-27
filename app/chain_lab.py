"""单元16 的编排层：装本组资料 → 组 LCEL 链 → 跑四节课场景 → 同题对照旧 Agent。

分工与前面几个单元一样，本文件只负责把任务卡的四节课串成一条命令：

    第1节  标出可复用的检索函数与真实资料来源
    第2节  RunnableLambda 接入检索，assign 保留问题、片段和来源
    第3节  Prompt + 模型 + Parser 组合成链，跑一次真实问答
    第4节  同一问题对照单元15 的 ToolAgent，核对关键事实与异常场景

    数据侧（一点没换）：本组 3md + 1pdf + 1csv → Chroma（单元12 解析 + 单元09 检索）
    链    侧：app/lc_chain.py（六段 LCEL）
    对照侧：app/toolkit.py + app/tool_agent.py（原 Agent 保留，不被覆盖）

    uv run --extra embedding --extra chroma --extra langchain python -m app.chain_lab
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.chroma_store import (
    PROJECT_ID,
    UNIT16_COLLECTION,
    UNIT16_COLLECTION_METADATA,
    ChromaVectorStore,
    resolve_unit16_db_path,
)
from app.data_store import PROJECT_ROOT
from app.evaluation import normalize_source
from app.lc_chain import (
    EMPTY_CONTEXT,
    INSUFFICIENT_MARK,
    MAX_QUESTION_CHARS,
    MIN_SIMILARITY,
    ChainParts,
    ChainResult,
    OfflineChatModel,
    build_chain,
    build_chat_model,
    hits_to_dicts,
    render_hits,
    render_result,
    render_stage_table,
    run_chain,
    validate_question,
)
from app.loaders import load_materials
from app.parse_lab import EXPECTED_RECORDS, PROJECTS
from app.retriever import RetrievalService
from app.tool_agent import (
    DEFAULT_MAX_STEPS,
    DeepSeekToolModel,
    ToolAgent,
    ToolResult,
    build_tool_decision_model,
)
from app.tool_lab import FAQ_FILE, KNOWLEDGE_DIR, build_faq_db, resolve_faq_db_path
from app.toolkit import build_registry

# 任务卡写明了：电商售后组直接使用单元12 对应的实践项目资料，与单元14/15 同一批。
PROJECT_KEY = "ecom"

# 第3节课要跑的真实问题：资料里确实有答案（DOC-02-01 普通商品退货规则）。
DEFAULT_QUESTION = "普通商品退货有哪些要求？"
# 第4节课的"资料不足"演示：本组 7 条资料里没有跨境清关的内容，
# 检索会返回片段但相似度过不了门槛，链只能按"没有证据"处理。
INSUFFICIENT_QUESTION = "跨境保税商品的清关时效是多久？"
# 无效输入四例：空、空白、过短、过长、非字符串。
INVALID_CASES: list[tuple[str, object]] = [
    ("空字符串", ""),
    ("只有空白", "   "),
    ("长度不足", "退"),
    ("超出长度", "退货" * (MAX_QUESTION_CHARS // 2 + 5)),
    ("不是字符串", 123),
]

# 第4节课的对照：关键事实抽的是"能对得上的硬东西"，不是措辞。
FACT_PATTERN = re.compile(r"\d+(?:天|个工作日|小时|元|%)|DOC-02-\d{2}-[\w一-鿿-]+\.\w+")

SCENARIO_LABELS = {
    "answer": "组合链一次运行（第3节）",
    "steps": "逐段流转（第1、2节）",
    "empty": "资料不足（第4节异常）",
}

# 第1节要在项目里标出来的东西：可复用检索函数 + 真实资料来源。
REUSABLE_POINTS = [
    ("检索函数", "app/retriever.py · RetrievalService.search()",
     "单元09 起就没改过；本单元只是用 RunnableLambda 把它包起来"),
    ("资料来源", f"{PROJECTS[PROJECT_KEY].folder.name}（{EXPECTED_RECORDS} 条）",
     "单元12 解析入库，本单元重新写进 .chroma_unit16/ 保住自己的现场"),
    ("证据门槛", f"余弦相似度 ≥ {MIN_SIMILARITY}",
     "在本组 7 条资料上实测标定，不是正确率；换资料要重新标"),
    ("原有 Agent", "app/tool_agent.py · ToolAgent（单元15）",
     "保留不动，只作同题对照；新链不覆盖旧实现"),
]


@dataclass
class ChainScenario:
    """一个场景 = 一个问题 + 一次真实运行的结果。"""

    key: str
    label: str
    question: str
    result: ChainResult | None = None
    error: str = ""


@dataclass
class Comparison:
    """同题对照：LCEL 链 vs 单元15 的 ToolAgent。"""

    question: str
    chain: ChainResult
    baseline: ToolResult
    chain_facts: list[str] = field(default_factory=list)
    baseline_facts: list[str] = field(default_factory=list)
    verdict: str = ""


@dataclass
class ChainLabRun:
    """一次完整演练的输出；字段全部来自真实运行，渲染函数只读不写。"""

    database_path: Path
    collection_name: str
    knowledge_dir: Path
    kb_records: int
    model_name: str
    provider: str
    key_configured: bool
    min_similarity: float
    scenarios: list[ChainScenario] = field(default_factory=list)
    comparison: Comparison | None = None
    invalid_rows: list[tuple[str, str]] = field(default_factory=list)

    def scenario(self, key: str) -> ChainScenario | None:
        for item in self.scenarios:
            if item.key == key:
                return item
        return None


# ---------------------------------------------------------------- ① 装配


def build_stack(
    database_path: Path | None = None,
    model=None,
    min_similarity: float = MIN_SIMILARITY,
):
    """准备好"能跑一次链"的三件套：向量库 + 检索服务 + LCEL 链。"""
    records, failures = load_materials(KNOWLEDGE_DIR, PROJECT_ID)
    if not records:
        raise RuntimeError(
            f"没有可入库的资料，请检查资料目录：{KNOWLEDGE_DIR}；解析失败：{failures}"
        )
    store = ChromaVectorStore(
        Path(database_path) if database_path else resolve_unit16_db_path(),
        UNIT16_COLLECTION,
        UNIT16_COLLECTION_METADATA,
    )
    store.upsert_documents(records)
    service = RetrievalService(store)

    def search(question: str) -> list[dict]:
        """第1节要标出来的可复用函数：单元09 的检索服务，一个字都没改。"""
        return hits_to_dicts(service.search(question, PROJECT_ID, top_k=3))

    chat_model = model if model is not None else build_chat_model()
    parts: ChainParts = build_chain(search, chat_model, min_similarity=min_similarity)
    return parts, service, store


def build_baseline_agent(service: RetrievalService, decision_model=None):
    """第4节要的对照组：单元15 的 ToolAgent，用**同一份**检索服务与资料。"""
    db_path, _rows = build_faq_db()
    registry = build_registry(service, PROJECT_ID, FAQ_FILE, db_path, KNOWLEDGE_DIR)
    model = decision_model if decision_model is not None else build_tool_decision_model(registry)
    return ToolAgent(model, registry)


# ---------------------------------------------------------------- ② 运行


def run_scenario(
    key: str,
    question: str,
    parts: ChainParts,
    *,
    inspect_stages: bool = False,
) -> ChainScenario:
    """跑一个场景；链可复用（同一批资料，只换问题）。"""
    scenario = ChainScenario(
        key=key,
        label=SCENARIO_LABELS.get(key, key),
        question=question,
    )
    try:
        scenario.result = run_chain(
            question, parts, inspect_stages=inspect_stages
        )
    except ValueError as error:
        scenario.error = f"输入未通过校验：{error}"
    return scenario


def run_invalid(parts: ChainParts) -> list[tuple[str, str]]:
    """无效输入：五种坏输入分别撞一次守门，全部停在跑链之前。"""
    rows: list[tuple[str, str]] = []
    for label, raw in INVALID_CASES:
        try:
            validate_question(raw)
            rows.append((f"{label}（{raw!r}）", "竟然通过了校验（异常！）"))
        except ValueError as error:
            shown = raw if isinstance(raw, str) else raw
            if isinstance(raw, str) and len(raw) > 24:
                shown = raw[:24] + f"…（共 {len(' '.join(raw.split()))} 字）"
            rows.append((f"{label}（{shown!r}）", f"已拒绝：{error}"))
    return rows


def key_facts(*texts: str) -> list[str]:
    """从答案里抽出"能对得上的硬事实"：时限数字、金额数字、资料文件名。

    刻意不比措辞——任务卡写明了"无需逐字比较模型措辞"。
    """
    found: list[str] = []
    for text in texts:
        for item in FACT_PATTERN.findall(text or ""):
            if item not in found:
                found.append(item)
    return found


def compare_with_agent(
    question: str,
    parts: ChainParts,
    service: RetrievalService,
    baseline: ToolAgent | None = None,
    decision_model=None,
) -> Comparison:
    """同一问题、同一份资料，跑新链与旧 Agent，核对关键事实。"""
    chain_result = run_chain(question, parts)
    agent = baseline if baseline is not None else build_baseline_agent(service, decision_model)
    try:
        baseline_result = agent.run(question, DEFAULT_MAX_STEPS)
    except ValueError as error:
        baseline_result = ToolResult(
            question=question,
            answer="",
            sources=[],
            stop_reason="error",
            max_steps=DEFAULT_MAX_STEPS,
            trace=[],
            decision_calls=0,
            reject_reason=str(error),
        )

    chain_facts = key_facts(chain_result.answer, *chain_result.sources)
    baseline_text = baseline_result.answer + " " + " ".join(baseline_result.sources)
    baseline_facts = key_facts(baseline_text)
    common = [item for item in chain_facts if item in baseline_facts]

    chain_sources = set(chain_result.sources)
    baseline_sources = {normalize_source(item) for item in baseline_result.sources}
    if chain_result.insufficient and not baseline_result.finished:
        verdict = "链判定资料不足，旧 Agent 也未完成：一致地拒绝了回答"
    elif chain_result.insufficient or INSUFFICIENT_MARK in baseline_result.answer:
        verdict = "有一方判定资料不足：以人工复核为准，不要求措辞一致"
    elif common:
        verdict = f"关键事实一致（共同依据：{'、'.join(common)}）"
    elif chain_sources & baseline_sources:
        verdict = (
            "引用的资料文件一致，但答案里没出现同样的数字/编号："
            "措辞差异可接受，数字差异要人工核对"
        )
    else:
        verdict = "两条路径引用的资料不同：需要人工确认哪条才是本问题的依据"
    return Comparison(
        question=question,
        chain=chain_result,
        baseline=baseline_result,
        chain_facts=chain_facts,
        baseline_facts=baseline_facts,
        verdict=verdict,
    )


def run_class(
    database_path: Path | None = None,
    model=None,
    question: str = DEFAULT_QUESTION,
    insufficient_question: str = INSUFFICIENT_QUESTION,
    decision_model=None,
) -> ChainLabRun:
    """按任务卡顺序跑完四节课：标来源 → 逐段 → 组合链 → 对照与异常。

    model / decision_model 可以显式传入，好让离线测试不碰真实 API。
    """
    parts, service, store = build_stack(database_path, model)
    run = ChainLabRun(
        database_path=Path(database_path) if database_path else resolve_unit16_db_path(),
        collection_name=store.collection.name,
        knowledge_dir=KNOWLEDGE_DIR,
        kb_records=store.count(PROJECT_ID),
        model_name=_model_name(parts.model),
        provider=(os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower(),
        key_configured=bool(os.getenv("DEEPSEEK_API_KEY")),
        min_similarity=MIN_SIMILARITY,
    )
    run.scenarios.append(run_scenario("steps", question, parts, inspect_stages=True))
    run.scenarios.append(run_scenario("answer", question, parts))
    run.scenarios.append(run_scenario("empty", insufficient_question, parts))
    run.invalid_rows = run_invalid(parts)
    run.comparison = compare_with_agent(
        question, parts, service, decision_model=decision_model
    )
    return run


def _model_name(model: object) -> str:
    if isinstance(model, OfflineChatModel):
        return f"{model.model_name}（离线摘抄，不联网）"
    if isinstance(model, DeepSeekToolModel):  # pragma: no cover - 兜底
        return f"DeepSeek {model.model}"
    name = getattr(model, "model_name", None) or getattr(model, "model", "")
    return f"ChatDeepSeek {name}（真实调用）" if name else type(model).__name__


# ---------------------------------------------------------------- ③ 渲染


def render_header(run: ChainLabRun) -> str:
    """第1节要确认的三件事：资料路径、项目标识、模型与密钥。"""
    expected = (
        "，与单元12 一致）" if run.kb_records == EXPECTED_RECORDS
        else f"，单元12 是 {EXPECTED_RECORDS} 条：不一致）"
    )
    return "\n".join([
        f"知识资料目录：{_relative(run.knowledge_dir)}（{run.kb_records} 条{expected}",
        f"项目标识：{PROJECT_ID}；向量集合：{run.collection_name}（{_relative(run.database_path)}）",
        f"证据门槛：余弦相似度 ≥ {run.min_similarity}；模型：{run.model_name}",
        f"密钥已配置：{run.key_configured}",
    ])


def render_reusable() -> str:
    """第1节要在现有项目里标出来的可复用件与资料来源。"""
    lines = ["可复用件与真实资料来源（本单元不另造数据）：", "项\t位置\t说明"]
    for name, location, note in REUSABLE_POINTS:
        lines.append(f"{name}\t{location}\t{note}")
    return "\n".join(lines)


def render_scenario(scenario: ChainScenario) -> str:
    if scenario.result is None:
        return f"--- {scenario.label}\n问题：{scenario.question}\n{scenario.error}"
    blocks = [f"--- {scenario.label}", render_result(scenario.result)]
    if scenario.key in ("answer", "empty"):
        blocks.append("")
        blocks.append(render_hits(scenario.result))
    return "\n".join(blocks)


def render_invalid(rows: list[tuple[str, str]]) -> str:
    """无效输入的处理：守门在跑链之前，链一次都没被调用。"""
    lines = ["无效输入（守门在链之前，检索与模型都不会被触发）：", "输入\t结果"]
    for shown, outcome in rows:
        lines.append(f"{shown}\t{outcome}")
    return "\n".join(lines)


def render_comparison(comparison: Comparison) -> str:
    """第4节要交的对照：关键事实与来源，不比较措辞。"""
    chain = comparison.chain
    baseline = comparison.baseline
    lines = [
        "同题对照（同一问题、同一份资料、同一个检索服务）：",
        "项\tLCEL 链（单元16）\t单元15 ToolAgent",
        f"模型\t{chain.model_name}\t{_baseline_model(baseline)}",
        f"引用来源\t{'、'.join(chain.sources) or '（无）'}\t"
        + ("、".join(normalize_source(s) for s in baseline.sources) or "（无）"),
        f"停止状态\t{'完成' if not chain.insufficient else '资料不足'}\t{baseline.stop_reason}",
        f"关键事实\t{'、'.join(comparison.chain_facts) or '（无）'}\t"
        + ("、".join(comparison.baseline_facts) or "（无）"),
        "",
        f"结论：{comparison.verdict}",
        "说明：对照的是来源与数字这类硬事实；模型措辞不要求一致。",
    ]
    return "\n".join(lines)


def _baseline_model(baseline: ToolResult) -> str:
    if baseline.stop_reason == "error":
        return f"运行失败：{baseline.reject_reason}"
    return "DeepSeek（真实调用）" if _deepseek_enabled() else "离线脚本策略（LLM_PROVIDER=mock）"


def _deepseek_enabled() -> bool:
    return (os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower() == "deepseek"


def render_run(run: ChainLabRun) -> str:
    """一次完整演练的输出，与 main.py 的 /chain demo 逐字一致。"""
    blocks = ["=== 单元16：LangChain 核心组合链 ===", render_header(run), "", render_reusable(), ""]
    for key in ("steps", "answer", "empty"):
        scenario = run.scenario(key)
        if scenario is not None:
            blocks.extend([render_scenario(scenario), ""])
    if run.invalid_rows:
        blocks.extend([render_invalid(run.invalid_rows), ""])
    if run.comparison is not None:
        blocks.extend([render_comparison(run.comparison), ""])
    blocks.append(render_stage_table())
    return "\n".join(blocks)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_QUESTION",
    "INSUFFICIENT_QUESTION",
    "INVALID_CASES",
    "ChainLabRun",
    "ChainScenario",
    "Comparison",
    "build_baseline_agent",
    "build_stack",
    "compare_with_agent",
    "key_facts",
    "render_comparison",
    "render_header",
    "render_invalid",
    "render_reusable",
    "render_run",
    "render_scenario",
    "run_class",
    "run_invalid",
    "run_scenario",
]


if __name__ == "__main__":  # pragma: no cover - 手动跑演示时用
    print(render_run(run_class()))
