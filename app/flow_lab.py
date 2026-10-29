"""单元18 的编排层：装本组资料与工具 → 组工作流 → 五条路径逐条验证 → 渲染证据。

和前面几个单元一样，本文件只负责把任务卡的四节课串成一条命令：

    第1节  概念记录 + 项目问题清单 + 本组能复用的资料与工具
    第2节  画出流程图与分支条件，跑通最小业务场景（知识分支）
    第3节  顺序步骤 + 条件路由跑起来，每步的输入输出都看得见
    第4节  逐条验证四条分支与兜底，核对"未命中不调模型""没证据不编答案"

    数据侧（一点没换）：本组 5 个资料文件 → Chroma（单元12 解析 + 单元09 检索）
                       同一份 FAQ CSV → SQLite 只读库（单元15）
    能力侧（一点没换）：app/toolkit.py 的 lookup_faq / search_knowledge / compute_return_deadline
    编排侧（本单元新增）：app/flow_chain.py（路由 → 取证据 → 分支 → 摘要 → 润色）

    uv run --extra embedding --extra chroma --extra langchain python -m app.flow_lab
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.chroma_store import PROJECT_ID
from app.flow_chain import (
    ROUTE_DEADLINE,
    ROUTE_FAQ,
    ROUTE_KNOWLEDGE,
    ROUTE_LABELS,
    ROUTE_UNKNOWN,
    FlowParts,
    FlowResult,
    ScriptedFlowModel,
    build_flow_model,
    build_workflow,
    render_concepts,
    render_flow,
    render_result,
    render_routes,
    render_steps,
    run_flow,
)
from app.lc_chain import MAX_QUESTION_CHARS, MIN_SIMILARITY
from app.retriever import RetrievalService
from app.tool_lab import KNOWLEDGE_DIR, build_kit

# 任务卡写明：电商售后组直接用单元12 对应的实践项目资料，与单元14—17 同一批。
PROJECT_KEY = "ecom"

# 第2节课的最小业务场景：资料里确实有答案（DOC-02-01 普通商品退货规则）。
DEFAULT_QUESTION = "普通商品退货有哪些要求？"
# 四条分支各配一条本组真实问题——"分支可达"就是靠这四条证明的。
FAQ_QUESTION = "EC-FAQ-01 的标准答复是什么？"
KNOWLEDGE_QUESTION = DEFAULT_QUESTION
DEADLINE_QUESTION = "2026年12月1日签收的普通商品，退货评估最晚到哪天？"
UNKNOWN_QUESTION = "帮我写一首关于秋天的诗"
# 分支能到、证据不到：本组 7 条资料里没有跨境清关的内容，检索过不了相似度门槛。
NO_EVIDENCE_QUESTION = "跨境保税商品的清关时效是多久？"

SCENARIO_LABELS: dict[str, str] = {
    ROUTE_KNOWLEDGE: "知识分支（第2节最小场景）",
    ROUTE_FAQ: "FAQ 编号分支",
    ROUTE_DEADLINE: "日期计算分支",
    "no_evidence": "分支到达但没证据（第4节异常）",
    ROUTE_UNKNOWN: "未命中 → 安全兜底（第4节异常）",
}

# 无效输入五例：全部停在跑工作流之前，一次检索、一次模型调用都不会发生。
INVALID_CASES: list[tuple[str, object]] = [
    ("空字符串", ""),
    ("只有空白", "   "),
    ("长度不足", "退"),
    ("超出长度", "退货" * (MAX_QUESTION_CHARS // 2 + 5)),
    ("不是字符串", 123),
]

# 第4节课要逐条对上的完成标准：三条标准各自靠哪条场景证明。
CHECKLIST: list[tuple[str, str, str]] = [
    ("至少两条分支都能到达", "知识 / FAQ / 日期 三条分支各一条真实问题", "分支列"),
    ("每步输入输出可查看", "步骤表里每步都记了输入字段数、输出字段数与说明", "步骤表"),
    ("未知类型进入安全兜底", "“帮我写一首关于秋天的诗” → unknown → 0 次模型调用", "停止原因"),
]


@dataclass
class FlowScenario:
    """一个场景 = 一个问题 + 一次真实运行的结果（或一条错误原因）。"""

    key: str
    label: str
    question: str
    result: FlowResult | None = None
    error: str = ""

    @property
    def route(self) -> str:
        return self.result.route if self.result else ""

    @property
    def stop_reason(self) -> str:
        return self.result.stop_reason if self.result else self.error


@dataclass
class FlowLabRun:
    """一次完整演练的输出；字段全部来自真实运行，渲染函数只读不写。"""

    knowledge_dir: Path
    kb_records: int
    faq_rows: int
    model_name: str
    provider: str
    key_configured: bool
    min_similarity: float
    scenarios: list[FlowScenario] = field(default_factory=list)
    invalid_rows: list[tuple[str, str]] = field(default_factory=list)

    def scenario(self, key: str) -> FlowScenario | None:
        for item in self.scenarios:
            if item.key == key:
                return item
        return None

    @property
    def reached_routes(self) -> list[str]:
        """本次真的走到的分支（去重、按出现顺序）——"分支可达"的证据。"""
        seen: list[str] = []
        for item in self.scenarios:
            if item.result and item.route and item.route not in seen:
                seen.append(item.route)
        return seen

    @property
    def zero_model_calls(self) -> list[str]:
        """一次模型都没调用的场景（兜底与无证据两条都该在这里）。"""
        return [
            item.key for item in self.scenarios
            if item.result and item.result.model_calls == 0
        ]


# ---------------------------------------------------------------- ① 装配


def build_stack(
    database_path: Path | None = None,
    model=None,
    min_similarity: float = MIN_SIMILARITY,
):
    """准备好"能跑一次工作流"的几件套：向量库 + FAQ 库 + 工具注册表 + 工作流。

    工具与资料**全部沿用单元15**（`tool_lab.build_kit` 建的那一套）：本单元只换
    "谁来编排"，不换数据口径——这句要写进文档，否则分数说不清是哪份资料算出来的。
    """
    _agent, registry, store, kb_count, faq_rows = build_kit(database_path)
    service = RetrievalService(store)
    chat_model = model if model is not None else build_flow_model()
    parts = build_workflow(
        registry, service, chat_model,
        project_id=PROJECT_ID, min_similarity=min_similarity,
    )
    return parts, registry, service, kb_count, faq_rows


# ---------------------------------------------------------------- ② 运行


def run_scenario(key: str, question: str, parts: FlowParts) -> FlowScenario:
    """跑一个场景；工作流可复用（同一批资料，只换问题）。"""
    scenario = FlowScenario(
        key=key,
        label=SCENARIO_LABELS.get(key, key),
        question=question,
    )
    try:
        scenario.result = run_flow(question, parts)
    except ValueError as error:
        scenario.error = f"输入未通过校验：{error}"
    return scenario


def run_invalid(parts: FlowParts) -> list[tuple[str, str]]:
    """无效输入：五种坏输入分别撞一次守门，全部停在跑工作流之前。"""
    rows: list[tuple[str, str]] = []
    for label, raw in INVALID_CASES:
        before = parts.log.total
        try:
            run_flow(raw, parts)
            rows.append((f"{label}（{raw!r}）", "竟然跑起来了（异常！）"))
        except ValueError as error:
            shown: object = raw
            if isinstance(raw, str) and len(raw) > 24:
                shown = raw[:24] + f"…（共 {len(' '.join(raw.split()))} 字）"
            rows.append((
                f"{label}（{shown!r}）",
                f"已拒绝：{error}；步骤表新增 {parts.log.total - before} 条",
            ))
    return rows


def run_class(
    database_path: Path | None = None,
    model=None,
    question: str = DEFAULT_QUESTION,
    min_similarity: float = MIN_SIMILARITY,
) -> FlowLabRun:
    """按任务卡顺序跑完四节课：最小场景 → 三条分支 → 两条异常 → 无效输入。"""
    parts, registry, service, kb_count, faq_rows = build_stack(
        database_path, model, min_similarity
    )
    service_kb = kb_count
    run = FlowLabRun(
        knowledge_dir=KNOWLEDGE_DIR,
        kb_records=service_kb,
        faq_rows=faq_rows,
        model_name=_model_name(parts),
        provider=(os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower(),
        key_configured=bool(os.getenv("DEEPSEEK_API_KEY")),
        min_similarity=min_similarity,
    )
    # 第2节：先跑通最小业务场景；第3、4节：另外两条分支 + 两条异常。
    for key, text in (
        (ROUTE_KNOWLEDGE, question),
        (ROUTE_FAQ, FAQ_QUESTION),
        (ROUTE_DEADLINE, DEADLINE_QUESTION),
        ("no_evidence", NO_EVIDENCE_QUESTION),
        (ROUTE_UNKNOWN, UNKNOWN_QUESTION),
    ):
        run.scenarios.append(run_scenario(key, text, parts))
    run.invalid_rows = run_invalid(parts)
    _ = registry, service  # 装配阶段已用到；这里只是让返回值可读
    return run


def _model_name(parts: FlowParts) -> str:
    steps = getattr(parts.summary_chain, "steps", [])
    model = steps[1] if len(steps) >= 2 else None
    if isinstance(model, ScriptedFlowModel):
        return "scripted-flow-model（离线编排，不联网）"
    name = getattr(model, "model_name", None) or getattr(model, "model", "")
    return f"ChatDeepSeek {name}（真实调用）" if name else type(model).__name__


# ---------------------------------------------------------------- ③ 渲染


def render_header(run: FlowLabRun) -> str:
    """第1节课要确认的三件事：资料、工具、模型与密钥。"""
    return "\n".join([
        f"知识资料目录：{_relative(run.knowledge_dir)}（向量库内 {run.kb_records} 条；"
        "与单元12/15 同一批资料）",
        f"结构化数据：常见售后问题.csv 同步进 SQLite 只读库，库内 {run.faq_rows} 行",
        f"工具：lookup_faq / search_knowledge / compute_return_deadline（单元15 注册表，一行没改）",
        f"模型：{run.model_name}（LLM_PROVIDER={run.provider}）；"
        f"密钥已配置：{run.key_configured}",
        f"证据门槛：余弦相似度 ≥ {run.min_similarity}（单元16 在同一批资料上标定的演示值）",
    ])


def render_summary_table(run: FlowLabRun) -> str:
    """第4节课要交的"逐条验证"：一条场景一行，分支、模型调用、停止原因一目了然。"""
    lines = [
        "逐条验证（每条都是真实运行，不是设想）：",
        "场景\t问题\t分支\t模型调用\t停止原因",
    ]
    for item in run.scenarios:
        lines.append(
            f"{item.label}\t{item.question}\t{item.route or '—'}\t"
            f"{item.result.model_calls if item.result else '—'}\t{item.stop_reason}"
        )
    lines.append("")
    lines.append(f"走到的分支：{'、'.join(run.reached_routes) or '（无）'}")
    lines.append(
        "模型调用 0 次的场景："
        + ("、".join(run.zero_model_calls) or "（无）")
        + "（兜底与无证据两条都该在列表里）"
    )
    return "\n".join(lines)


def render_checklist(run: FlowLabRun) -> str:
    """组长验收用的对照表：三条完成标准分别由哪条场景证明。"""
    lines = ["完成标准对照（组长验收用）：", "完成标准\t本组怎么证明\t看哪里"]
    lines.extend(f"{name}\t{how}\t{where}" for name, how, where in CHECKLIST)
    lines.append("")
    lines.append(
        f"另外核对：本次一共走到 {len(run.reached_routes)} 条分支（"
        f"{'、'.join(run.reached_routes)}），其中 "
        f"{len(run.zero_model_calls)} 条路径一次模型都没调用。"
    )
    return "\n".join(lines)


def render_scenario(scenario: FlowScenario) -> str:
    """一个场景的证据：结果 + 这一步的步骤表。"""
    if scenario.result is None:
        return f"【{scenario.label}】{scenario.question}\n{scenario.error}"
    lines = [f"【{scenario.label}】", render_result(scenario.result)]
    lines.append("")
    lines.append(render_steps(scenario.result))
    return "\n".join(lines)


def render_run(run: FlowLabRun) -> str:
    """按任务卡顺序渲染整套证据。"""
    blocks = [
        "＝" * 30 + " 第1节：资料、工具、概念与问题清单 " + "＝" * 30,
        render_header(run),
        "",
        render_concepts(),
        "",
        "＝" * 30 + " 第2节：流程图与分支条件 " + "＝" * 30,
        render_flow(),
        "",
        render_routes(),
        "",
        "＝" * 30 + " 第3、4节：逐条跑分支，每步输入输出都留痕 " + "＝" * 30,
    ]
    for item in run.scenarios:
        blocks.append(render_scenario(item))
        blocks.append("")
    blocks.append("无效输入（守门在跑工作流之前）：")
    blocks.append("输入\t结果")
    blocks.extend(f"{label}\t{verdict}" for label, verdict in run.invalid_rows)
    blocks.append("")
    blocks.append(render_summary_table(run))
    blocks.append("")
    blocks.append(render_checklist(run))
    return "\n".join(blocks)


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


__all__ = [
    "CHECKLIST",
    "DEFAULT_QUESTION",
    "DEADLINE_QUESTION",
    "FAQ_QUESTION",
    "FlowLabRun",
    "FlowScenario",
    "INVALID_CASES",
    "KNOWLEDGE_QUESTION",
    "NO_EVIDENCE_QUESTION",
    "SCENARIO_LABELS",
    "UNKNOWN_QUESTION",
    "build_stack",
    "render_checklist",
    "render_header",
    "render_run",
    "render_scenario",
    "render_summary_table",
    "run_class",
    "run_invalid",
    "run_scenario",
]


if __name__ == "__main__":  # pragma: no cover - 手动跑演示时用
    print(render_run(run_class()))
