"""单元13 的编排层：评估题 → 检索 → 回答 → 两类指标 → 落库 → 反馈 → 汇总。

分工与其他单元一样：
- 检索与回答：`app/retriever.py` + `app/rag.py`（单元09/10 的成果，直接复用）
- 入库 `app/chroma_store.py`、解析 `app/loaders.py`（单元12 的成果，直接复用）
- 这里只做本单元的新动作：**打分、落库、汇总**，一条命令串成一条流水线。

闭环是四步，每一步都能在课堂上打断点：

    ① 检索  RAGService.answer() 走单元09 的检索服务，取 Top-3 证据
    ② 回答  真实 DeepSeek 在编号上下文里作答，引用编号核验后留下真实来源
    ③ 打分  retrieval_scores（规则、零模型）+ ragas_scores（DeepSeek 当评审）
    ④ 落库  LogStore 把证据、答案、两类指标、耗时写进 SQLite 的 rag_logs

⓹ 之后是 `set_feedback()` 与 `summary()`：单题好坏标记上去，再从表里算均值。

    uv run --extra embedding --extra chroma --extra parse --extra evaluation python -m app.eval_lab
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.chroma_store import (
    PROJECT_ID,
    UNIT13_COLLECTION,
    UNIT13_COLLECTION_METADATA,
    ChromaVectorStore,
    resolve_unit13_db_path,
)
from app.data_store import PROJECT_ROOT
from app.evaluation import TOP_K, normalize_source, ragas_scores, retrieval_scores
from app.loaders import load_materials
from app.monitoring import LogStore, render_rows, render_summary
from app.parse_lab import EXPECTED_RECORDS, PROJECTS
from app.rag import RecordingLLM, RAGService
from app.retriever import RetrievalService

# 任务卡要求"把本组题目放入代码目录的 evaluation_cases.json"，所以激活题集在项目根目录。
EVAL_CASES_PATH = PROJECT_ROOT / "evaluation_cases.json"
# 完成标准：至少 3 道题均有参考答案和相关来源。
EXPECTED_CASES = 3
# 评估用的知识库就是本组单元12 收进来的那批资料（3md + 1pdf + 1csv），不再另收一份。
PROJECT_KEY = "ecom"
KNOWLEDGE_DIR = PROJECTS[PROJECT_KEY].folder
EXPECTED_KB_RECORDS = EXPECTED_RECORDS
# 评估日志库：运行产物，与 .chroma_unit13/ 一样不进仓库。
DEFAULT_LOG_DB = PROJECT_ROOT / "rag_monitor.sqlite3"

# 提交内容要求"说明每个指标在检查什么"，这张表就是答案，渲染进证据文档。
METRIC_EXPLANATIONS = [
    ("Precision@3", "规则", "Top-3 里有多少条命中了人工标注的来源（按条数计）",
     "同一文件的多行 CSV 各占一个位置，全命中也要一条条算"),
    ("Recall@3", "规则", "标注的相关来源被找回了多少（按去重文件计）",
     "重复片段不虚增覆盖率；Recall 低说明直接证据没进 Top-3"),
    ("Context Precision", "模型", "排在前面的检索文本里有多少真正有用",
     "量的是排序质量：有用的有没有被排到前面"),
    ("Context Recall", "模型", "参考答案里的事实能否被检索到的文本支撑",
     "量的是证据覆盖：缺了哪条证据就答不完整"),
    ("Faithfulness", "模型", "回答里的每个事实能否在检索文本里找到出处",
     "忠实度不等于正确率：答非所问但句句有据仍会得高分"),
]


@dataclass
class CaseResult:
    """一道评估题的完整结果；字段全部来自真实运行，渲染函数只读不写。"""

    case_id: str
    question: str
    answer: str
    hits: list[dict] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    ragas: dict = field(default_factory=dict)
    seconds: float = 0.0
    log_id: int | None = None
    llm_called: bool = True

    @property
    def ragas_error(self) -> str | None:
        return self.ragas.get("error")


@dataclass
class EvalRun:
    """一次评估运行的结果；真实跑出来什么就渲染什么。"""

    database_path: Path
    collection_name: str
    knowledge_dir: Path
    kb_records: int
    total_cases: int
    results: list[CaseResult] = field(default_factory=list)
    log_path: Path = DEFAULT_LOG_DB

    @property
    def run_cases(self) -> int:
        return len(self.results)


def load_eval_cases(path: Path = EVAL_CASES_PATH) -> list[dict]:
    """读取评估题集；文件不存在或不是合法 JSON 都在这里给出明确报错。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到评估题集：{path}")
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"评估题集不是合法 JSON：{exc}") from exc
    if not isinstance(cases, list):
        raise ValueError("评估题集必须是题目列表")
    validate_cases(cases)
    return cases


def validate_cases(cases: list[dict]) -> None:
    """逐题校验："题干 + 至少一个相关来源 + 参考答案"缺一不可，报错精确到第几题。"""
    if len(cases) < EXPECTED_CASES:
        raise ValueError(
            f"完成标准要求至少 {EXPECTED_CASES} 道题，当前只有 {len(cases)} 道"
        )
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"第 {index} 道题不是一个对象")
        missing = [
            name
            for name in ("id", "question", "expected_sources", "reference_answer")
            if not case.get(name)
        ]
        if missing:
            raise ValueError(f"第 {index} 道题缺少字段：{'、'.join(missing)}")


def build_eval_service(database_path: Path | None = None, llm=None) -> tuple[RAGService, ChromaVectorStore]:
    """准备评估用的知识库：解析本组资料 → 入库 → 返回可直接问答的 RAG 服务。

    库目录是 `.chroma_unit13/`，与单元12 演示用的 `.chroma_unit12/` 分开：
    评估题的 expected_sources 只对应本组那批文件，混进别的项目资料就说不清分数。
    """
    if llm is None:
        from app.llm import build_llm

        llm = build_llm()

    records, failures = load_materials(KNOWLEDGE_DIR, PROJECT_ID)
    if not records:
        raise RuntimeError(
            f"没有可入库的文本，请检查资料目录：{KNOWLEDGE_DIR}；解析失败：{failures}"
        )
    store = ChromaVectorStore(
        Path(database_path) if database_path else resolve_unit13_db_path(),
        UNIT13_COLLECTION,
        UNIT13_COLLECTION_METADATA,
    )
    store.upsert_documents(records)
    # 套一层计数器，证明"未命中那题一次模型都没调用"。
    service = RAGService(RetrievalService(store), RecordingLLM(llm), top_k=TOP_K)
    return service, store


def run_evaluation(
    limit: int = EXPECTED_CASES,
    service: RAGService | None = None,
    database_path: Path | None = None,
    cases_path: Path | None = None,
    log_path: Path | None = None,
    with_ragas: bool = True,
) -> EvalRun:
    """跑评估闭环：逐题检索、回答、打分、落库。

    limit 是本次最多评估的题数（任务卡：先 1 题确认，再 3 题）；
    with_ragas=False 时只算人工检索指标，不调用评审模型——离线演示用。
    """
    cases = load_eval_cases(Path(cases_path) if cases_path else EVAL_CASES_PATH)
    if limit < 1:
        raise ValueError("limit 必须大于 0")

    if service is None:
        service, store = build_eval_service(database_path)
        kb_count = store.count(PROJECT_ID)
    else:
        store = getattr(service.retriever, "store", None)
        kb_count = store.count(PROJECT_ID) if store is not None else 0

    logs = LogStore(Path(log_path) if log_path else DEFAULT_LOG_DB)
    collection = getattr(store, "collection", None)
    run = EvalRun(
        database_path=resolve_unit13_db_path() if database_path is None else Path(database_path),
        collection_name=getattr(collection, "name", UNIT13_COLLECTION),
        knowledge_dir=KNOWLEDGE_DIR,
        kb_records=kb_count,
        total_cases=len(cases),
        log_path=logs.path,
    )

    for case in cases[:limit]:
        question = case["question"]
        started = time.perf_counter()
        # ①② 检索 + 回答：全部复用单元09/10 的成果，本单元不重写检索。
        rag = service.answer(question, PROJECT_ID, TOP_K)
        elapsed = time.perf_counter() - started

        # 记录的来源要留完整路径（追溯出处用），文件名单独存一列（打分与展示用）。
        hits = [
            {
                "source": hit.source,
                "source_name": normalize_source(hit.source),
                "distance": round(hit.distance, 4),
                "text": hit.text,
            }
            for hit in rag.hits
        ]

        # ③ 打分：先算零模型的检索指标。
        precision, recall = retrieval_scores(
            [hit.source for hit in rag.hits], case["expected_sources"]
        )

        # ③ 打分：再让 DeepSeek 当评审；失败如实记录 error，绝不写假分数。
        ragas: dict = {}
        if not with_ragas:
            ragas = {"skipped": "本次未调用评审模型"}
        elif not rag.hits:
            ragas = {"skipped": "没有检索命中，跳过 RAGAS 评分"}
        else:
            try:
                ragas = ragas_scores(
                    question,
                    rag.answer,
                    [hit.text for hit in rag.hits],
                    case["reference_answer"],
                )
            except Exception as exc:  # noqa: BLE001 - 网络/鉴权/限流都要如实呈现
                ragas = {"error": f"{type(exc).__name__}: {exc}"}

        # ④ 落库：证据、答案、两类指标、耗时一起写进 SQLite。
        log_id = logs.record(
            case["id"], question, hits, rag.answer, precision, recall, ragas, elapsed
        )
        run.results.append(
            CaseResult(
                case_id=case["id"],
                question=question,
                answer=rag.answer,
                hits=hits,
                sources=list(rag.sources),
                precision=precision,
                recall=recall,
                ragas=ragas,
                seconds=elapsed,
                log_id=log_id,
                llm_called=rag.llm_called,
            )
        )
    return run


def set_feedback(log_id: int, feedback: str, log_path: Path | None = None) -> int:
    """给一条既有日志打用户反馈；ID 不存在或值不合法由 LogStore 抛出。"""
    LogStore(Path(log_path) if log_path else DEFAULT_LOG_DB).set_feedback(log_id, feedback)
    return log_id


def read_summary(log_path: Path | None = None) -> dict:
    """从日志表汇总平均指标、平均耗时与反馈计数。"""
    return LogStore(Path(log_path) if log_path else DEFAULT_LOG_DB).summary()


# ------------------------------------------------------------------ 渲染


def render_header(run: EvalRun) -> str:
    return "\n".join(
        [
            f"项目：{PROJECT_ID}；知识库：{_relative(run.knowledge_dir)}（{run.kb_records} 条）",
            f"数据库目录：{_relative(run.database_path)}；集合：{run.collection_name}",
            f"日志数据库：{_relative(run.log_path)}；本次评估 {run.run_cases}/{run.total_cases} 题",
        ]
    )


def render_case(result: CaseResult) -> str:
    """渲染一道题的完整证据：Top-3 来源、答案、两类指标、日志 ID、耗时。"""
    lines = [f"--- {result.case_id}：{result.question}"]
    if not result.hits:
        lines.append("  未检索到任何证据（拒答分支，未调用模型）")
    for rank, hit in enumerate(result.hits, start=1):
        lines.append(
            f"  {rank}. 距离={hit['distance']:.4f}  来源={hit['source_name']}"
        )
    lines.append(f"  答案：{_preview(result.answer)}")
    if result.sources:
        lines.append(f"  核验来源：{'、'.join(normalize_source(s) for s in result.sources)}")
    lines.append(
        f"  Precision@3={result.precision:.2f}  Recall@3={result.recall:.2f}  "
        f"RAGAS={result.ragas}"
    )
    lines.append(f"  日志ID={result.log_id}  耗时={result.seconds:.2f}秒")
    return "\n".join(lines)


def render_metrics_explanation() -> str:
    """提交内容要求的那张表：每个指标在检查什么。"""
    lines = ["指标对照：两个规则指标 + 三个模型指标"]
    for name, kind, question, note in METRIC_EXPLANATIONS:
        lines.append(f"  {name}（{kind}）：检查——{question}")
        lines.append(f"      {note}")
    return "\n".join(lines)


def render_run(run: EvalRun) -> str:
    """一次评估运行的完整输出，与 main.py 的 /eval 逐字一致。"""
    blocks = ["=== 单元13：RAG 系统评估与监控 ===", render_header(run), ""]
    for result in run.results:
        blocks.extend([render_case(result), ""])
    blocks.append(render_metrics_explanation())
    return "\n".join(blocks)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _preview(text: str, limit: int = 180) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "……"


__all__ = [
    "DEFAULT_LOG_DB",
    "EXPECTED_CASES",
    "build_eval_service",
    "read_summary",
    "render_case",
    "render_header",
    "render_metrics_explanation",
    "render_rows",
    "render_run",
    "render_summary",
    "run_evaluation",
    "set_feedback",
    "validate_cases",
]


if __name__ == "__main__":  # pragma: no cover - 手动跑评估时用
    print(render_run(run_evaluation()))
