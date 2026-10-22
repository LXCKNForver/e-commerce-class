"""单元10：RAG 编排层——把第九单元的检索结果接到真实生成模型。

整条链一共五步，每一步都能在课堂上打断点看变量：

    ① 检索  RetrievalService.search(question, project_id, top_k) -> list[SearchResult]
    ② 未命中  没有证据时**直接拒答，一次模型都不调用**（省 token，也避免模型凭常识瞎编）
    ③ 上下文  给每条证据编号 [1][2][3]，编号与来源文件一一对应
    ④ 生成    问题 + 编号上下文一起发给真实 DeepSeek，由 rag_prompts 约束回答范围
    ⑤ 核验    只保留"回答中实际引用且编号合法"的来源，越界编号一律判为不可核验

职责分界：**检索归第九单元，编排归本单元**。本文件不写一句向量相似度计算，
也不碰 Chroma，检索全部走 RetrievalService。

两个必须讲清楚的边界：
1. 引用核验只能证明"编号存在且合法"，**不能证明答案每句话都忠于原文**。
2. 向量检索总会返回最近邻，所以"命中"不等于"相关"，相关性阈值留到后续优化单元。

    uv run --extra embedding --extra chroma python -c "from app.rag import build_rag_service as b; r=b(); print(r.answer('定制商品能退货吗？','e-commerce-after-sales').answer)"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.chroma_store import PROJECT_ID, ChromaVectorStore, resolve_unit10_db_path
from app.rag_prompts import NO_CITATION_ANSWER, NO_EVIDENCE_ANSWER, build_messages
from app.retriever import SearchResult, RetrievalService

# 本组售后场景真实会问的两句话；第二句能在 DOC-02-05 定制商品例外上稳定命中。
DEMO_QUESTION = "定制商品签收后还能申请退货吗？"
REFUSAL_QUESTION = "这个问题与本组售后资料无关，例如年假可以休几天？"
# 用在别的项目上演示未命中：向量检索一定返回最近邻，只有跨项目才能可靠拿到空结果。
EMPTY_PROJECT = "another_project"
# 单条资料送进上下文的最大字符数；切片策略留到后续优化单元，这里整篇文档都在阈值内。
CONTEXT_CHARS = 1200

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class RAGResult:
    """一次 RAG 问答的完整结果，字段足够支撑课堂上的来源核验。"""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    hits: list[SearchResult] = field(default_factory=list)
    llm_called: bool = False
    error: str | None = None

    @property
    def refused(self) -> bool:
        """拒答分支：不是模型说不知道，而是根本没走到生成那一步。"""
        return not self.llm_called

    @property
    def ok(self) -> bool:
        return not self.refused and self.error is None and bool(self.sources)


class RecordingLLM:
    """给任意模型套一层计数器，用来证明"未命中时一次模型都没调用"。

    课堂上讲第②步时最有说服力的不是代码，而是这个计数停在 0。
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[list[dict]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.inner.chat(messages)


class RAGService:
    """检索 → 拒答/上下文 → 生成 → 引用核验。"""

    def __init__(
        self,
        retriever: RetrievalService,
        llm,
        top_k: int = 3,
        context_chars: int = CONTEXT_CHARS,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.context_chars = context_chars

    # ---------------------------------------------------------------- ① 检索
    def retrieve(self, question: str, project_id: str, top_k: int) -> list[SearchResult]:
        return self.retriever.search(question, project_id, top_k=top_k)

    # ---------------------------------------------------------------- ③ 上下文
    def build_context(self, hits: list[SearchResult]) -> str:
        """把命中资料编成带编号的上下文；编号即后续引用的合法范围。"""
        blocks = []
        for index, hit in enumerate(hits, start=1):
            snippet = hit.text[: self.context_chars]
            blocks.append(f"[{index}] 来源：{hit.source}\n内容：{snippet}")
        return "\n\n".join(blocks)

    # ---------------------------------------------------------------- ⑤ 核验
    @staticmethod
    def extract_citations(text: str) -> set[int]:
        """抽取回答里出现的全部 [n] 编号（去重）。"""
        return {int(number) for number in CITATION_PATTERN.findall(text)}

    def verify_sources(self, generated: str, hits: list[SearchResult]) -> list[int]:
        """只留下"引用了且编号合法"的部分；越界编号说明模型编了不存在的证据。"""
        cited = self.extract_citations(generated)
        valid = sorted(number for number in cited if 1 <= number <= len(hits))
        # 引用数与被保留数不一致 = 出现了越界编号，宁可判不可核验也不放行。
        return valid if len(valid) == len(cited) else []

    # ---------------------------------------------------------------- 主流程
    def answer(
        self,
        question: str,
        project_id: str = PROJECT_ID,
        top_k: int | None = None,
    ) -> RAGResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("问题不能为空")
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError("必须指定 project_id，否则会跨项目取资料")

        # ① 检索：这一步还没有任何模型参与。
        hits = self.retrieve(question, project_id, top_k or self.top_k)

        # ② 未命中：没有证据就拒答，绝不把空上下文丢给模型让它自由发挥。
        if not hits:
            return RAGResult(question, NO_EVIDENCE_ANSWER, hits=hits, llm_called=False)

        # ③ 上下文：编号与来源一一对应，模型只能在编号范围内引用。
        context = self.build_context(hits)

        # ④ 生成：真实 DeepSeek；失败时如实报告，不降级编答案。
        try:
            generated = self.llm.chat(build_messages(question, context))
        except Exception as exc:  # noqa: BLE001 - 网络/鉴权/限流都要如实呈现
            return RAGResult(
                question,
                f"模型调用失败（{type(exc).__name__}）：{exc}",
                hits=hits,
                llm_called=True,
                error=str(exc),
            )

        generated = (generated or "").strip()
        if not generated or NO_EVIDENCE_ANSWER in generated:
            return RAGResult(question, NO_EVIDENCE_ANSWER, hits=hits, llm_called=True)

        # ⑤ 来源核验：合法编号换成真实文件来源，去重后保持出现顺序。
        verified = self.verify_sources(generated, hits)
        if not verified:
            return RAGResult(question, NO_CITATION_ANSWER, hits=hits, llm_called=True)

        sources = list(dict.fromkeys(hits[number - 1].source for number in verified))
        return RAGResult(
            question, generated, sources, verified, hits=hits, llm_called=True
        )


def build_rag_service(
    database_path: Path | None = None,
    records: list | None = None,
    llm=None,
    top_k: int = 3,
) -> RAGService:
    """建好单元10 自己的库并完成首次导入，返回可直接问答的 RAG 服务。

    任务卡要求"数据库留在本单元目录"，所以库目录是 `.chroma_unit10/`；
    稳定 ID 保证重复导入只是更新，记录数不会增长。
    """
    from app.ingest import load_markdown

    if llm is None:
        from app.llm import build_llm

        llm = build_llm()

    store = ChromaVectorStore(database_path or resolve_unit10_db_path())
    store.upsert_documents(records if records is not None else load_markdown())
    return RAGService(RetrievalService(store), llm, top_k=top_k)


def build_context_preview(rag: RAGService, hits: list[SearchResult], preview: int = 200) -> str:
    """把即将发给模型的编号上下文原样打印出来，供课堂核对第③步。"""
    lines = []
    for index, block in enumerate(rag.build_context(hits).split("\n\n"), start=1):
        head = block.splitlines()[0]
        body = " ".join(block.splitlines()[1:])
        lines.append(f"  {head}")
        lines.append(f"     {' '.join(body.split())[:preview]}")
    return "\n".join(lines)


def render_result(result: RAGResult) -> str:
    """把一次问答渲染成可贴进验收证据的文本。"""
    lines = [f"问题：{result.question}"]
    lines.append(
        f"检索命中：{len(result.hits)} 条；调用模型：{'是' if result.llm_called else '否（未命中分支）'}"
    )
    for rank, hit in enumerate(result.hits, start=1):
        lines.append(f"  [{rank}] 距离={hit.distance:.4f}  来源={hit.source}")
    lines.append(f"回答：{result.answer}")
    if result.cited:
        lines.append(f"引用编号：{result.cited}")
    lines.append("来源：" + ("、".join(result.sources) if result.sources else "无"))
    if result.error:
        lines.append(f"异常：{result.error}")
    return "\n".join(lines)
