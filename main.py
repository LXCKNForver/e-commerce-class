"""电商售后服务智能体：单元02—10 累计入口。

单元02 流式输出与多轮对话：/chat 走纯 stream 模式，保留多轮上下文。
单元03 Prompt工程与模板：/compare 用同一份订单数据、同一个问题对比差/好提示词。
单元04 Function Calling：默认走工具闭环，先查工具再回答，/trace 看调用轨迹。
单元05 工具注册与调度：所有工具统一登记在 ToolRegistry，/registry 看登记表，
/dispatch 不访问模型直接验证调度边界，/log 看带轮次与异常的调用日志。
单元06 Embedding与向量化：/embed 用本地中文模型把售后文本转成 512 维向量并排序，
只比较"像不像"，不做问答、不读知识库、不做业务判断。
单元07 向量数据库选型：/vector 看记录字段与精确遍历基线，
/chroma 建真实 Chroma 空集合并核对集合 ID；本单元不入数据，写入与查询留给单元08。
单元08 Chroma 写入与查询：/ingest 把 模拟数据/knowledge/ 里的售后资料批量写入集合，
重新连接后核对记录数并做项目内 Top3 检索；本单元同样不生成业务答案。
单元09 向量检索服务：/retrieve 用独立库 .chroma_unit9/ 把查询封装成 search/search_many，
支持 Top-K、来源过滤与项目隔离，只返回证据不生成回答。
单元10 RAG问答：/rag 用独立库 .chroma_unit10/ 复用单元09 的检索服务，
无证据时直接拒答且不调用模型，命中时编好带编号的证据上下文再交给真实 DeepSeek，
最后核验引用编号是否合法、来源是否真实存在。

直接输入问题 = 工具闭环模式（这是本项目跑通业务闭环的主路径）。
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

from app.conversation import Conversation
from app.demo_questions import NORMAL_QUESTIONS, OVERBOUND_QUESTIONS
from app.llm import MockLLM, build_llm, extract_order_id
from app.prompts import SYSTEM_PROMPT, build_messages, format_order_context
from app.prompt_lab import compare_prompts, render_comparison
from app.registry import ToolParam, ToolRegistry, ToolSpec, build_registry
from app.tool_loop import SYSTEM_PROMPT as TOOL_SYSTEM_PROMPT
from app.tool_loop import answer_with_registry, format_trace
from app.tools import ToolError, execute_tool_call, query_order

HELP_TEXT = """可用命令：
  <直接输入问题>   工具闭环模式：调用订单/退货/工单/政策工具后再回答
  /agent <问题>    同上，并强制打印本次工具调用轨迹
  /chat <问题>     单元02 流式模式：不使用工具，逐段显示回答
  /compare [问题]  单元03：同一份订单数据 + 同一个问题，对比差/好两套提示词
  /missing         单元04：演示工具名与参数校验（缺参、多余参数、未知工具）
  /missingloop     单元04(离线)：模型漏传参数 -> 工具报错 -> 自动补全重试
  /registry        单元05：打印注册表（工具、用途、参数、只读标记）
  /dispatch        单元05：不访问模型，直接验证调度：正常 + 未知工具 + 缺参 + 异常
  /log             单元05：最近一次调度的调用日志（轮次、状态、耗时、异常提示）
  /trace           查看最近一次工具调用的完整轨迹
  /embed [问题]    单元06：本地中文模型向量化 + 余弦相似度排序 + 边界核对
  /vector          单元07：记录字段图 + 精确遍历检索 + 元数据过滤 + 项目隔离
  /chroma          单元07：真实 Chroma 建空集合、只读查看、重复运行核对集合 ID
  /ingest          单元08：批量写入本组售后资料 + 重新连接 + 项目内 Top3 检索
  /retrieve        单元09：检索服务——Top-3/Top-5、来源过滤、项目隔离、批量查询
  /rag [问题]      单元10：RAG 问答——拒答分支、编号上下文、真实模型生成、引用核验
  /demo            跑通 3 个正常问题 + 2 个越权问题
  /history         查看当前会话历史
  /clear           清空会话历史
  /help            显示本帮助
  /exit            退出程序"""


def print_streamed(text: str, *, chunk: int = 6) -> None:
    """逐段打印，模拟终端流式显示效果。"""
    for start in range(0, len(text), chunk):
        print(text[start : start + chunk], end="", flush=True)
    print()


class AgentSession:
    """一个会话同时维护三种状态：可读历史、工具对话、最近一次调用轨迹。"""

    def __init__(self, llm, registry: ToolRegistry | None = None):
        self.llm = llm
        self.registry = registry or build_registry()
        self.chat = Conversation()
        self.dialog: list[dict] = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}]
        self.last_trace: list[dict] = []
        self.last_order_id: str | None = None

    def resolve_context(self, question: str) -> str:
        order_id = extract_order_id(question) or self.last_order_id
        if not order_id:
            return "（未提供订单号，无法查询模拟订单数据）"

        self.last_order_id = order_id
        result = query_order(order_id)
        if not result["ok"]:
            return f"（{result['message']}）"
        return format_order_context(result["order"])

    # ------------------------------------------------ 单元04：工具闭环

    def ask_with_tools(self, question: str, *, show_trace: bool = False) -> str:
        self.dialog.append({"role": "user", "content": question})
        answer, trace = answer_with_registry(
            self.llm, self.dialog, self.registry, require_tool_call=False
        )
        self.last_trace = trace
        self.chat.add_pair(question, answer)

        if show_trace or trace:
            print("\n" + "─" * 60)
            print("🔍 [思考过程：工具调用轨迹]")
            print(format_trace(trace))
            print("─" * 60)
            print("💬 [最终客服回复]\n")

        print_streamed(answer)
        return answer

    # ------------------------------------------------ 单元02：多轮流式

    def ask_stream(self, question: str) -> str:
        self.chat.add("user", question)

        context = self.resolve_context(question)
        recent = list(self.chat.history)
        recent[-1] = {
            "role": "user",
            "content": build_messages(question, context, quality="strong")[1]["content"],
        }
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent

        print("助手：", end="", flush=True)
        parts: list[str] = []
        for part in self.llm.stream(messages):
            parts.append(part)
            print(part, end="", flush=True)
        print()

        answer = "".join(parts)
        self.chat.add("assistant", answer)
        return answer

    # ------------------------------------------------ 单元03：提示词对比

    def compare(self, question: str) -> None:
        print(render_comparison(compare_prompts(self.llm, question)))

    def clear(self) -> None:
        self.chat.clear()
        self.dialog = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}]
        self.last_trace = []
        self.last_order_id = None
        self.registry.log.clear()


# ---------------------------------------------------------------- 演示脚本


MISSING_ARGUMENT_DEMOS = [
    {"name": "query_order", "arguments": {}},
    {"name": "check_return_eligibility", "arguments": {"order_id": "SO-2001"}},
    {"name": "check_return_eligibility", "arguments": {"order_id": "", "reason": "质量问题"}},
    {"name": "check_return_eligibility", "arguments": {"order_id": "SO-2001", "reason": "质量问题", "amount": 999}},
    {"name": "refund_order", "arguments": {"order_id": "SO-2001"}},
    {"name": "check_return_eligibility", "arguments": {"order_id": "SO-2001", "reason": "质量问题"}},
]


def demo_argument_validation() -> None:
    print("工具参数白名单与必填校验演示：")
    for call in MISSING_ARGUMENT_DEMOS:
        print(f"\n调用：{call['name']}  参数：{json.dumps(call['arguments'], ensure_ascii=False)}")
        try:
            result = execute_tool_call(call)
            print(f"  校验通过 -> 返回结果：{json.dumps(result, ensure_ascii=False)}")
        except ToolError as exc:
            print(f"  校验失败 -> {exc}")


def _unstable_handler(order_id: str) -> dict:
    raise RuntimeError("模拟数据库连接中断")


DISPATCH_DEMOS = [
    {"call_id": "demo-1", "name": "query_order", "arguments": {"order_id": "SO-2001"}},
    {"call_id": "demo-2", "name": "lookup_policy_doc", "arguments": {"doc_id": "DOC-02-05"}},
    {"call_id": "demo-3", "name": "refund_order", "arguments": {"order_id": "SO-2001"}},
    {"call_id": "demo-4", "name": "check_return_eligibility", "arguments": {"order_id": "SO-2001"}},
]


def demo_dispatch() -> None:
    registry = build_registry()
    print(f"已注册工具（{len(registry.names())} 个）：{', '.join(registry.names())}\n")

    print("场景一：一批请求里同时有成功与失败，逐条处理互不牵连。")
    for step in registry.dispatch(DISPATCH_DEMOS, max_calls=4, round_index=1):
        print(f"\n请求 {step['call_id']} -> {step['name']}")
        print(f"  参数：{json.dumps(step['arguments'], ensure_ascii=False)}")
        print(f"  状态：{step['status']}（{step['elapsed_ms']}ms）")
        print(f"  结果：{json.dumps(step['result'], ensure_ascii=False)[:220]}")

    unstable = build_registry()
    unstable.register(ToolSpec(
        name="unstable_demo_tool",
        description="课堂演示专用，不会出现在给模型的清单里。",
        params=(ToolParam("order_id", "演示用订单号"),),
        handler=_unstable_handler,
        usage="演示处理函数抛出非 ToolError 异常时的兜底",
    ))

    print("\n场景二：处理函数抛出非 ToolError 异常，被转成可读结果而不是让程序崩溃。")
    for step in unstable.dispatch(
        [
            {"call_id": "demo-5", "name": "unstable_demo_tool", "arguments": {"order_id": "SO-2001"}},
            {"call_id": "demo-6", "name": "query_order", "arguments": {"order_id": "SO-2002"}},
        ],
        round_index=2,
    ):
        print(f"\n请求 {step['call_id']} -> {step['name']}")
        print(f"  状态：{step['status']}")
        print(f"  结果：{json.dumps(step['result'], ensure_ascii=False)[:220]}")

    print("\n调度日志（含轮次、状态、耗时与异常提示）：")
    print(registry.log.render())
    print(unstable.log.render())


def demo_missing_argument_loop(llm) -> None:
    if not isinstance(llm, MockLLM):
        print("当前是 DeepSeek 模式，无法人为制造漏参；已改为展示静态校验。")
        demo_argument_validation()
        return

    sloppy_llm = MockLLM(omit_argument="reason")
    session = AgentSession(sloppy_llm)
    print("演示：模型第一轮漏传 reason，工具报错后自动补全重试。")
    session.ask_with_tools("订单SO-2001已签收，我想退货，需要什么条件？", show_trace=True)


def demo_embedding(query: str | None = None) -> None:
    try:
        from app.embedding import model_ready
        from app.vector_lab import (
            DEMO_QUERY,
            RISK_EXPECTED_ROUTE,
            RISK_QUERY,
            boundary_report,
            misjudge_check,
            render_boundary,
            render_misjudge,
            render_report,
            run_lab,
        )
    except ImportError as exc:
        print(f"单元06 依赖未安装（{exc}）。请先执行：uv sync --extra embedding")
        return

    ready, detail = model_ready()
    if not ready:
        print(f"向量模型不可用：{detail}")
        return

    target = query or DEMO_QUERY
    print(render_report(run_lab(target)))
    print()
    print(render_boundary(boundary_report()))
    print()
    print(render_misjudge(misjudge_check(RISK_QUERY, RISK_EXPECTED_ROUTE)))


# ---------------------------------------------------------------- 单元07：向量数据库


def demo_vector_store() -> None:
    """单元07：先看清"记录长什么样"和精确遍历基线，再谈索引。"""
    try:
        from app.vector_store import (
            DEMO_QUERY,
            TENANT_ID,
            InMemoryVectorStore,
            build_demo_records,
            render_field_diagram,
        )
    except ImportError as exc:
        print(f"单元07 依赖未安装（{exc}）。请先执行：uv sync --extra embedding")
        return

    print(render_field_diagram())
    print()

    store = InMemoryVectorStore("after_sale_knowledge_base_demo")
    items = build_demo_records()
    store.add_many(items)
    print(
        f"已写入 {store.count()} 条记录"
        f"（知识文档 {store.count(doc_type='knowledge')} 条、"
        f"样例问题 {store.count(doc_type='question')} 条），"
        f"每条 {store.dimension} 维。"
    )
    print(f"样例记录：{store.records[0].document_id} -> {store.records[0].text[:40]}…")
    print()

    print(f"精确遍历检索：{DEMO_QUERY}")
    for hit in store.query(DEMO_QUERY, TENANT_ID, top_k=3):
        print(
            f"  {hit.score:.4f}  {hit.record.document_id}  "
            f"类型={hit.record.doc_type}  {hit.record.text[:40]}"
        )

    print(f"\n元数据过滤 doc_type=knowledge：只剩 {store.count(doc_type='knowledge')} 条")
    for hit in store.query(DEMO_QUERY, TENANT_ID, top_k=2, doc_type="knowledge"):
        print(f"  {hit.score:.4f}  {hit.record.document_id}")

    print(f"项目隔离查询（换成别的项目）：{len(store.query(DEMO_QUERY, 'other-project'))} 条")
    print("重复 document_id 会被拒绝：", end="")
    try:
        store.add(
            items[0]["document_id"],
            items[0]["text"],
            items[0]["source"],
            items[0]["doc_type"],
            TENANT_ID,
        )
    except ValueError as exc:
        print(exc)
    print("注意：以上是全量遍历的精确检索，没有索引；数据量一大就要靠 HNSW/IVF 换速度。")


def demo_chroma() -> None:
    """单元07：真实 Chroma 建空集合 + 只读查看 + 重复运行核对集合 ID。"""
    try:
        from inspect_chroma import inspect_collection
        from setup_chroma import CHROMA_PATH, create_first_collection
    except ImportError as exc:
        print(f"单元07 依赖未安装（{exc}）。请先执行：uv sync --extra chroma")
        return

    print("第一次建库：")
    _, first = create_first_collection()
    print(f"  集合名称={first.name}  集合ID={first.id}  记录数={first.count()}")

    print("重复运行（集合 ID 应保持不变）：")
    _, second = create_first_collection()
    print(f"  集合名称={second.name}  集合ID={second.id}  记录数={second.count()}")
    print(f"  两次集合 ID 一致：{first.id == second.id}")

    info = inspect_collection()
    print("\n只读查看：")
    print(f"  Chroma 版本：{info['chroma_version']}")
    print(f"  持久化目录：{CHROMA_PATH}")
    print(f"  集合元数据：{info['metadata']}")
    print(f"  当前记录数：{info['count']}（本单元只建空集合，第八单元再写入）")


# ---------------------------------------------------------------- 单元08：写入与查询


def demo_chroma_write() -> None:
    """单元08：批量写入本组真实资料，再重新连接做一次项目内检索。"""
    try:
        from app.chroma_store import (
            DEMO_QUESTION,
            PROJECT_ID,
            ChromaVectorStore,
            render_record_fields,
        )
        from app.ingest import load_markdown, render_inventory
    except ImportError as exc:
        print(
            f"单元08 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma"
        )
        return

    # 第1课：先看清楚要喂进去的是什么资料。此时还没碰数据库。
    records = load_markdown()
    print(render_inventory(records))
    print()
    print(render_record_fields())
    print()

    # 第2课：连接库、记下导入前数量、批量写入。
    store = ChromaVectorStore()
    print(f"持久化目录：{store.database_path}")
    print(f"集合：{store.collection.name}  集合 ID：{store.collection.id}")
    print(f"导入前记录数：{store.count()}")
    store.upsert_documents(records)
    print(f"导入后记录数：{store.count()}")

    # 第4课：重新连接磁盘上的库，确认记录还在、集合还是同一个。
    reopened = ChromaVectorStore()
    print(f"重新连接后集合 ID：{reopened.collection.id}"
          f"（与首次一致：{reopened.collection.id == store.collection.id}）")
    print(f"重新连接后记录数：{reopened.count()}")

    print(f"\n检索问题：{DEMO_QUESTION}（项目：{PROJECT_ID}）")
    for rank, hit in enumerate(reopened.query(DEMO_QUESTION, PROJECT_ID), start=1):
        print(f"{rank}. 距离={hit['distance']:.4f}  余弦={hit['similarity']:.4f}")
        print(f"   来源={hit['source']}")
        print(f"   ID={hit['id']}")
    print("距离越小越接近、余弦越大越接近；两者都不是正确率。")
    print("本单元只做检索，不生成业务答案；重复运行时稳定 ID 会让记录数保持不变。")


# ---------------------------------------------------------------- 单元09：检索服务


def demo_retrieval() -> None:
    """单元09：把查询封装成服务——Top-K、来源过滤、项目隔离、批量、空问题。"""
    try:
        from app.chroma_store import PROJECT_ID, ChromaVectorStore, resolve_unit9_db_path
        from app.ingest import load_markdown
        from app.retriever import (
            DEMO_QUESTION,
            DEMO_SOURCE,
            SECOND_QUESTION,
            RetrievalService,
            render_results,
        )
    except ImportError as exc:
        print(
            f"单元09 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma"
        )
        return

    # 课前准备：单元09 用自己的库，不复制单元08 的数据库。
    records = load_markdown()
    database_path = resolve_unit9_db_path()
    store = ChromaVectorStore(database_path)
    before = store.count()
    store.upsert_documents(records)
    print(f"数据库目录：{database_path}")
    print(f"导入前/后：{before} / {store.count()} 条；项目：{PROJECT_ID}")
    print(f"集合：{store.collection.name}  集合 ID：{store.collection.id}")

    service = RetrievalService(store)
    # 一：同一问题改 K。Top-K 只控制返回条数，不改动入库数据。
    print()
    print(render_results("Top-3", service.search(DEMO_QUESTION, PROJECT_ID, top_k=3)))
    print()
    print(render_results("Top-5", service.search(DEMO_QUESTION, PROJECT_ID, top_k=5)))

    # 二：来源过滤在 Chroma 端执行，只允许返回指定的那一份资料。
    print()
    print(
        render_results(
            f"来源过滤（只允许 {DEMO_SOURCE}）",
            service.search(DEMO_QUESTION, PROJECT_ID, top_k=3, source=DEMO_SOURCE),
        )
    )

    # 三：项目不匹配时应为空，证明不会跨项目返回资料。
    print()
    print(
        render_results(
            "其他项目（another_project）",
            service.search(DEMO_QUESTION, "another_project", top_k=3),
        )
    )

    # 四：一次调用查多个问题，输出顺序与输入问题顺序相同。
    print()
    for question, results in zip(
        [DEMO_QUESTION, SECOND_QUESTION],
        service.search_many([DEMO_QUESTION, SECOND_QUESTION], PROJECT_ID, top_k=2),
    ):
        print(render_results(f"批量检索 {question}", results))
        print()

    # 五：空问题是边界，必须给出明确提示而不是静默返回空。
    try:
        service.search("   ", PROJECT_ID)
    except ValueError as exc:
        print(f"空问题错误提示：ValueError: {exc}")

    print("距离越小越接近；本单元只返回证据，不生成回答（生成留给后续单元）。")


# ---------------------------------------------------------------- 单元10：RAG 问答


def demo_rag(question: str | None = None) -> None:
    """单元10：检索 → 无证据拒答 → 编号上下文 → 真实生成 → 引用核验。"""
    try:
        from app.chroma_store import PROJECT_ID, ChromaVectorStore, resolve_unit10_db_path
        from app.ingest import load_markdown
        from app.rag import (
            DEMO_QUESTION,
            EMPTY_PROJECT,
            NO_EVIDENCE_ANSWER,
            RAGService,
            RecordingLLM,
            build_context_preview,
            render_result,
        )
        from app.retriever import RetrievalService
    except ImportError as exc:
        print(
            f"单元10 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma"
        )
        return

    # 第1课：本单元独立库，与其他单元的库互不干扰。
    records = load_markdown()
    database_path = resolve_unit10_db_path()
    store = ChromaVectorStore(database_path)
    before = store.count()
    store.upsert_documents(records)
    print(f"数据库目录：{database_path}")
    print(f"导入前/后：{before} / {store.count()} 条；项目：{PROJECT_ID}")
    print(f"集合：{store.collection.name}  集合 ID：{store.collection.id}")

    llm = RecordingLLM(build_llm())
    rag = RAGService(RetrievalService(store), llm)
    target = question or DEMO_QUESTION

    # 第2课：跨项目必然没有证据，这时必须拒答，且一次模型都不调用。
    print()
    print(f"未命中示例（项目：{EMPTY_PROJECT}）：")
    miss = rag.answer(target, EMPTY_PROJECT)
    print(f"  回答：{miss.answer}")
    print(f"  命中 {len(miss.hits)} 条；是否等于拒答话术：{miss.answer == NO_EVIDENCE_ANSWER}")
    print(f"  模型累计调用次数：{llm.call_count}（应为 0）")

    # 第3课：正常问答前先看编号上下文长什么样，这是模型唯一能引用的范围。
    hits = rag.retrieve(target, PROJECT_ID, rag.top_k)
    print()
    print("编号上下文（第③步，模型只能在这些编号里引用）：")
    print(build_context_preview(rag, hits))

    # 第4课：真实模型生成 + 引用核验。
    if type(llm.inner).__name__ == "MockLLM":
        print()
        print("提示：当前是离线 mock 模式，MockLLM 的回答不带 [n] 引用编号，")
        print("     所以下面这条必然被第⑤步判为不可核验。这是编排规则在生效，不是 bug；")
        print("     想看真实 DeepSeek 的回答，在 .env 里把 LLM_PROVIDER 改成 deepseek。")
    print()
    result = rag.answer(target, PROJECT_ID)
    print(render_result(result))
    print()
    print(f"模型累计调用次数：{llm.call_count}")
    print("引用核验只能证明编号存在且合法，不能证明每句话都忠于原文；")
    print("越界编号会被判为不可核验，宁可不给来源也不放行。")


def run_demo(session: AgentSession) -> None:
    print("开始演示：3 个正常问题 + 2 个越权问题（工具闭环模式）。")
    for index, question in enumerate(NORMAL_QUESTIONS, start=1):
        print(f"\n--- 正常问题{index} ---")
        print(f"你：{question}")
        session.ask_with_tools(question, show_trace=True)
    for index, question in enumerate(OVERBOUND_QUESTIONS, start=1):
        print(f"\n--- 越权问题{index} ---")
        print(f"你：{question}")
        session.ask_with_tools(question, show_trace=True)


# ---------------------------------------------------------------- 入口


def main() -> None:
    llm = build_llm()
    session = AgentSession(llm)

    provider = "DeepSeek 真实调用" if not isinstance(llm, MockLLM) else "Mock 离线模式"
    print(f"电商售后服务智能体已启动（{provider}）")
    print("输入 /help 查看命令。")

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n对话结束。")
            break

        if not question:
            continue

        command, _, argument = question.partition(" ")
        command = command.lower()
        payload = argument.strip()

        if command == "/exit":
            print("对话结束。")
            break
        if command == "/help":
            print(HELP_TEXT)
            continue
        if command == "/clear":
            session.clear()
            print("助手：会话历史与工具上下文已清空。")
            continue
        if command == "/history":
            print(session.chat.render())
            continue
        if command == "/trace":
            print(format_trace(session.last_trace))
            continue
        if command == "/demo":
            run_demo(session)
            continue
        if command == "/missing":
            demo_argument_validation()
            continue
        if command == "/registry":
            print(f"已注册工具（{len(session.registry.names())} 个）：")
            print(session.registry.describe())
            continue
        if command == "/dispatch":
            demo_dispatch()
            continue
        if command == "/log":
            print(session.registry.log.render())
            continue
        if command == "/missingloop":
            demo_missing_argument_loop(llm)
            continue
        if command == "/embed":
            demo_embedding(payload or None)
            continue
        if command == "/vector":
            demo_vector_store()
            continue
        if command == "/chroma":
            demo_chroma()
            continue
        if command == "/ingest":
            demo_chroma_write()
            continue
        if command == "/retrieve":
            demo_retrieval()
            continue
        if command == "/rag":
            demo_rag(payload or None)
            continue
        if command == "/compare":
            target = payload or NORMAL_QUESTIONS[0]
            print(f"对比问题：{target}")
            session.compare(target)
            continue
        if command == "/chat":
            if not payload:
                print("用法：/chat 你的问题")
                continue
            print(f"你：{payload}")
            session.ask_stream(payload)
            continue
        if command == "/agent":
            if not payload:
                print("用法：/agent 你的问题")
                continue
            print(f"你：{payload}")
            session.ask_with_tools(payload, show_trace=True)
            continue

        try:
            session.ask_with_tools(question)
        except (ToolError, RuntimeError, ValueError) as exc:
            print(f"助手：本次处理失败（{exc}）。请补充订单号后重试，或转人工客服。")
        except Exception as exc:
            print(f"助手：本次处理出现未预期异常（{type(exc).__name__}: {exc}）。")
            print("      会话已保留，请稍后重试或转人工客服。")


if __name__ == "__main__":
    main()