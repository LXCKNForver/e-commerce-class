"""电商售后服务智能体：单元02—16 累计入口。

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
单元11 分块优化与BGE重排：/chunk 用独立库 .chroma_unit11/ 建立整篇与子块两个集合，
把 80 字符窗口、20 字符重叠的子块召回 Top5，再用本地 BGE Cross-Encoder 重排取前三，
与整篇检索 Top3 并列对照两版候选、两版回答与引用，只改候选来源、不改生成环节。
单元12 文档解析与知识库：/parse [项目] 用独立库 .chroma_unit12/ 把 Markdown、PDF、CSV
三种格式解析成同一种带 project_id / source / location 的知识记录，所选项目解析 7 条，
入库后从数据库回读 PDF 第 1 页与 CSV 第 2 行，重跑不重复，再切到另一个项目组
证明库内共 14 条、但按项目计数仍然各 7 条。
单元13 RAG系统评估与监控：/eval 用独立库 .chroma_unit13/ 装载本组资料，对
evaluation_cases.json 里的 3 道评估题跑完整闭环——复用单元09/10 的检索与回答，
再给每题算两类分数：规则算的 Precision@3/Recall@3（比对人工标注的来源文件），
以及 DeepSeek 当评审的 RAGAS Context Precision/Recall 与 Faithfulness；
每题连同 Top-3 证据、答案、分数、耗时写进本地 SQLite 的 rag_logs 表，
/eval summary 汇总均值与反馈，/eval feedback ID good|bad 给单题打标记。

单元14 ReAct Agent：/act 让模型只提动作、由 Python 执行真实检索——复用单元09 的
检索服务和单元12 的本组资料（独立库 .chroma_unit14/），逐轮打印决策、动作、观察，
再用 max_steps 2 重跑同一道复杂题，对比 finished 与 max_steps 两种停止状态。
单元15 Agent 工具集：/tools 把能力拆成三项只读工具——search_knowledge 查 Chroma
（单元09 检索）、lookup_faq 按编号查 SQLite（本组 FAQ CSV 同步而来，只读 + 参数化
SQL）、compute_return_deadline 用 Python 算日期（天数必须先在资料原文里查到）；
工具名与参数先过白名单再由 Python 执行，未授权的工具与格式不对的参数在执行前被拒。
单元16 LangChain核心：/chain 用 Runnable、Prompt、Model、Parser 把上面这些现成能力
串成一条 LCEL 链——RunnableLambda 包住单元09 的检索（不改写它），assign 只加字段，
所以问题、片段与来源一路都在；Prompt 规定证据边界，ChatDeepSeek 生成，
StrOutputParser 取出字符串。同一问题再跑一遍单元15 的 ToolAgent 作对照，
只比对来源与数字这类硬事实。原有 Agent 一行没删，新链不覆盖旧实现。
单元17 LangChain Agents：/lcagent 把单元15 注册表里那三项只读工具包成 LangChain 的
BaseTool，交给 create_agent 生成的执行器去跑循环——工具还是那套白名单能力（一个字没改），
变的是"循环不再手写"。每一步的输入、输出、失败原因与耗时都留在步骤日志里，
所以工具出错能定位到第几步；max_steps 是模型决策次数上限，触顶就收尾且不产出答案。
同一问题再跑一遍单元15 手写的循环作对照，只比调用的工具与数字这类硬事实。
单元19 对话记忆：/memory 给上面那个执行器加上"记性"——顺序固定为读 → 答 → 写：
先按会话回忆（完整历史 / 摘要 / 向量长期记忆三种模式），把「刚才那项」解析成具体编号；
再让执行器**重新**调工具取事实（记忆只解决指代，不解决答案）；最后只把正常完成的轮次
写回记忆。对话写在 `.chroma_unit19_memory/`，与装资料的 `.chroma_unit15/` 严格分开——
混在一起就会把上一轮说过的话当成制度原文引用。会话隔离靠键前缀 + 库端过滤双保险。

单元18 Chain编排：/flow 把"先路由、再取证据、再摘要、再润色"编成一条带条件分支的
工作流——路由用编号/日期/业务词三条规则（不调模型），四条分支各配一条本组真实问题，
分支到了但没证据、以及三条规则都没命中的问题，一律走兜底：不检索、不调模型、不编答案。
每一步的输入字段、输出字段与耗时都记在步骤表里，所以"哪一步出的问题"看得见。
资料与工具仍是单元12/15 那一套，本单元只换"谁来编排"。

直接输入问题 = 工具闭环模式（这是本项目跑通业务闭环的主路径）。
"""

from __future__ import annotations

import json
import os
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
  /chunk [问题]    单元11：80字符分块+20重叠召回Top5 → 本地BGE重排取前三 → 两版回答对照
  /parse [项目]    单元12：md/pdf/csv 解析为同一种记录 → 入库 → 回读 → 重跑 → 换项目对照
                  项目填 ecom（默认，本组电商售后）或 travel（课程另一组）
  /eval [参数]    单元13：评估闭环——检索/回答/两类指标/SQLite 日志/反馈/汇总
                  不带参数跑完 3 题；/eval 1 只跑 1 题；/eval summary 看汇总；
                  /eval feedback 2 bad 给第 2 条日志打反馈
  /act [参数]     单元14：ReAct——模型提动作，Python 执行真实检索
                  不带参数用单事项问题；/act 你的问题 自定义单事项问题；
                  /act double 跑双事项问题（观察第二轮检索）；
                  /act limit 用 max_steps 2 重跑（看未完成的停止状态）；
                  /act demo 按任务卡顺序跑完三段并做轨迹对比
  /tools [参数]    单元15：Agent 工具集——三项只读工具 + 白名单
                  不带参数跑单工具（FAQ 按编号查）；/tools 你的问题 自定义；
                  /tools combo 组合问题（观察连续调用两个工具）；
                  /tools calc 计算工具（签收日 + 资料时限 = 截止日）；
                  /tools reject 未知工具与非法参数被拒；
                  /tools demo 按任务卡顺序跑完四段
  /chain [参数]    单元16：LangChain LCEL 链——检索 → assign → Prompt → 模型 → Parser
                  不带参数用本组真实问题跑一次组合链；/chain 你的问题 自定义；
                  /chain steps 逐段打印六段的输入输出；
                  /chain compare 同一问题对照单元15 的 ToolAgent（只比对硬事实）；
                  /chain empty 资料不足时的处理；/chain bad 无效输入被守门拦下；
                  /chain demo 按任务卡四节课顺序跑完一遍
  /lcagent [参数]  单元17：LangChain Agent 执行器——同一批工具，循环交给框架跑
                  不带参数用最小业务场景跑一次（按编号查标准答复）；
                  /lcagent 你的问题 自定义；/lcagent combo 连续调用两个工具；
                  /lcagent error 工具报错定位到第几步；/lcagent limit 步数上限触顶；
                  /lcagent reject 清单外的工具（模型提了也执行不了）；
                  /lcagent compare 同一问题对照单元15 手写的循环；
                  /lcagent tools 看执行器配置与接入的工具；
                  /lcagent demo 按任务卡四节课顺序跑完一遍
  /flow [参数]     单元18：Chain 编排——路由 → 取证据 → 分支 → 摘要 → 润色
                  不带参数跑知识分支（本组真实问题）；/flow 你的问题 自定义；
                  /flow faq 按 FAQ 编号精确查；/flow deadline 查时限再算截止日期；
                  /flow unknown 未命中分支 → 安全兜底（0 次模型调用）；
                  /flow empty 分支到了但没证据 → 资料不足（0 次模型调用）；
                  /flow routes 看流程图与分支规则表；
                  /flow steps 同一问题逐段打印每步的输入输出；
                  /flow bad 无效输入被守门拦下；/flow demo 按四节课顺序跑完一遍
  /memory [参数]   单元19：对话记忆——读 → 答 → 写
                  不带参数问一句（默认问 EC-FAQ-01），接着问就是同一会话的下一轮；
                  /memory buffer|summary|vector|hybrid 用指定模式连跑两轮；
                  /memory flow 记忆流程与两类库的分工；/memory modes 三种模式对比；
                  /memory restart 重启后还能召回什么；/memory isolation 两个会话不串话；
                  /memory new 会话名 换一个会话；/memory bad 无效输入被守门拦下；
                  /memory demo 按任务卡四节课顺序跑完一遍
  /plan [参数]     单元20：Plan-and-Solve——先出计划，再逐步执行
                  不带参数跑最小场景（单步：按编号查标准答复）；/plan 你的目标 自定义；
                  /plan multi 三步计划（查答复 → 检索规则 → 算截止日期）；
                  /plan replan 参数格式错 → 重规划一次 → 跑完；
                  /plan abort 修订后仍失败 → 终止并报错（无答复）；
                  /plan missing 编号合法但库里没有 → 只返回「未找到」；
                  /plan flow 流程图与概念对照；/plan bad 无效输入被守门拦下；
                  /plan demo 按任务卡四节课顺序跑完一遍
  /agents [参数]   单元21：多智能体架构——规划/执行/证据核验/答复/评审 五角色 + 共享黑板
                  不带参数跑正常协作（三步计划 → 观察 → 证据 → 草稿 → 评审通过 → 交付）；
                  /agents 你的目标 自定义目标；
                  /agents revise 评审拒绝 → 修订一次 → 通过 → 交付；
                  /agents reject 两次都不通过 → 停止自动交付、转人工复核；
                  /agents gap 证据不足 → 如实说明而不是编；
                  /agents abort 执行阶段终止 → 连草稿都没有，评审不启动；
                  /agents roles 架构图与角色输入输出表；/agents bad 无效输入被守门拦下；
                  /agents demo 按任务卡四节课跑完一遍
  /group [参数]    单元22：AutoGen 实战——规划员/答复执行员/评审员 三个角色同处一间群聊
                  不带参数跑正常协作（证据 → 规划 → 草稿 → 复核通过 → 交付）；
                  /group 你的任务 自定义任务；
                  /group tight 只给 2 轮 → 复核没轮到 → 未交付；
                  /group budget 预算花完就停 → 未交付（轮次还有余）；
                  /group reject 复核不通过 → 答复留空、转人工；
                  /group gap 编号查不到 → 如实说明、复核照样通过；
                  /group roles 群聊时序与角色表；/group bad 无效输入被守门拦下；
                  /group demo 按任务卡四节课跑完一遍
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


# ---------------------------------------------------------------- 单元11：分块 + 重排


def demo_rag_optimized(question: str | None = None) -> None:
    """单元11：分块召回 → BGE 重排 → 取前三 → 两版回答对照。"""
    try:
        from app.chroma_store import (
            PROJECT_ID,
            UNIT11_CHUNK_COLLECTION,
            UNIT11_WHOLE_COLLECTION,
            ChromaVectorStore,
            resolve_unit11_db_path,
        )
        from app.chunking import (
            CHUNK_OVERLAP,
            CHUNK_SIZE,
            DEMO_SOURCE as CHUNK_DEMO_SOURCE,
            chunks_of,
            render_chunk_positions,
            split_documents,
            split_whole_documents,
        )
        from app.ingest import load_markdown
        from app.rag import RecordingLLM
        from app.rag_opt import (
            DEMO_QUESTION,
            RAGOptimizer,
            render_answers,
            render_evidence_check,
            render_recall,
        )
        from app.rerank import load_reranker, render_order_change, render_ranked, rerank_ready
        from app.retriever import RetrievalService
    except ImportError as exc:
        print(
            f"单元11 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma"
        )
        return

    # 第1课：本单元独立库，整篇与子块各占一个集合，与其他单元互不干扰。
    records = load_markdown()
    database_path = resolve_unit11_db_path()
    whole_store = ChromaVectorStore(database_path, UNIT11_WHOLE_COLLECTION)
    chunk_store = ChromaVectorStore(database_path, UNIT11_CHUNK_COLLECTION)
    whole_docs = split_whole_documents(records)
    chunks = split_documents(records)
    whole_store.upsert_documents(whole_docs)
    chunk_store.upsert_documents(chunks)
    print(f"数据库目录：{database_path}")
    print(f"集合：{whole_store.collection.name}（{whole_store.count()} 条）"
          f" / {chunk_store.collection.name}（{chunk_store.count()} 条）")
    print(f"项目：{PROJECT_ID}；切分参数：{CHUNK_SIZE} 字符窗口 + {CHUNK_OVERLAP} 字符重叠")

    # 0–8 分钟：相邻块的起止位置与重叠内容，这是后面所有环节的地基。
    print()
    print(render_chunk_positions(chunks_of(chunks, CHUNK_DEMO_SOURCE)))

    target = question or DEMO_QUESTION
    # 缺少本地权重就早说，避免讲到重排那一步才报错。
    ready, detail = rerank_ready()
    if not ready:
        print()
        print(f"本地 BGE 重排不可用：{detail}")
        print("请按课程材料说明下载 bge-reranker-base，或在 .env 里设置 RERANK_MODEL_DIR。")
        return
    load_reranker()

    llm = RecordingLLM(build_llm())
    optimizer = RAGOptimizer(
        RetrievalService(whole_store), RetrievalService(chunk_store), llm
    )
    # 一次性跑完整轮：召回、重排、两版回答都在这里完成，下面按课堂节奏分段打印。
    run = optimizer.compare(target, PROJECT_ID, len(whole_docs), len(chunks))

    # 8–16 分钟：同一道题走两条召回路径，整篇 Top3 与分块 Top5 并列。
    print()
    print(render_recall(run))

    # 16–28 分钟：原问题与每个候选成对送进 BGE，配对后降序，取前三。
    print()
    print(render_ranked(run.ranked))
    print()
    print(render_order_change(run.candidates[:3], run.optimized_hits))
    print()
    print(render_evidence_check(run))

    # 28–35 分钟：两版共用同一个模型与提示词，只有候选来源不同。
    if type(llm.inner).__name__ == "MockLLM":
        print()
        print("提示：当前是离线 mock 模式，MockLLM 的回答不带 [n] 引用编号，")
        print("     所以下面两版都会被引用核验判为不可核验。这是编排规则在生效，不是 bug；")
        print("     想看真实 DeepSeek 的回答，在 .env 里把 LLM_PROVIDER 改成 deepseek。")
    print()
    print(render_answers(run))
    print()
    print(f"模型累计调用次数：{llm.call_count}（两版各一次）")
    print("重排只能改变已经召回的候选顺序，不能找回候选池之外的证据；")
    print("BGE 分数是相关性 logit，与 Chroma 的 L2 距离方向相反、量纲不同，不能混算。")


# ---------------------------------------------------------------- 单元12：文档解析


def demo_parse(project_key: str | None = None) -> None:
    """单元12：三种格式 → 同一种记录 → 入库 → 回读 → 重跑 → 换项目。"""
    try:
        from app.parse_lab import (
            project_keys,
            render_run,
            run_parse,
        )
    except ImportError as exc:
        print(
            f"单元12 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra parse"
        )
        return

    print(f"可选项目：{project_keys()}（不填默认 ecom，即本组电商售后）")
    try:
        run = run_parse(project_key)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        # 项目名写错、资料目录缺失、依赖没装，都在这里给出可执行的下一步。
        print(f"解析未执行：{error}")
        return
    print()
    print(render_run(run))
    print()
    print("三种格式在磁盘上的最小单位不同（文档 / 页 / 行），解析层把它们压平成同一种记录；")
    print("入库后一律按 project_id 过滤计数，所以资料和别的项目写在同一个集合里也不会串。")


# ---------------------------------------------------------------- 单元13：RAG 评估与监控


def demo_evaluation(argument: str | None = None) -> None:
    """单元13：评估闭环——跑去打分落库 / 看汇总 / 打反馈。

    参数三种：不带（跑完 3 题）、题数（只跑前几题）、summary / feedback。
    任务卡的课堂节奏是"先 1 题确认，再跑 3 题"，所以留了 `/eval 1`。
    """
    try:
        from app.eval_lab import (
            DEFAULT_LOG_DB,
            EXPECTED_CASES,
            build_eval_service,
            read_summary,
            render_rows,
            render_run,
            render_summary,
            run_evaluation,
            set_feedback,
        )
        from app.monitoring import LogStore
    except ImportError as exc:
        print(
            f"单元13 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra parse --extra evaluation"
        )
        return

    words = (argument or "").strip().split()

    # 汇总直接从日志表里算，不重跑模型、不重复花钱。
    if words and words[0] == "summary":
        print(render_summary(read_summary()))
        rows = LogStore(DEFAULT_LOG_DB).rows(3)
        if rows:
            print()
            print(render_rows(rows))
        return

    # 反馈只标记某一题，历史分数一律不改——评估记录要能复盘。
    if words and words[0] == "feedback":
        if len(words) != 3:
            print("用法：/eval feedback 日志ID good|bad（例如 /eval feedback 2 bad）")
            return
        try:
            log_id = int(words[1])
        except ValueError:
            print(f"日志ID 必须是整数：{words[1]}")
            return
        try:
            set_feedback(log_id, words[2])
        except (ValueError, KeyError) as error:
            print(f"反馈未保存：{error}")
            return
        print(f"日志 {log_id} 的反馈已保存：{words[2]}")
        print(render_summary(read_summary()))
        return

    limit = EXPECTED_CASES
    if words:
        if not words[0].isdigit():
            print(
                "用法：/eval（跑 3 题）| /eval 1（只跑 1 题）"
                " | /eval summary | /eval feedback 日志ID good|bad"
            )
            return
        limit = int(words[0])
    if limit < 1:
        print("题数必须大于 0；用法：/eval（跑 3 题）| /eval 1（只跑 1 题）")
        return

    try:
        service, _store = build_eval_service()
    except (ImportError, FileNotFoundError, RuntimeError) as error:
        print(f"评估未执行：{error}")
        print("请先执行：uv sync --extra embedding --extra chroma --extra parse --extra evaluation")
        return

    inner = getattr(service.llm, "inner", service.llm)
    if type(inner).__name__ == "MockLLM":
        print()
        print("提示：当前是离线 mock 模式，MockLLM 的回答不带 [n] 引用编号，")
        print("     会被引用核验挡下，评分只是流程演示、不代表真实效果；")
        print("     想把本研究跑真实，在 .env 里把 LLM_PROVIDER 改成 deepseek。")

    run = run_evaluation(limit, service=service)
    print()
    print(render_run(run))
    print()
    print(render_rows(LogStore(run.log_path).rows(limit)))
    print()
    print(f"日志数据库：{run.log_path}")
    print("下一步：/eval summary 看汇总；/eval feedback 日志ID good|bad 标记单题反馈。")
    print("两类指标不是一回事：Precision@3/Recall@3 是规则算的（按标注来源），")
    print("RAGAS 三个分数是 DeepSeek 当评审给的模型判断。")


# ---------------------------------------------------------------- 单元14：ReAct Agent


def demo_react_agent(argument: str | None = None) -> None:
    """单元14：单事项 → 双事项 → 限轮次，最后做轨迹对比。

    三种用法对应任务卡的三段时间：不带参数单事项、`double` 双事项、`limit` 轮次受限；
    `demo` 一口气跑完三段（课件截图与证据脚本都用这条）。
    """
    try:
        from app.agent_lab import (
            DEFAULT_MAX_STEPS,
            DOUBLE_QUESTION,
            LIMITED_MAX_STEPS,
            SINGLE_QUESTION,
            build_agent,
            render_run,
            render_scenario,
            render_why_python,
            run_class,
            run_scenario,
        )
    except ImportError as exc:
        print(
            f"单元14 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra parse"
        )
        return

    payload = (argument or "").strip()
    command, _, rest = payload.partition(" ")
    command = command.lower()
    question = rest.strip()

    if command == "demo":
        try:
            run = run_class()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"Agent 未执行：{error}")
            return
        print(render_run(run))
        print()
        print("限制轮次只改变停在哪一轮，不改变模型会不会多查一次——那是成本约束，不是质量信号。")
        return

    # 三种场景都先准备好库与决策模型，之后只换问题与轮次上限。
    try:
        agent, store, count = build_agent(DEFAULT_MAX_STEPS)
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"Agent 未执行：{error}")
        return

    is_scripted = type(agent.decision_model).__name__ == "ScriptedDecisionModel"
    print(f"知识库 {count} 条（来自单元12 的本组资料）；决策模型：{type(agent.decision_model).__name__}")
    if is_scripted:
        print()
        print("提示：当前 LLM_PROVIDER=mock，用的是离线脚本策略，")
        print("     它的动作按“逐项检索、检索完汇总”写死，不代表模型真实推理；")
        print("     想看真实 DeepSeek 的决策，在 .env 里把 LLM_PROVIDER 改成 deepseek。")

    if command == "double":
        scenario, *_ = run_scenario("double", question or DOUBLE_QUESTION, DEFAULT_MAX_STEPS, agent=agent)
        print()
        print(render_scenario(scenario))
        print()
        print("双事项问题看的就是第 2 轮：模型只有看到第一轮观察后才会决定要不要再查一次。")
        return
    if command in ("limit", "limited", "max2"):
        scenario, *_ = run_scenario("limited", question or DOUBLE_QUESTION, LIMITED_MAX_STEPS, agent=agent)
        print()
        print(render_scenario(scenario))
        print()
        print(render_why_python())
        return
    # command 不是保留字时，整段 payload 就是用户自己的问题。
    scenario, *_ = run_scenario("single", payload or SINGLE_QUESTION, DEFAULT_MAX_STEPS, agent=agent)
    print()
    print(render_scenario(scenario))
    print()
    print("模型每轮只输出一个 JSON 动作，检索与来源核验都在 Python 里执行；")
    print("这也是为什么 finish 的来源必须是真的检索到的文件。")


# ---------------------------------------------------------------- 单元15：Agent 工具集


def demo_toolkit(argument: str | None = None) -> None:
    """单元15：单工具 → 连续调用 → 拒绝路径 → 计算工具。

    子命令对应任务卡的四段时间：不带参数单工具、`combo` 连续调用、
    `reject` 拒绝路径、`calc` 计算工具；`demo` 按课堂顺序一次跑完四段。
    """
    try:
        from app.tool_agent import DEFAULT_MAX_STEPS
        from app.tool_lab import (
            CALC_QUESTION,
            COMBO_QUESTION,
            SINGLE_QUESTION,
            build_kit,
            render_calc_note,
            render_run,
            render_scenario,
            run_class,
            run_scenario,
        )
    except ImportError as exc:
        print(
            f"单元15 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra parse"
        )
        return

    payload = (argument or "").strip()
    command, _, rest = payload.partition(" ")
    command = command.lower()
    question = rest.strip()

    if command == "demo":
        try:
            run = run_class()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"工具集未执行：{error}")
            return
        print(render_run(run))
        print()
        print("三个工具各管一类数据：检索靠相似度、FAQ 靠编号、日期靠 Python 算；")
        print("模型只提议调哪个，能不能执行永远由注册表说了算。")
        return

    # 其余子命令共用同一套数据源与决策模型，只换问题。
    try:
        agent, _registry, _store, kb_count, faq_rows = build_kit()
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"工具集未执行：{error}")
        return

    print(f"知识库 {kb_count} 条；FAQ 库 {faq_rows} 行；"
          f"决策模型：{type(agent.decision_model).__name__}")
    if type(agent.decision_model).__name__ == "ScriptedToolModel":
        print()
        print("提示：当前 LLM_PROVIDER=mock，用的是离线脚本策略（先查编号、再查制度、")
        print("     算之前先取时限），不代表模型真实推理；改 .env 里的 LLM_PROVIDER 可看真实决策。")

    if command in ("combo", "combine"):
        scenario, *_ = run_scenario("combo", question or COMBO_QUESTION, DEFAULT_MAX_STEPS, agent=agent)
        print()
        print(render_scenario(scenario))
        print()
        print("连续调用看的是第 2 轮：模型看到第 1 轮的观察后，才决定换另一个工具。")
        return
    if command in ("calc", "compute"):
        scenario, *_ = run_scenario("calc", question or CALC_QUESTION, DEFAULT_MAX_STEPS, agent=agent)
        print()
        print(render_scenario(scenario))
        print()
        print(render_calc_note())
        return
    if command in ("reject", "rebel"):
        from app.tool_lab import run_rejections

        scenario = run_rejections(agent, DEFAULT_MAX_STEPS)
        print()
        print(render_scenario(scenario))
        print()
        print("拒绝发生在执行之前：未授权的工具和格式不对的参数，永远不会碰到数据。")
        return
    # command 不是保留字时，整段 payload 就是用户自己的问题。
    scenario, *_ = run_scenario("single", payload or SINGLE_QUESTION, DEFAULT_MAX_STEPS, agent=agent)
    print()
    print(render_scenario(scenario))
    print()
    print("模型每轮只输出一个 JSON 工具调用，工具名与参数先过白名单，再由 Python 执行。")


# ---------------------------------------------------------------- 单元16：LangChain 组合链


def demo_chain(argument: str | None = None) -> None:
    """单元16：组合链 → 逐段流转 → 同题对照 → 资料不足与无效输入。

    子命令对应任务卡的四节课：不带参数跑组合链（第3节）、`steps` 看逐段流转（第1、2节）、
    `compare` 同题对照单元15 的 ToolAgent、`empty`/`bad` 看异常场景（第4节）、
    `demo` 按课堂顺序一次跑完。
    """
    try:
        from app.chain_lab import (
            DEFAULT_QUESTION,
            INSUFFICIENT_QUESTION,
            build_stack,
            compare_with_agent,
            render_comparison,
            render_invalid,
            render_reusable,
            render_run,
            render_scenario,
            run_class,
            run_invalid,
            run_scenario,
        )
        from app.lc_chain import MIN_SIMILARITY, render_stage_table
    except ImportError as exc:
        print(
            f"单元16 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra langchain"
        )
        return

    payload = (argument or "").strip()
    command, _, rest = payload.partition(" ")
    command = command.lower()
    question = rest.strip()

    if command == "demo":
        try:
            run = run_class()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"组合链未执行：{error}")
            return
        print(render_run(run))
        print()
        print("六段里只有第⑤段是模型，其余五段都是 Python：检索、字段、提示词、解析各管一头；")
        print("所以换模型只是换第⑤段，资料和来源口径一点没变。")
        return

    try:
        from app.chroma_store import PROJECT_ID

        parts, service, store = build_stack()
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"组合链未执行：{error}")
        return

    print(f"知识库 {store.count(PROJECT_ID)} 条；证据门槛 余弦相似度 ≥ {MIN_SIMILARITY}；"
          f"模型：{parts.model.model_name if hasattr(parts.model, 'model_name') else type(parts.model).__name__}")
    if type(parts.model).__name__ == "OfflineChatModel":
        print()
        print("提示：当前 LLM_PROVIDER=mock，链上第⑤段用的是离线摘抄模型——")
        print("     它只把证据原文抄进答案，不代表模型的真实生成能力；改 .env 的 LLM_PROVIDER 可看真实回答。")

    if command in ("steps", "stage", "flow"):
        scenario = run_scenario("steps", question or DEFAULT_QUESTION, parts, inspect_stages=True)
        print()
        print(render_scenario(scenario))
        print()
        print(render_stage_table())
        print()
        print("assign 是加字段、不是换值：问题一路都在，所以最后还能把来源一并打印出来。")
        return
    if command in ("compare", "cmp"):
        comparison = compare_with_agent(question or DEFAULT_QUESTION, parts, service)
        print()
        print(render_comparison(comparison))
        print()
        print("对照的是来源和时限数字这类硬事实；模型措辞不同不算问题，数字不同才要人工核对。")
        return
    if command in ("empty", "insufficient", "lack"):
        scenario = run_scenario("empty", question or INSUFFICIENT_QUESTION, parts)
        print()
        print(render_scenario(scenario))
        print()
        print("证据不足时链照样跑完，但递给模型的是「未检索到相关资料」——答案只能是资料不足。")
        return
    if command in ("bad", "invalid"):
        print()
        print(render_invalid(run_invalid(parts)))
        print()
        print("守门在链之前：问题不合格就不检索、不调用模型，链一次都没跑。")
        return

    scenario = run_scenario("answer", payload or DEFAULT_QUESTION, parts)
    print()
    print(render_scenario(scenario))
    print()
    print(render_reusable())
    print()
    print("链只是把已有能力串起来：检索还是单元09 的那个函数，资料还是单元12 的那批真实文件。")


# ---------------------------------------------------------------- 单元17：LangChain Agent 执行器


def demo_lcagent(argument: str | None = None) -> None:
    """单元17：执行器 → 连续调用 → 工具错误与步数上限 → 同题对照单元15 的循环。

    子命令对应任务卡的四节课：不带参数跑最小业务场景（第2节）、`combo` 看连续调用
    与步骤追踪（第3节）、`error`/`limit`/`reject` 看异常路径（第3、4节）、
    `compare` 对照单元15 手写的循环、`demo` 按课堂顺序一次跑完。
    """
    try:
        from app.lc_agent import render_executor
        from app.lc_agent_lab import (
            BAD_ARG_QUESTION,
            COMBO,
            LIMIT_MAX_STEPS,
            LIMIT_QUESTION,
            MINIMAL_QUESTION,
            PLANS,
            UNKNOWN_QUESTION,
            build_stack,
            compare_with_legacy,
            render_comparison,
            render_concepts,
            render_invalid,
            render_run,
            render_scenario,
            render_tool_binding,
            run_class,
            run_invalid,
            run_scenario,
        )
    except ImportError as exc:
        print(
            f"单元17 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra langchain"
        )
        return

    payload = (argument or "").strip()
    command, _, rest = payload.partition(" ")
    command = command.lower()
    question = rest.strip()

    if command == "demo":
        try:
            run = run_class()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"执行器未运行：{error}")
            return
        print(render_run(run))
        print()
        print("工具还是单元15 那三项、资料还是单元12 那 7 条；换的是谁来跑循环——")
        print("手写 while 换成了框架的执行器，每一步仍留在日志里，出错能定位到第几步。")
        return

    try:
        executor, legacy_agent, registry, _store, kb_count, faq_rows = build_stack()
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"执行器未运行：{error}")
        return

    if command in ("tools", "kit", "config"):
        print()
        print(render_executor(executor))
        print()
        print(render_tool_binding(registry, executor))
        print()
        print(render_concepts())
        return

    print(f"知识库 {kb_count} 条；FAQ 库 {faq_rows} 行；工具 {len(executor.tools)} 项；"
          f"步数上限 {executor.max_steps}；模型：{executor.model.__class__.__name__}")
    if executor.model.__class__.__name__ == "ScriptedToolModel":
        print()
        print("提示：当前 LLM_PROVIDER=mock，执行器里用的是离线剧本模型（按剧本提工具调用）——")
        print("     它不代表模型真实的选型能力，但工具执行、步骤日志与步数上限都是真的；")
        print("     改 .env 里的 LLM_PROVIDER 可看真实模型自己选工具。")

    if command in ("combo", "combine"):
        scenario = run_scenario("combo", question or COMBO, executor)
        print()
        print(render_scenario(scenario))
        print()
        print("第2步是在第1步的观察回来之后才发生的：这就是执行器替我们跑的循环。")
        return
    if command in ("error", "fail"):
        scenario = run_scenario("error", question or BAD_ARG_QUESTION, executor,
                                plan=PLANS["error"], scripted=True)
        print()
        print(render_scenario(scenario))
        print()
        print("报错没有让演示中断：失败原因连同步号一起回填给模型，所以能定位到第几步。")
        return
    if command in ("limit", "max"):
        scenario = run_scenario("limit", question or LIMIT_QUESTION, executor,
                                plan=PLANS["limit"], max_steps=LIMIT_MAX_STEPS, scripted=True)
        print()
        print(render_scenario(scenario))
        print()
        print("触顶就是没答完：回答为空、执行器给出上限提示，不拿它当答案用。")
        return
    if command in ("reject", "rebel", "unknown"):
        scenario = run_scenario("unknown", question or UNKNOWN_QUESTION, executor,
                                plan=PLANS["unknown"], scripted=True)
        print()
        print(render_scenario(scenario))
        print()
        print("工具清单里没有的名字，执行器根本不执行——模型只能从注册表里挑。")
        return
    if command in ("compare", "cmp"):
        comparison = compare_with_legacy(question or MINIMAL_QUESTION, executor, legacy_agent)
        print()
        print(render_comparison(comparison))
        print()
        print("两条路径用的是同一批工具与同一份资料；比的是工具与数字，不是措辞。")
        return
    if command in ("bad", "invalid"):
        print()
        print(render_invalid(run_invalid()))
        print()
        print("守门在跑执行器之前：问题不合格就不调用模型、也不碰任何工具。")
        return

    scenario = run_scenario("minimal", payload or MINIMAL_QUESTION, executor)
    print()
    print(render_scenario(scenario))
    print()
    print("最小场景只用到一个工具：先跑通它，再往上加连续调用与异常路径。")


def demo_flow(argument: str | None = None) -> None:
    """单元18：路由分支 → 取证据 → 摘要润色 → 兜底；每步输入输出都打印出来。

    子命令对应任务卡的四节课：`routes` 看流程图与分支规则（第1、2节）、
    不带参数跑最小业务场景（第2节）、`steps` 逐段看每步输入输出（第3节）、
    `faq`/`deadline`/`unknown`/`empty`/`bad` 逐条验证分支与兜底（第4节）。
    """
    try:
        from app.flow_chain import render_concepts, render_flow, render_routes
        from app.flow_lab import (
            DEADLINE_QUESTION,
            FAQ_QUESTION,
            NO_EVIDENCE_QUESTION,
            UNKNOWN_QUESTION,
            build_stack,
            render_scenario,
            render_summary_table,
            run_class,
            run_invalid,
            run_scenario,
        )
    except ImportError as exc:
        print(
            f"单元18 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra langchain"
        )
        return

    payload = (argument or "").strip()
    command, _, rest = payload.partition(" ")
    command = command.lower()
    question = rest.strip()

    if command == "demo":
        try:
            run = run_class()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"工作流未运行：{error}")
            return
        print(render_flow())
        print()
        print(render_routes())
        print()
        for scenario in run.scenarios:
            print(render_scenario(scenario))
            print()
        print(render_summary_table(run))
        print()
        print("资料还是单元12 那 7 条、工具还是单元15 那三项；换的是编排——")
        print("谁先做、走哪条分支由程序写死，模型只负责摘要与润色这两步。")
        return

    try:
        parts, _registry, _service, kb_count, faq_rows = build_stack()
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"工作流未运行：{error}")
        return

    if command in ("routes", "branches", "config"):
        print()
        print(render_flow())
        print()
        print(render_routes())
        print()
        print(render_concepts())
        return

    model_steps = getattr(parts.summary_chain, "steps", [])
    model_name = type(model_steps[1]).__name__ if len(model_steps) >= 2 else "（未知）"
    print(f"知识库 {kb_count} 条；FAQ 库 {faq_rows} 行；模型：{model_name}")
    print("提示：LLM_PROVIDER=mock 时摘要与润色由离线模型完成（只摘录证据，不生成新事实）；")
    print("     路由、分支判断、兜底与步骤表都不依赖模型，离线与真实模式完全一致。")

    if command in ("faq", "id"):
        scenario = run_scenario("faq", question or FAQ_QUESTION, parts)
        print()
        print(render_scenario(scenario))
        print()
        print("按编号命中一行就走人：这条路不问向量库，也不会被相似度门槛卡住。")
        return
    if command in ("deadline", "calc", "date"):
        scenario = run_scenario("deadline", question or DEADLINE_QUESTION, parts)
        print()
        print(render_scenario(scenario))
        print()
        print("天数来自资料原文，日期加法由 Python 做：模型只写摘要，不参与算日子。")
        return
    if command in ("unknown", "fallback", "reject"):
        scenario = run_scenario("unknown", question or UNKNOWN_QUESTION, parts)
        print()
        print(render_scenario(scenario))
        print()
        print("三条规则都没命中就不猜：不检索、不调模型（模型调用 0 次），固定话术转人工。")
        return
    if command in ("empty", "noevidence", "insufficient"):
        scenario = run_scenario("no_evidence", question or NO_EVIDENCE_QUESTION, parts)
        print()
        print(render_scenario(scenario))
        print()
        print("分支到了但证据过不了门槛：同样 0 次模型调用，不拿最近邻硬凑答案。")
        return
    if command in ("bad", "invalid"):
        print()
        print("输入\t结果")
        for label, verdict in run_invalid(parts):
            print(f"{label}\t{verdict}")
        print()
        print("守门在跑工作流之前：问题不合格就既不检索、也不调模型。")
        return
    if command in ("steps", "trace"):
        scenario = run_scenario("knowledge", question or "普通商品退货有哪些要求？", parts)
        print()
        print(render_scenario(scenario))
        print()
        print("步骤表就是「每步输入输出可查看」这条标准的证据：几步、花了多久、为什么这么判。")
        return

    scenario = run_scenario("knowledge", payload or "普通商品退货有哪些要求？", parts)
    print()
    print(render_scenario(scenario))
    print()
    print("最小场景先跑通一条分支，再往上加分支、兜底与异常路径。")


# 连续追问要跨命令留住执行器与记忆，所以放在模块级：`/memory 你的问题` 每敲一次
# 就是同一会话的下一轮（换会话用 `/memory new 会话名`）。
_MEMORY_STATE: dict = {}


def _memory_state():
    """延迟装配执行器与记忆管理器；只有第一次敲 `/memory` 时才建库、建图。"""
    if "executor" not in _MEMORY_STATE:
        from app.memory_lab import build_memory_manager, build_stack

        executor, _registry, _store, kb_count, faq_rows = build_stack()
        _MEMORY_STATE.update(
            executor=executor,
            manager=build_memory_manager(),
            session=None,
            kb_count=kb_count,
            faq_rows=faq_rows,
        )
    return _MEMORY_STATE


def demo_memory(argument: str | None = None) -> None:
    """单元19：读 → 答 → 写的多轮问答；记忆只管指代，事实仍由工具现查。

    子命令对应任务卡的四节课：`flow` 看流程图与两类库的分工（第1节）、
    不带参数跑两轮连续问答（第2节）、`modes`/`restart` 对比三种模式与跨运行召回
    （第3节）、`isolation`/`bad` 验证隔离与输入守门（第4节）。
    """
    try:
        from app.memory_chain import (
            MODE_HYBRID,
            MODES,
            render_memory_flow,
            render_modes,
            render_snapshot,
            run_turn,
        )
        from app.memory_lab import (
            DEFAULT_SESSION,
            PAIR_QUESTIONS,
            SESSION_A,
            SESSION_B,
            TURN1_QUESTION,
            build_memory_manager,
            build_stack,
            recall_after_restart,
            render_concepts,
            render_restart,
            render_run,
            run_class,
            run_isolation,
            run_mode_compare,
            run_session,
        )
    except ImportError as exc:
        print(
            f"单元19 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra langchain"
        )
        return

    payload = (argument or "").strip()
    command, _, rest = payload.partition(" ")
    command = command.lower()
    text = rest.strip()

    if command == "demo":
        try:
            run = run_class()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"记忆流程未运行：{error}")
            return
        print(render_run(run))
        return

    if command in ("flow", "concepts", "config"):
        print()
        print(render_memory_flow())
        print()
        print(render_concepts())
        print()
        print("记忆库与知识库分开，是为了不把上一轮说过的话当成制度原文引用。")
        return

    if command == "modes":
        try:
            executor, registry, _store, _kb, _rows = build_stack()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"记忆流程未运行：{error}")
            return
        records = run_mode_compare(executor, registry=registry)
        print()
        print(render_modes())
        print()
        print("（三种模式各跑两轮，第2轮都问「刚才那项」）")
        for record in records:
            resolution = record.snapshot.resolution
            print(
                f"{record.mode}\t第{record.index + 1}轮｜带进对话 {len(record.snapshot.history)} 条｜"
                f"召回 {len(record.snapshot.recalled)} 条｜"
                f"解析出 {resolution.object_id if resolution else '（未解析）'}｜"
                f"工具 {'、'.join(record.tools) if record.tools else '（无）'}"
            )
        print()
        print("完整历史与摘要只在内存里，向量记忆落盘：这就是「重启后还记不记得」的分界。")
        return

    if command in ("restart", "recall"):
        count, recalled = recall_after_restart()
        print()
        print(render_restart(count, recalled))
        print()
        print("想看真正的跨进程：另开一个终端再敲一次 /memory restart，读到的还是这几条。")
        return

    if command in ("isolation", "isolate", "sessions"):
        try:
            executor, registry, _store, _kb, _rows = build_stack()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"记忆流程未运行：{error}")
            return
        (first, second), manager = run_isolation(executor)
        print()
        print(f"会话 {SESSION_A}：带回来历史 {len(first.snapshot.history)} 条、"
              f"召回 {len(first.snapshot.recalled)} 条")
        print(f"会话 {SESSION_B}：带回来历史 {len(second.snapshot.history)} 条、"
              f"召回 {len(second.snapshot.recalled)} 条")
        print(f"库里 {SESSION_A} 有 {manager.long_term_count(SESSION_A)} 条、"
              f"{SESSION_B} 有 {manager.long_term_count(SESSION_B)} 条（过滤在库端做）")
        print()
        print("同一个管理器、两个会话：第二个会话问同样的问题，一条历史也拿不到。")
        return

    if command in ("bad", "invalid"):
        try:
            state = _memory_state()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"记忆流程未运行：{error}")
            return
        print()
        print("输入\t结果")
        for label, verdict in _memory_invalid_cases(state["manager"]):
            print(f"{label}\t{verdict}")
        print()
        print("会话 ID 守门在读写之前：不合格的 ID 连库都不会碰。")
        return

    if command == "new":
        state = _memory_state()
        state["session"] = text or DEFAULT_SESSION
        state["manager"] = build_memory_manager()
        print(f"已切到新会话：{state['session']}（记忆清空，重新开始记）")
        return

    try:
        state = _memory_state()
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"记忆流程未运行：{error}")
        return
    executor, manager = state["executor"], state["manager"]
    session = state.get("session") or DEFAULT_SESSION
    print(f"会话：{session}；资料 {state['kb_count']} 条；FAQ {state['faq_rows']} 行；"
          f"记忆写入 .chroma_unit19_memory/")
    print("（长期记忆是落盘的：看到前几次运行的轮次属正常，换新会话敲 /memory new 甲）")

    if command in MODES:
        # 指定模式：把两轮连着跑完，第二轮刻意不写编号（「刚才那项」）。
        for record in run_session(executor, manager, f"{session}-{command}", PAIR_QUESTIONS, command):
            print()
            print(f"第 {record.index + 1} 轮：{record.question}")
            print(f"  本轮调用的工具：{'、'.join(record.tools) if record.tools else '（无'}"
                  f"{'）' if not record.tools else '）'}")
            print(f"  答：{record.answer or '（空）'}")
            print(f"  {record.save_note}")
        print()
        print("第2轮没写编号也能答：记忆负责「哪一项」，答案仍由工具现查。")
        return

    question = payload or TURN1_QUESTION
    try:
        record = run_turn(executor, manager, session, question, MODE_HYBRID)
    except (ValueError, RuntimeError) as error:
        print(f"本轮没有运行：{error}")
        return
    print()
    print(render_snapshot(record.snapshot))
    print()
    print(f"第 {record.index + 1} 轮｜工具：{'、'.join(record.tools) if record.tools else '（无）'}")
    print(f"答：{record.answer or '（空）'}")
    print(f"{record.save_note}")
    print()
    print("接着问（比如「刚才那项，时限是多久？」）就是下一轮，记忆会自己接上。")


def _memory_invalid_cases(manager) -> list[tuple[str, str]]:
    """第4节的输入守门：会话 ID 与问题在进入记忆之前就被拦下。"""
    from app.memory_chain import MODE_HYBRID

    cases: list[tuple[str, str]] = []
    for label, raw in (
        ("空会话 ID", ""),
        ("含空格的会话 ID", "student a"),
        ("含中文的会话 ID", "学生甲"),
        ("超长会话 ID", "x" * 80),
    ):
        try:
            manager.load(raw, "EC-FAQ-01 的标准答复是什么？", MODE_HYBRID)
            verdict = "未被拦下（有问题）"
        except ValueError as error:
            verdict = f"已拒绝：{error}"
        cases.append((label, verdict))
    for label, question in (
        ("空问题", ""),
        ("纯空白问题", "   "),
        ("非字符串问题", 123),
        ("未知记忆模式", "EC-FAQ-01 的答复是什么"),
    ):
        try:
            mode = "not-a-mode" if label == "未知记忆模式" else MODE_HYBRID
            manager.load("class-demo", question, mode)
            verdict = "未被拦下（有问题）"
        except (ValueError, TypeError) as error:
            verdict = f"已拒绝：{type(error).__name__}：{error}"
        cases.append((label, verdict))
    return cases


def demo_plan(argument: str | None = None) -> None:
    """单元20：模型只出计划，Python 逐步执行；失败至多重规划一次，仍失败就终止报错。

    子命令对应任务卡的四节课：`flow` 看流程图与概念（第1节）、不带参数跑最小
    场景（第2节）、`multi` 跑多步计划看每步观察（第3节）、`replan`/`abort`/
    `missing`/`bad` 验证重规划、终止与守门（第4节）。
    """
    try:
        from app.plan_lab import (
            build_stack,
            render_flow,
            render_header,
            render_invalid,
            render_question_list,
            render_scenario,
            render_standards,
            render_summary_table,
            run_class,
            run_invalid,
            run_scenario,
        )
        from app.plan_solver import render_concepts
    except ImportError as exc:
        print(
            f"单元20 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra langchain"
        )
        return

    payload = (argument or "").strip()
    command, _, rest = payload.partition(" ")
    command = command.lower()
    goal = rest.strip()

    if command == "demo":
        try:
            run = run_class()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"规划流程未运行：{error}")
            return
        print(render_header(run))
        print()
        print(render_flow())
        print()
        print(render_question_list())
        print()
        for scenario in run.scenarios:
            print(render_scenario(scenario))
            print()
        print(render_summary_table(run))
        print()
        print(render_standards(run))
        print()
        print(render_invalid(run.invalid_rows))
        print()
        print("计划是数据不是指令：模型只能挑白名单里的工具并填参数，路径、SQL、代码一概碰不到。")
        print("没解决的失败就报错：状态 aborted、答复为空、连总结那次模型调用都不发生。")
        return

    try:
        registry, _model, kb_count, faq_rows = build_stack()
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"规划流程未运行：{error}")
        return

    if command in ("flow", "concepts", "config"):
        print()
        print(render_flow())
        print()
        print(render_concepts())
        print()
        print("计划与执行分开的意义：计划错了能改，执行错了能定位到第几步。")
        return

    if command in ("bad", "invalid"):
        print()
        print(render_invalid(run_invalid(registry)))
        print()
        print("守门在问模型之前：目标不合格就一次规划都不发。")
        return

    print(f"知识库 {kb_count} 条；FAQ 库 {faq_rows} 行；工具仍是单元15 那三项（注册表未改动）")
    if (os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower() == "deepseek":
        print("当前是 DeepSeek 真实模式：计划由模型自己拆；")
        print("replan/abort/missing 三个场景是**故障注入**（强制用剧本计划）——真实模型会把")
        print("BAD 编号自己改对，那就演示不到失败、重规划与终止了。")
    else:
        print("提示：LLM_PROVIDER=mock 时计划与答复由离线剧本给出（只摘录工具返回的原文）；")
        print("     真实模式在 .env 里把 LLM_PROVIDER 改成 deepseek，计划由模型自己拆。")

    if command in ("minimal", "multi", "replan", "abort", "missing"):
        scenario = run_scenario(command, goal or None, registry=registry)
    else:
        # 整句都是目标：自定义目标跑一次规划。
        scenario = run_scenario("minimal", payload or None, registry=registry)
    print()
    print(render_scenario(scenario))
    print()
    if scenario.run is not None and scenario.run.replanned:
        print("这次失败过一次：重规划只有一次机会，改完从第 1 步重新跑，旧观察留着当证据。")
    else:
        print("计划里的每一步都对应一次真实工具调用；答复只许用观察里出现过的事实。")


def demo_crew(argument: str | None = None) -> None:
    """单元21：五个专职角色 + 共享黑板 + 协调者；评审不通过只修订一次。

    子命令对应任务卡的四节课：`roles` 看架构图与角色表（第1节）、不带参数跑一次
    正常协作（第2、3节）、`revise`/`reject`/`gap`/`abort`/`bad` 验证评审回路与
    终止（第4节）。
    """
    try:
        from app.multi_agent import MAX_REVISIONS, render_concepts, render_flow, render_roles
        from app.multi_lab import (
            build_stack,
            render_header,
            render_invalid,
            render_question_list,
            render_scenario,
            render_standards,
            render_summary_table,
            run_class,
            run_invalid,
            run_scenario,
        )
    except ImportError as exc:
        print(
            f"单元21 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra langchain"
        )
        return

    payload = (argument or "").strip()
    command, _, rest = payload.partition(" ")
    command = command.lower()
    goal = rest.strip()

    if command == "demo":
        try:
            run = run_class()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"多智能体协作未运行：{error}")
            return
        print(render_header(run))
        print()
        print(render_flow())
        print()
        print(render_roles())
        print()
        print(render_question_list())
        print()
        for scenario in run.scenarios:
            print(render_scenario(scenario))
            print()
        print(render_summary_table(run))
        print()
        print(render_standards(run))
        print()
        print(render_invalid(run.invalid_rows))
        print()
        print(f"答复智能体只有草稿、评审智能体才有放行权；评审不通过最多修订 {MAX_REVISIONS} 次。")
        print("修订后仍不通过就停止自动交付：答复留空、状态转人工，草稿与评审意见留给人工复核。")
        return

    try:
        registry, _model, kb_count, faq_rows = build_stack()
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"多智能体协作未运行：{error}")
        return

    if command in ("roles", "flow", "concepts", "config"):
        print()
        print(render_flow())
        print()
        print(render_roles())
        print()
        print(render_concepts())
        print()
        print("角色之间不互相调用：全部通过共享黑板交接，每条消息都留发送者。")
        return

    if command in ("bad", "invalid"):
        print()
        print(render_invalid(run_invalid(registry)))
        print()
        print("守门在问任何角色之前：目标不合格就一个角色都不启动。")
        return

    print(f"知识库 {kb_count} 条；FAQ 库 {faq_rows} 行；工具仍是单元15 那三项（注册表未改动）")
    if (os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower() == "deepseek":
        print("当前是 DeepSeek 真实模式：计划、答复、评审都由模型自己写；")
        print("revise/reject/gap/abort 四个场景是**故障注入**（草稿与评审结论用剧本）——")
        print("真实模型不会故意写出不合格的草稿，那就演示不到拒绝、修订与停止交付了。")
    else:
        print("提示：LLM_PROVIDER=mock 时计划与评审由离线剧本给出（答复只摘录工具返回的原文）；")
        print("     真实模式在 .env 里把 LLM_PROVIDER 改成 deepseek，五个角色都换成真实模型。")

    if command in ("deliver", "revise", "reject", "gap", "abort"):
        scenario = run_scenario(command, goal or None, registry=registry)
    else:
        # 整句都是目标：自定义目标跑一次完整协作。
        scenario = run_scenario("deliver", payload or None, registry=registry)
    print()
    print(render_scenario(scenario))
    print()
    board = scenario.board
    if board is None:
        print(f"未运行：{scenario.error}")
    elif not board.delivered:
        print("未通过评审的草稿没有变成答复：答复留空，草稿与评审意见留在黑板上等人工复核。")
    elif board.revisions:
        print(f"这次修订过 {board.revisions} 次（上限 {MAX_REVISIONS}）：修订只改表述，不改工具事实。")
    else:
        print("五个角色各写各的产物，全部经黑板交接；评审通过才由协调者交付。")


def demo_group(argument: str | None = None) -> None:
    """单元22：三个角色在同一间群聊里轮流发言，轮次与成本双上限，复核通过才交付。

    子命令对应任务卡的四节课：`roles` 看群聊时序与角色表（第1节）、不带参数跑一次
    正常协作（第2节）、`tight` 看轮次上限（第3节）、`budget`/`reject`/`gap`/`bad`
    验证成本上限、复核不通过与守门（第4节）。
    """
    try:
        from app.chat_crew import (
            DEFAULT_TOKEN_BUDGET,
            MAX_ROUNDS,
            render_concepts,
            render_flow,
        )
        from app.chat_lab import (
            build_stack,
            render_agents,
            render_guards,
            render_header,
            render_invalid,
            render_question_list,
            render_scenario,
            render_standards,
            render_state_table,
            render_summary_table,
            run_class,
            run_guards,
            run_invalid,
            run_scenario,
        )
    except ImportError as exc:
        print(
            f"单元22 依赖未安装（{exc}）。"
            "请先执行：uv sync --extra embedding --extra chroma --extra langchain"
        )
        return

    payload = (argument or "").strip()
    command, _, rest = payload.partition(" ")
    command = command.lower()
    goal = rest.strip()

    if command == "demo":
        try:
            run = run_class()
        except (ImportError, RuntimeError, ValueError) as error:
            print(f"群聊协作未运行：{error}")
            return
        print(render_header(run))
        print()
        print(render_flow())
        print()
        print(render_agents())
        print()
        print(render_question_list())
        print()
        for scenario in run.scenarios:
            print(render_scenario(scenario))
            print()
        print(render_summary_table(run))
        print()
        print(render_state_table(run))
        print()
        print(render_standards(run))
        print()
        print(render_invalid(run.invalid_rows))
        print()
        print(render_guards(run.guard_rows))
        print()
        print(render_concepts())
        print()
        print("执行员写的是草稿：复核不发言或不通过，答复就是空的。")
        print("轮次与成本都是 Python 判断：停下来时状态一定是「未交付」，不补一轮、不硬撑。")
        return

    try:
        registry, _model, kb_count, faq_rows = build_stack()
    except (ImportError, RuntimeError, ValueError) as error:
        print(f"群聊协作未运行：{error}")
        return

    if command in ("roles", "flow", "concepts", "config"):
        print()
        print(render_flow())
        print()
        print(render_agents())
        print()
        print(render_concepts())
        print()
        print(f"轮次上限 {MAX_ROUNDS} 轮；常规成本预算 {DEFAULT_TOKEN_BUDGET}（估算 token）。")
        print("同一间群聊里后发言的人看得到前面所有人的原话，这是消息流的可见性。")
        return

    if command in ("bad", "invalid"):
        print()
        print(render_invalid(run_invalid(registry)))
        print()
        print(render_guards(run_guards(registry)))
        print()
        print("守门在让任何角色开口之前：任务或参数不合格，一次模型调用都不发。")
        return

    print(f"知识库 {kb_count} 条；FAQ 库 {faq_rows} 行；工具仍是单元15 那三项（注册表未改动）")
    if (os.getenv("LLM_PROVIDER", "mock") or "mock").strip().lower() == "deepseek":
        print("当前是 DeepSeek 真实模式：三位角色的发言都由模型自己写；")
        print("tight/budget/reject/gap 四个场景是**故障注入**（发言与复核结论用剧本）——")
        print("真实模型不会故意少说一轮、也不会自己写一份不合格的草稿，那就演示不到终止条件了。")
    else:
        print("提示：LLM_PROVIDER=mock 时发言由离线剧本给出（执行员只摘录工具返回的原文）；")
        print("     真实模式在 .env 里把 LLM_PROVIDER 改成 deepseek，三个角色都换成真实模型。")

    if command in ("deliver", "tight", "budget", "reject", "gap"):
        scenario = run_scenario(command, goal or None, registry=registry)
    else:
        # 整句都是任务：自定义任务跑一次完整协作。
        scenario = run_scenario("deliver", payload or None, registry=registry)
    print()
    print(render_scenario(scenario))
    print()
    collab = scenario.collaboration
    if collab is None:
        print(f"未运行：{scenario.error}")
    elif not collab.delivered:
        print(f"这次没有交付（{collab.stop_reason}）：{collab.stop_note}")
        print("草稿仍在消息流里，可逐条追溯——停下来不等于丢掉证据。")
    else:
        print("复核通过才把执行员的草稿变成答复：放行权在评审员，不在写草稿的人手里。")


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
        if command == "/chunk":
            demo_rag_optimized(payload or None)
            continue
        if command == "/parse":
            demo_parse(payload or None)
            continue
        if command == "/eval":
            demo_evaluation(payload or None)
            continue
        if command == "/act":
            demo_react_agent(payload or None)
            continue
        if command == "/tools":
            demo_toolkit(payload or None)
            continue
        if command == "/chain":
            demo_chain(payload or None)
            continue
        if command == "/lcagent":
            demo_lcagent(payload or None)
            continue
        if command == "/flow":
            demo_flow(payload or None)
            continue
        if command == "/memory":
            demo_memory(payload or None)
            continue
        if command == "/plan":
            demo_plan(payload or None)
            continue
        if command == "/agents":
            demo_crew(payload or None)
            continue
        if command == "/group":
            demo_group(payload or None)
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