"""电商售后服务智能体：单元02—06 累计入口。

单元02 流式输出与多轮对话：/chat 走纯 stream 模式，保留多轮上下文。
单元03 Prompt工程与模板：/compare 用同一份订单数据、同一个问题对比差/好提示词。
单元04 Function Calling：默认走工具闭环，先查工具再回答，/trace 看调用轨迹。
单元05 工具注册与调度：所有工具统一登记在 ToolRegistry，/registry 看登记表，
/dispatch 不访问模型直接验证调度边界，/log 看带轮次与异常的调用日志。
单元06 Embedding与向量化：/embed 用本地中文模型把样例文本转成向量并做相似度排序，
本单元只比较"像不像"，不做问答、不读知识库、不做业务判断。

直接输入问题 = 工具闭环模式（这是本项目跑通业务闭环的主路径）。
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

# 不强制覆盖已有环境变量：命令行 `LLM_PROVIDER=mock` 仍能切回离线模式，
# .env 只负责补齐未设置的项（如 DEEPSEEK_API_KEY）。
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
        # 单元05：会话持有自己的注册表，模型可见清单与调用日志都从这里出。
        self.registry = registry or build_registry()
        self.chat = Conversation()
        self.dialog: list[dict] = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}]
        self.last_trace: list[dict] = []
        # 追问时用户常省略订单号，记住本会话最近一次提到的订单。
        self.last_order_id: str | None = None

    def resolve_context(self, question: str) -> str:
        """把订单事实渲染进提示词：流式模式没有工具可用，这一步是它的取数入口。"""
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
            print("--- 工具调用轨迹 ---")
            print(format_trace(trace))
            print("--- 最终回答 ---")
        print_streamed(answer)
        return answer

    # ------------------------------------------------ 单元02：多轮流式

    def ask_stream(self, question: str) -> str:
        self.chat.add("user", question)

        # 流式模式不调工具，改用模板把订单事实注入本轮用户消息；
        # history 里仍保存原始提问，保证 /history 可读。
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
    """Unit04 验收：工具名与参数可校验，缺参时给出清晰错误。"""
    print("工具参数白名单与必填校验演示：")
    for call in MISSING_ARGUMENT_DEMOS:
        print(f"\n调用：{call['name']}  参数：{json.dumps(call['arguments'], ensure_ascii=False)}")
        try:
            result = execute_tool_call(call)
            print(f"  校验通过 -> 返回结果：{json.dumps(result, ensure_ascii=False)}")
        except ToolError as exc:
            print(f"  校验失败 -> {exc}")


def _unstable_handler(order_id: str) -> dict:
    """仅供演示：模拟处理函数抛出的非 ToolError 异常。"""
    raise RuntimeError("模拟数据库连接中断")


# 单元05 调度演示：不访问模型，直接验证注册表边界。
DISPATCH_DEMOS = [
    {"call_id": "demo-1", "name": "query_order", "arguments": {"order_id": "SO-2001"}},
    {"call_id": "demo-2", "name": "lookup_policy_doc", "arguments": {"doc_id": "DOC-02-05"}},
    {"call_id": "demo-3", "name": "refund_order", "arguments": {"order_id": "SO-2001"}},
    {"call_id": "demo-4", "name": "check_return_eligibility", "arguments": {"order_id": "SO-2001"}},
]


def demo_dispatch() -> None:
    """单元05 验收：两个业务工具可调度；未知工具被拒绝；异常不中断其他请求。"""
    registry = build_registry()
    print(f"已注册工具（{len(registry.names())} 个）：{', '.join(registry.names())}\n")

    print("场景一：一批请求里同时有成功与失败，逐条处理互不牵连。")
    for step in registry.dispatch(DISPATCH_DEMOS, max_calls=4, round_index=1):
        print(f"\n请求 {step['call_id']} -> {step['name']}")
        print(f"  参数：{json.dumps(step['arguments'], ensure_ascii=False)}")
        print(f"  状态：{step['status']}（{step['elapsed_ms']}ms）")
        print(f"  结果：{json.dumps(step['result'], ensure_ascii=False)[:220]}")

    # 额外登记一个"会抛异常"的工具，验证兜底分支；它只存在于这个临时注册表里。
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
    """离线演示：模型漏传参数 -> 工具报错回填 -> 模型补全后成功。"""
    if not isinstance(llm, MockLLM):
        print("当前是 DeepSeek 模式，无法人为制造漏参；已改为展示静态校验。")
        demo_argument_validation()
        return

    sloppy_llm = MockLLM(omit_argument="reason")
    session = AgentSession(sloppy_llm)
    print("演示：模型第一轮漏传 reason，工具报错后自动补全重试。")
    session.ask_with_tools("订单SO-2001已签收，我想退货，需要什么条件？", show_trace=True)


def demo_embedding(query: str | None = None) -> None:
    """单元06 验收链：真实文本 -> 向量 -> 排序 -> 边界核对 -> 误判风险。"""
    # 延迟导入：没装 embedding 依赖时，其余单元的命令仍然可用。
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
        except Exception as exc:  # noqa: BLE001 - 交互入口的最后一道兜底
            # 单元05 要求：任何工具或模型异常都不能让主程序退出，会话要继续可用。
            print(f"助手：本次处理出现未预期异常（{type(exc).__name__}: {exc}）。")
            print("      会话已保留，请稍后重试或转人工客服。")


if __name__ == "__main__":
    main()
