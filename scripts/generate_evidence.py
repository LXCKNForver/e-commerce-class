"""一键生成 docs/课堂验收证据.md。

会真实调用 .env 里配置的模型跑一遍单元02—05 的演示路径，运行前确认 DEEPSEEK_API_KEY 可用；
只想离线看结构，可在命令行覆盖：

用法：
    uv run python scripts/generate_evidence.py            # 按 .env 配置
    LLM_PROVIDER=mock uv run python scripts/generate_evidence.py   # 离线模拟
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 把项目根目录放 sys.path 第 1 位，避免直接 `python scripts/xxx.py`
# 时被 site-packages 中的 Flask `app` 占包（同名包导致 import 错位）。
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# 不覆盖命令行已设置的环境变量，方便用 LLM_PROVIDER=mock 离线生成。
load_dotenv(PROJECT_ROOT / ".env")

from app.demo_questions import NORMAL_QUESTIONS, OVERBOUND_QUESTIONS
from app.llm import FallbackLLM, MockLLM, build_llm
from app.prompt_lab import compare_prompts
from app.prompts import STRONG_USER_TEMPLATE, render_template
from app.registry import ToolParam, ToolRegistry, ToolSpec, build_registry
from app.tool_loop import SYSTEM_PROMPT as TOOL_SYSTEM_PROMPT
from app.tool_loop import answer_question, answer_with_registry, answer_with_tools, format_trace
from app.tools import ToolError, execute_tool_call
from main import DISPATCH_DEMOS, MISSING_ARGUMENT_DEMOS, AgentSession, _unstable_handler

# 包一层降级：真实模型欠费/密钥失效/断网时不中断生成，
# 证据文件照常产出，并把降级原因如实写在文件开头。
llm = FallbackLLM(build_llm())
lines: list[str] = []


def emit(text: str = "") -> None:
    lines.append(text)


def block(title: str) -> None:
    emit(f"### {title}")
    emit()
    emit("```text")


def end_block() -> None:
    emit("```")
    emit()


emit("# 单元02—05 课堂验收证据")
emit()
emit(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
emit(f"- 运行模式：{type(llm.primary).__name__}（DeepSeekLLM=真实调用，MockLLM=离线模拟）")
emit("- 数据来源：data/records.csv 模拟订单；data/knowledge/ 样例政策原文（DOC-02-01 至 DOC-02-15）")
emit()

# ---------------------------------------------------------------- 单元02

emit("## 单元02 流式输出与多轮对话")
emit()
session = AgentSession(llm)
for index, question in enumerate(
    [
        "订单SO-2001已签收，我能退货吗？",
        "那物流有异常怎么办？",
        "退货还需要什么材料？",
    ],
    start=1,
):
    emit(f"### 第{index}轮")
    emit()
    emit(f"**顾客：** {question}")
    emit()
    emit("**助手（流式输出）：**")
    emit()
    emit("```text")
    emit(session.ask_stream(question))
    emit("```")
    emit()

block("/history 输出")
emit(session.chat.render())
end_block()

block("/clear 与 /exit")
session.clear()
emit(session.chat.render())
emit("对话结束。")
end_block()

# ---------------------------------------------------------------- 单元03

emit("## 单元03 Prompt 工程与模板")
emit()
emit("同一份订单数据、同一个问题、同一个模型，只有提示词不同。")
emit()

for index, question in enumerate(
    [
        NORMAL_QUESTIONS[0],
        "订单SO-2004是定制印字杯，已签收，我想七天无理由退货，可以吗？",
        "订单SO-9999能退货吗？",
    ],
    start=1,
):
    result = compare_prompts(llm, question)
    emit(f"### 问题{index}：{question}")
    emit()
    emit("**输入数据（两次调用完全一致）**")
    emit()
    emit("```text")
    emit(result["context"])
    emit("```")
    emit()
    emit("**差提示词回答**")
    emit()
    emit("```text")
    emit(result["weak"])
    emit("```")
    emit()
    emit("**好提示词回答**")
    emit()
    emit("```text")
    emit(result["strong"])
    emit("```")
    emit()

block("缺变量渲染报错")
try:
    render_template(STRONG_USER_TEMPLATE, question="订单SO-2001能退货吗？")
except ValueError as exc:
    emit(f"render_template(STRONG_USER_TEMPLATE, question=...) -> ValueError: {exc}")
end_block()

# ---------------------------------------------------------------- 单元04

emit("## 单元04 Function Calling")
emit()

answer, trace = answer_question(llm, "订单SO-2001已签收，我能退货吗？")
block("成功路径：订单查询 + 退货资格判断")
emit("用户问题：订单SO-2001已签收，我能退货吗？")
emit(format_trace(trace))
emit("")
emit(f"模型最终回答：{answer}")
end_block()

sloppy = MockLLM(omit_argument="reason")
_, repair_trace = answer_with_tools(
    sloppy, [{"role": "user", "content": "订单SO-2004定制杯有质量问题，能退货吗？"}]
)
block("缺参路径：模型漏传 reason -> 工具报错 -> 自动补全重试")
emit(format_trace(repair_trace))
end_block()

block("参数校验矩阵（工具名与参数可校验）")
for call in MISSING_ARGUMENT_DEMOS:
    emit(f"调用：{call['name']}  参数：{json.dumps(call['arguments'], ensure_ascii=False)}")
    try:
        emit(f"  校验通过 -> {json.dumps(execute_tool_call(call), ensure_ascii=False)}")
    except ToolError as exc:
        emit(f"  校验失败 -> {exc}")
end_block()

answer, trace = answer_question(llm, OVERBOUND_QUESTIONS[0])
block("越权请求：只出工单草稿")
emit(f"用户问题：{OVERBOUND_QUESTIONS[0]}")
emit(format_trace(trace))
emit("")
emit(f"模型最终回答：{answer}")
end_block()

emit("## 单元05 工具注册与调度")
emit()

registry = build_registry()
block(f"注册表（{len(registry.names())} 个受控工具，业务工具 ≥ 2）")
emit(f"工具清单：{', '.join(registry.names())}")
emit()
emit(registry.describe())
end_block()

block("调度器：一个批次的成功与拒绝（不访问模型）")
for step in registry.dispatch(DISPATCH_DEMOS, max_calls=4, round_index=1):
    emit(f"call_id={step['call_id']}  工具={step['name']}  状态={step['status']}（{step['elapsed_ms']}ms）")
    emit(f"  参数：{json.dumps(step['arguments'], ensure_ascii=False)}")
    emit(f"  结果：{json.dumps(step['result'], ensure_ascii=False)[:240]}")
end_block()

block("处理函数异常也不会让主程序崩溃（兜底分支）")
# 在真实注册表上加一个"会抛异常"的工具：既保留真实工具，又能演示兜底分支。
unstable = build_registry()
unstable.register(ToolSpec(
    name="unstable_demo_tool",
    description="课堂演示专用，不在给模型的清单里。",
    params=(ToolParam("order_id", "演示用订单号"),),
    handler=_unstable_handler,
    usage="演示处理函数抛出非 ToolError 异常时的兜底",
))
for step in unstable.dispatch(
    [
        {"call_id": "u-1", "name": "unstable_demo_tool", "arguments": {"order_id": "SO-2001"}},
        {"call_id": "u-2", "name": "query_order", "arguments": {"order_id": "SO-2002"}},
    ],
    round_index=2,
):
    emit(f"call_id={step['call_id']}  工具={step['name']}  状态={step['status']}")
    emit(f"  结果：{json.dumps(step['result'], ensure_ascii=False)[:240]}")
end_block()

block("两次调用日志（轮次 / 状态 / 耗时 / 异常提示）")
emit(registry.log.render())
emit()
emit(unstable.log.render())
end_block()

block("两个业务工具在同一会话里被分别调度")
dialog: list[dict] = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}]
conv_registry = build_registry()
order_question = "订单SO-2001已签收，普通键盘未拆封，我能退货吗？"
dialog.append({"role": "user", "content": order_question})
ans, tr = answer_with_registry(llm, dialog, conv_registry, require_tool_call=False)
emit(f"**顾客：** {order_question}")
emit(format_trace(tr))
emit(f"**最终回答：** {ans}")
emit()
policy_question = "请给我看看 DOC-02-05 的政策原文怎么写。"
dialog.append({"role": "user", "content": policy_question})
ans, tr = answer_with_registry(llm, dialog, conv_registry, require_tool_call=False)
emit(f"**顾客：** {policy_question}")
emit(format_trace(tr))
emit(f"**最终回答：** {ans}")
end_block()

block("本次问询的注册表日志（同一会话两次工具调用都在册）")
emit(conv_registry.log.render())
end_block()

emit("## 附：证据与五人分工对应")
emit()
emit("| 单元 | 交付物 | 本文件位置 |")
emit("|---|---|---|")
emit("| 单元02 | 对话入口、会话类、流式接口 | 单元02 第1—3轮问答与 /history、/clear |")
emit("| 单元03 | 差/好两套模板、变量渲染、真实调用入口 | 单元03 三个问题的同题对比与缺变量报错 |")
emit("| 单元04 | 工具 Schema、handler、调用入口 | 单元04 成功路径、缺参路径、校验矩阵、越权草稿 |")
emit("| 单元05 | ToolRegistry、调度器、第二个业务工具 | 单元05 注册表 / 调度 / 异常兜底 / 同会话两工具 / 调用日志 |")

out_path = PROJECT_ROOT / "docs" / "课堂验收证据.md"
out_path.parent.mkdir(parents=True, exist_ok=True)

# 降级信息只有全部跑完才知道，所以在末尾补一段说明，并同步修正开头的"运行模式"。
if llm.degradations:
    lines[3] = (
        f"- 运行模式：{type(llm.primary).__name__} 调用失败，"
        f"已降级为 {type(llm.fallback).__name__}（离线模拟）"
    )
    lines.insert(4, f"- **降级告警：真实模型不可用，本次已自动切到离线模拟，共 {len(llm.degradations)} 次**")
    lines.extend(
        [
            "",
            "## 附：降级记录",
            "",
            "本次运行中真实模型调用失败，以下片段是离线模拟结果，不是真实模型输出：",
            "",
        ]
    )
    for item in llm.degradations:
        lines.append(f"- {item}")
else:
    lines.insert(4, "- 本次全部为真实模型调用，未触发降级")

out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"已生成：{out_path}（{len(lines)} 行）")
if llm.degradations:
    print("注意：真实模型不可用，已降级为离线模拟：")
    for item in llm.degradations:
        print(f"  - {item}")
