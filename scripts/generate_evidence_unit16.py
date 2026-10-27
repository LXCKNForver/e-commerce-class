"""生成单元16 的课堂验收证据。

按任务卡的四节课顺序**真实执行**并抓取输出，数字与报错文案全部取自实际运行：

    uv run --extra embedding --extra chroma --extra langchain python scripts/generate_evidence_unit16.py

覆盖要交的东西：
    第1节  在现有项目里标出可复用的检索函数与真实资料来源
    第2节  RunnableLambda 接入检索，assign 保留问题、片段和来源
    第3节  Prompt + Model + Parser 组合成链，跑一次真实问答
    第4节  同一问题对照单元15 的 ToolAgent；资料不足与无效输入的处理

**密钥安全**：脚本不打印密钥，输出统一做 sk- 掩码兜底。
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元16验收证据.md"

BOOTSTRAP = """
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))
"""

# 第1节：资料清单 + 可复用件 + 库与模型配置。
PREPARE_FLOW = BOOTSTRAP + r"""
import os
from app.chroma_store import PROJECT_ID, UNIT16_COLLECTION, resolve_unit16_db_path
from app.loaders import load_materials
from app.chain_lab import KNOWLEDGE_DIR, build_stack, render_reusable
from app.parse_lab import EXPECTED_RECORDS

records, failures = load_materials(KNOWLEDGE_DIR, PROJECT_ID)
print(f"资料目录：{KNOWLEDGE_DIR}")
print(f"解析结果：{len(records)} 条（完成标准要求 {EXPECTED_RECORDS} 条）  解析失败：{failures or '无'}")
for record in records:
    print(f"  {record.file_type}\t{record.location_type}={record.location}\t"
          f"{record.source.rsplit('/', 1)[-1]}")

parts, service, store = build_stack()
print()
print(f"独立库目录：{resolve_unit16_db_path()}")
print(f"向量集合：{store.collection.name}（本项目 {store.count(PROJECT_ID)} 条）")
print(f"项目标识：{PROJECT_ID}")
print(f"模型：{type(parts.model).__name__}（LLM_PROVIDER={os.getenv('LLM_PROVIDER')}）"
      f"  密钥已配置：{bool(os.getenv('DEEPSEEK_API_KEY'))}")
print(f"链的输入：{parts.chain.input_schema.model_json_schema().get('properties', {})}")
values = parts.chain.invoke({"question": "普通商品退货有哪些要求？"})
print(f"一次 invoke 拿回来的字段：{sorted(values)}（assign 一路加字段，不是替换）")
print(f"证据 {len(values['context'])} 字；采用来源 {len(values['sources'])} 个；答案 {len(values['answer'])} 字")
print()
print(render_reusable())
"""

# 第2、3节：六段流转表 + 一次真实运行（逐段 + 组合）。
STAGE_FLOW = BOOTSTRAP + r"""
from app.chain_lab import DEFAULT_QUESTION, build_stack, render_scenario, run_scenario
from app.lc_chain import render_stage_table

parts, _service, _store = build_stack()
print(render_stage_table())
print()
print(render_scenario(run_scenario("steps", DEFAULT_QUESTION, parts, inspect_stages=True)))
"""

# 第3节：一次组合链运行（一次 invoke 拿到全部字段）。
ANSWER_FLOW = BOOTSTRAP + r"""
from app.chain_lab import DEFAULT_QUESTION, build_stack, render_scenario, run_scenario

parts, _service, _store = build_stack()
print(render_scenario(run_scenario("answer", DEFAULT_QUESTION, parts)))
"""

# 第4节：同题对照 + 资料不足 + 无效输入。
COMPARE_FLOW = BOOTSTRAP + r"""
from app.chain_lab import (
    DEFAULT_QUESTION,
    INSUFFICIENT_QUESTION,
    build_stack,
    compare_with_agent,
    render_comparison,
    render_invalid,
    render_scenario,
    run_invalid,
    run_scenario,
)

parts, service, _store = build_stack()
print(render_comparison(compare_with_agent(DEFAULT_QUESTION, parts, service)))
print()
print(render_scenario(run_scenario("empty", INSUFFICIENT_QUESTION, parts)))
print()
print(render_invalid(run_invalid(parts)))
"""

# 换模型只换第⑤段：同一条链、同一份资料，换成离线摘抄模型再跑一次。
SWAP_FLOW = BOOTSTRAP + r"""
from app.chain_lab import DEFAULT_QUESTION, build_stack, render_scenario, run_scenario
from app.lc_chain import OfflineChatModel

parts, _service, _store = build_stack(model=OfflineChatModel())
scenario = run_scenario("answer", DEFAULT_QUESTION, parts)
print(f"模型：{scenario.result.model_name}")
print(f"采用来源：{scenario.result.sources}（与真实模型那一次是同一批命中）")
print(f"回答：{scenario.result.answer[:200]}")
"""

BOUNDARIES = [
    (
        "问题为空：守门在链之前，检索与模型都不会被触发",
        BOOTSTRAP + r"""
from app.lc_chain import validate_question
try:
    validate_question("   ")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "问题过长：撑爆证据窗口的输入同样被拒",
        BOOTSTRAP + r"""
from app.lc_chain import MAX_QUESTION_CHARS, validate_question
try:
    validate_question("退" * (MAX_QUESTION_CHARS + 1))
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "问题不是字符串：类型也要先看一眼",
        BOOTSTRAP + r"""
from app.lc_chain import validate_question
try:
    validate_question({"question": "退货"})
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "证据门槛：检索有命中，但相似度不达标就不算证据",
        BOOTSTRAP + r"""
from app.lc_chain import EMPTY_CONTEXT, MIN_SIMILARITY, filter_hits, format_context

hits = [
    {"source": "模拟数据/单元12项目资料/电商售后服务智能体项目组/常见售后问题.csv",
     "text": "课程虚构资料，仅供教学演示。", "distance": 0.99, "similarity": 0.504},
    {"source": "模拟数据/单元12项目资料/电商售后服务智能体项目组/DOC-02-01-普通商品退货规则.md",
     "text": "签收后7天内且未使用可申请退货评估。", "distance": 0.61, "similarity": 0.694},
]
print(f"门槛：{MIN_SIMILARITY}   过滤后剩下：{len(filter_hits(hits))} 条")
weak = [hit for hit in hits if hit["similarity"] < MIN_SIMILARITY]
print(f"只留下不达标的那条时，上下文是：{format_context(weak)!r}")
print(f"与占位文本相同：{format_context(weak) == EMPTY_CONTEXT}")
""",
    ),
    (
        "证据边界写在 Prompt 里：把渲染出来的 system 消息原文打出来",
        BOOTSTRAP + r"""
from app.chain_lab import DEFAULT_QUESTION, build_stack
from app.lc_chain import EMPTY_CONTEXT

parts, _service, _store = build_stack()
values = parts.with_context.invoke({"question": DEFAULT_QUESTION})
messages = parts.prompt.invoke(values).to_messages()
print(f"system 消息：{messages[0].content}")
print(f"human 消息前 60 字：{messages[1].content[:60]}…")
print(f"把证据换成占位文本后，human 消息结尾是："
      f"{parts.prompt.invoke({'question': DEFAULT_QUESTION, 'context': EMPTY_CONTEXT}).to_messages()[1].content[-30:]!r}")
""",
    ),
    (
        "两种跑法字段一致：组合链一次 invoke 与逐段跑是同一批证据",
        BOOTSTRAP + r"""
from app.chain_lab import DEFAULT_QUESTION, build_stack
from app.lc_chain import run_chain

parts, _service, _store = build_stack()
one = run_chain(DEFAULT_QUESTION, parts)
two = run_chain(DEFAULT_QUESTION, parts, inspect_stages=True)
print(f"来源一致：{one.sources == two.sources}")
print(f"上下文一致：{one.context == two.context}")
print(f"逐段记录：{len(two.stages)} 段，模型输出类型：{two.message_kind}")
""",
    ),
    (
        "链的最后一格只做类型转换：StrOutputParser 不判断对错",
        BOOTSTRAP + r"""
from app.chain_lab import build_stack
from langchain_core.messages import AIMessage

parts, _service, _store = build_stack()
print(repr(parts.parser.invoke(AIMessage(content="签收后7天内"))))
print(f"解析器类型：{type(parts.parser).__name__}")
""",
    ),
    (
        "没配密钥时：真实模式当场报错，不静默降级",
        BOOTSTRAP + r"""
import os
from app.lc_chain import build_chat_model

saved = os.environ.pop("DEEPSEEK_API_KEY", None)
try:
    build_chat_model(provider="deepseek")
except RuntimeError as error:
    print(f"{type(error).__name__}: {error}")
finally:
    if saved is not None:
        os.environ["DEEPSEEK_API_KEY"] = saved
""",
    ),
    (
        "provider 写错：不猜，直接说只支持两种",
        BOOTSTRAP + r"""
from app.lc_chain import build_chat_model
try:
    build_chat_model(provider="openai")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
]


def run_code(code: str) -> str:
    """在独立进程里跑一段代码，抓回 stdout；失败直接抛出，不静默吞掉。"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return mask_secrets(result.stdout.strip())


def mask_secrets(text: str) -> str:
    """兜底掩码：任何 sk- 开头的密钥都不允许出现在证据文档里。"""
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***（已掩码）", text)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.path.insert(0, str(PROJECT_ROOT))

    prepare_text = run_code(PREPARE_FLOW)
    stage_text = run_code(STAGE_FLOW)
    answer_text = run_code(ANSWER_FLOW)
    compare_text = run_code(COMPARE_FLOW)
    swap_text = run_code(SWAP_FLOW)

    lines: list[str] = []
    lines.append("# 单元16 验收证据：LangChain 核心组合链")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")
    lines.append("- 场景：本组电商售后问答；链只串起已有能力，资料与检索函数一个字没改")
    lines.append("- 承诺：原有 ReAct Agent（单元14/15）**一行没删**，新链不覆盖旧实现")
    lines.append("")
    lines.append("## 一、准备：可复用件与真实资料来源（第1节）")
    lines.append("")
    lines.append("```text")
    lines.append(prepare_text)
    lines.append("```")
    lines.append("")
    lines.append("倒数第二行是链的输入 schema：只有一个 `question`；")
    lines.append("再往下是一次 `invoke` 真实拿回来的字段——`question / hits / context / "
                 "sources / answer` 五个都在，因为全程用的是 `assign`（加字段），不是管道式的替换。")
    lines.append("")
    lines.append("## 二、六段流转与逐段运行记录（第1、2节）")
    lines.append("")
    lines.append("```text")
    lines.append(stage_text)
    lines.append("```")
    lines.append("")
    lines.append("上面那张表是链的静态结构，下面这段是同一次运行**逐段 invoke** 记下来的")
    lines.append("真实输入输出：问题在第②段之后一直都在，所以最后还能把来源一起打印出来。")
    lines.append("")
    lines.append("## 三、组合链一次运行：真实模型回答（第3节）")
    lines.append("")
    lines.append("```text")
    lines.append(answer_text)
    lines.append("```")
    lines.append("")
    lines.append("一次 `invoke` 就把五个字段全带回来了：命中片段、证据文本、采用来源、答案。")
    lines.append("")
    lines.append("## 四、同题对照旧 Agent、资料不足与无效输入（第4节）")
    lines.append("")
    lines.append("```text")
    lines.append(compare_text)
    lines.append("```")
    lines.append("")
    lines.append("对照**不比措辞**：比的是两边引用的文件与出现的数字/编号。")
    lines.append("资料不足那一节要看的是：检索照样返回 3 条，但都没过相似度门槛，")
    lines.append("于是递给模型的是「未检索到相关资料」，答案只能是资料不足。")
    lines.append("")
    lines.append("## 五、换模型只换第⑤段")
    lines.append("")
    lines.append("```text")
    lines.append(swap_text)
    lines.append("```")
    lines.append("")
    lines.append("同一条链、同一批命中，只把第⑤段换成离线摘抄模型：其余五段的代码一行没动，")
    lines.append("来源还是那几个文件——说明资料口径与链的形状是分开的两件事。")
    lines.append("")
    lines.append("## 六、边界核对")
    lines.append("")
    lines.append("下面每一条都由 Python 在跑链**之前**或**拿到结果之后立即**拦下，")
    lines.append("报错文案直接说明被拒绝的原因。")
    for title, code in BOUNDARIES:
        lines.append("")
        lines.append(f"### {title}")
        lines.append("")
        lines.append("```text")
        lines.append(run_code(code))
        lines.append("```")
    lines.append("")
    lines.append("## 七、密钥与产物说明")
    lines.append("")
    lines.append("- 密钥写在项目根 `.env`，`.gitignore` 已排除，不进仓库，也不出现在任何输出里。")
    lines.append("- 本证据文档中的所有输出均经过 `sk-` 掩码处理。")
    lines.append("- 运行产物 `.chroma_unit16/`（向量库）是本地文件，不进仓库；换机器重跑 `/chain` 即可重建。")
    lines.append("- 想不调用 API 复现同一套流程，先 `LLM_PROVIDER=mock` 再跑：")
    lines.append("  第⑤段会换成离线摘抄模型，其余五段完全一致。")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
