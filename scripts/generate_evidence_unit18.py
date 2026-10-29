"""生成单元18 的课堂验收证据。

按任务卡的四节课顺序**真实执行**并抓取输出，数字与报错文案全部取自实际运行：

    uv run --extra deepseek --extra embedding --extra chroma --extra parse --extra langchain \
        python scripts/generate_evidence_unit18.py

覆盖要交的东西：
    第1节  概念记录与项目问题清单；资料与工具还是单元12/15 那一批
    第2节  流程图与分支规则表；最小业务场景（知识分支）跑通
    第3节  顺序步骤：每步的输入字段、输出字段、耗时与说明
    第4节  逐条验证三条分支 + 兜底 + 无证据 + 无效输入

**密钥安全**：脚本不打印密钥，输出统一做 sk- 掩码兜底。
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元18验收证据.md"

BOOTSTRAP = """
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))
"""

# 第1、2节：资料与工具清单 + 流程图 + 分支规则 + 概念记录。
PREPARE_FLOW = BOOTSTRAP + r"""
import os
from app.chroma_store import PROJECT_ID
from app.flow_chain import render_concepts, render_flow, render_routes
from app.flow_lab import build_stack
from app.loaders import load_materials
from app.parse_lab import EXPECTED_RECORDS

from app.flow_lab import KNOWLEDGE_DIR

records, failures = load_materials(KNOWLEDGE_DIR, PROJECT_ID)
print(f"资料目录：{KNOWLEDGE_DIR}")
print(f"解析结果：{len(records)} 条（单元12 是 {EXPECTED_RECORDS} 条）  解析失败：{failures or '无'}")
for record in records:
    print(f"  {record.file_type}\t{record.location_type}={record.location}\t"
          f"{record.source.rsplit('/', 1)[-1]}")

parts, registry, _service, kb_count, faq_rows = build_stack()
steps = getattr(parts.summary_chain, "steps", [])
print()
print(f"向量库内本项目 {kb_count} 条；FAQ 库 {faq_rows} 行；工具 {len(registry.names())} 项：{registry.names()}")
print(f"模型：{type(steps[1]).__name__ if len(steps) > 1 else '?'}"
      f"（LLM_PROVIDER={os.getenv('LLM_PROVIDER')}）  密钥已配置：{bool(os.getenv('DEEPSEEK_API_KEY'))}")
print()
print(render_flow())
print()
print(render_routes())
print()
print(render_concepts())
"""

# 第2、4节：路由规则对一批问题逐条分类（含边界：只有日期没问时限、闲聊）。
ROUTE_FLOW = BOOTSTRAP + r"""
from app.flow_chain import route_question

questions = [
    "EC-FAQ-01 的标准答复是什么？",
    "EC-FAQ-02 商品有质量问题要提供什么？",
    "普通商品退货有哪些要求？",
    "物流轨迹长期没有更新怎么办？",
    "2026年12月1日签收的普通商品，退货评估最晚到哪天？",
    "2026年12月1日你们几点上班？",
    "帮我写一首关于秋天的诗",
    "今天天气怎么样",
]
print("问题\t分支\t命中\t判定理由")
for question in questions:
    decision = route_question(question)
    print(f"{question}\t{decision.route}\t{decision.matched or '—'}\t{decision.reason}")
"""

# 第2、3节：三条分支各跑一次（真实资料 + 当前配置的模型）。
BRANCH_FLOW = BOOTSTRAP + r"""
from app.flow_lab import (
    DEADLINE_QUESTION, FAQ_QUESTION, KNOWLEDGE_QUESTION,
    build_stack, render_scenario, run_scenario,
)

parts, _registry, _service, _kb, _rows = build_stack()
for key, question in (
    ("knowledge", KNOWLEDGE_QUESTION),
    ("faq", FAQ_QUESTION),
    ("deadline", DEADLINE_QUESTION),
):
    print(render_scenario(run_scenario(key, question, parts)))
    print()
"""

# 第4节：兜底、无证据、无效输入。
EDGE_FLOW = BOOTSTRAP + r"""
from app.flow_lab import (
    NO_EVIDENCE_QUESTION, UNKNOWN_QUESTION,
    build_stack, render_scenario, run_invalid, run_scenario,
)

parts, _registry, _service, _kb, _rows = build_stack()
print(render_scenario(run_scenario("unknown", UNKNOWN_QUESTION, parts)))
print()
print(render_scenario(run_scenario("no_evidence", NO_EVIDENCE_QUESTION, parts)))
print()
print("无效输入（守门在跑工作流之前）：")
print("输入\t结果")
for label, verdict in run_invalid(parts):
    print(f"{label}\t{verdict}")
"""

# 第3节：同一问题的步骤表 + 每步的字段清单（"每步输入输出可查看"的直接证据）。
STEPS_FLOW = BOOTSTRAP + r"""
from app.flow_chain import render_steps
from app.flow_lab import KNOWLEDGE_QUESTION, build_stack, run_scenario

parts, _registry, _service, _kb, _rows = build_stack()
result = run_scenario("knowledge", KNOWLEDGE_QUESTION, parts).result
print(render_steps(result))
print()
print("每步吃进哪些字段、吐出哪些字段（同一次运行）：")
for step in result.steps:
    print(f"第{step.index}步 {step.name}")
    print(f"  输入：{'、'.join(step.input_keys)}")
    print(f"  输出：{'、'.join(step.output_keys)}")
"""

# 第4节：五条路径的横向对照（哪条分支到了、哪条一次模型都没调）。
SUMMARY_FLOW = BOOTSTRAP + r"""
from app.flow_lab import build_stack, render_checklist, render_summary_table, run_class

print(render_summary_table(run_class()))
print()
print(render_checklist(run_class()))
"""

# 换模型不换编排：路由、分支、兜底、步骤表都不依赖模型。
MOCK_FLOW = BOOTSTRAP + r"""
import os
os.environ["LLM_PROVIDER"] = "mock"
from app.flow_chain import ScriptedFlowModel
from app.flow_lab import UNKNOWN_QUESTION, build_stack, render_result, run_scenario

parts, _registry, _service, _kb, _rows = build_stack(model=ScriptedFlowModel())
print("模型：离线编排模型（只摘录证据，不生成新事实）")
normal = run_scenario("knowledge", "普通商品退货有哪些要求？", parts).result
print(f"知识分支：模型调用 {normal.model_calls} 次；停止原因：{normal.stop_reason}")
fallback = run_scenario("unknown", UNKNOWN_QUESTION, parts).result
print(f"兜底分支：模型调用 {fallback.model_calls} 次；停止原因：{fallback.stop_reason}")
print(f"兜底答复：{fallback.answer}")
"""

BOUNDARIES = [
    (
        "问题为空：守门在跑工作流之前，既不检索也不调模型",
        BOOTSTRAP + r"""
from app.lc_chain import validate_question
try:
    validate_question("   ")
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
        "库里没有这个 FAQ 编号：查不到就是查不到，不返回像答案的东西",
        BOOTSTRAP + r"""
from app.flow_chain import render_result
from app.flow_lab import build_stack, run_scenario

parts, _registry, _service, _kb, _rows = build_stack()
result = run_scenario("faq", "EC-FAQ-09 的标准答复是什么？", parts).result
print(render_result(result))
""",
    ),
    (
        "日期不合法：算不出日子就不算，也不让模型口算",
        BOOTSTRAP + r"""
from app.flow_chain import render_result
from app.flow_lab import build_stack, run_scenario

parts, _registry, _service, _kb, _rows = build_stack()
result = run_scenario("deadline", "2026年13月45日签收的普通商品，退货评估最晚到哪天？", parts).result
print(render_result(result))
""",
    ),
    (
        "provider 写错：不猜，直接说只支持两种",
        BOOTSTRAP + r"""
from app.flow_chain import build_flow_model
try:
    build_flow_model(provider="openai")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "未显式指定 mock 时的模型身份：真实模式缺密钥当场报错，不静默降级",
        BOOTSTRAP + r"""
import os
from app.flow_chain import build_flow_model

saved = os.environ.pop("DEEPSEEK_API_KEY", None)
try:
    build_flow_model(provider="deepseek")
except RuntimeError as error:
    print(f"{type(error).__name__}: {error}")
finally:
    if saved is not None:
        os.environ["DEEPSEEK_API_KEY"] = saved
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
    route_text = run_code(ROUTE_FLOW)
    branch_text = run_code(BRANCH_FLOW)
    edge_text = run_code(EDGE_FLOW)
    steps_text = run_code(STEPS_FLOW)
    summary_text = run_code(SUMMARY_FLOW)
    mock_text = run_code(MOCK_FLOW)

    lines: list[str] = []
    lines.append("# 单元18 验收证据：Chain 链式编排（顺序步骤 + 条件路由 + 安全兜底）")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")
    lines.append("- 场景：本组电商售后问答；资料与三项只读工具**全部沿用单元12/15**，本单元不另造")
    lines.append("- 本单元只换了一件事：**谁来编排**——谁先做、走哪条分支由程序写死，"
                 "模型只负责摘要与润色这两步")
    lines.append("- 承诺：`app/toolkit.py`（单元15 工具）、`app/lc_agent.py`（单元17 执行器）、"
                 "`app/lc_chain.py`（单元16 链）**一行没删**，新工作流不覆盖旧实现")
    lines.append("")
    lines.append("## 一、准备：资料、工具、概念与问题清单（第1节）")
    lines.append("")
    lines.append("```text")
    lines.append(prepare_text)
    lines.append("```")
    lines.append("")
    lines.append("要看的有三处：资料仍是 7 条（与单元12 一致）、工具仍是从单元15 注册表里取的三项、")
    lines.append("流程图里 `unknown` 这一支直接指向兜底——兜底不是异常处理，是**一条正常分支**。")
    lines.append("")
    lines.append("## 二、路由规则逐条验（第2节）")
    lines.append("")
    lines.append("```text")
    lines.append(route_text)
    lines.append("```")
    lines.append("")
    lines.append("这批问题里有两处要特别看：`EC-FAQ-02` 也带编号，所以走编号分支而不是关键词分支；")
    lines.append("「2026年12月1日你们几点上班」有日期却没问时限，**不**进计算分支——这说明日期分支的条件")
    lines.append("是「日期 + 时限类词」两件事同时成立，不是见到日期就算日子。")
    lines.append("")
    lines.append("## 三、三条分支各跑一次（第2、3节）")
    lines.append("")
    lines.append("```text")
    lines.append(branch_text)
    lines.append("```")
    lines.append("")
    lines.append("三条分支的取证据方式完全不同（按编号查 SQLite / 向量检索 / 检索加日期加法），")
    lines.append("但出口字段是**同一组**（evidence、sources、has_evidence），所以后面的摘要与润色")
    lines.append("两条分支共用同一套提示词——这就是「统一入口」在编排层的含义。")
    lines.append("")
    lines.append("## 四、每步输入输出可查看（第3节）")
    lines.append("")
    lines.append("```text")
    lines.append(steps_text)
    lines.append("```")
    lines.append("")
    lines.append("步骤表里字段数量逐级上涨（只加不改），所以 `question` 一路都在；")
    lines.append("摘要单独成一步，是为了出事时能分清「摘要漏了数字」还是「润色加了话」。")
    lines.append("")
    lines.append("## 五、兜底、无证据与无效输入（第4节）")
    lines.append("")
    lines.append("```text")
    lines.append(edge_text)
    lines.append("```")
    lines.append("")
    lines.append("两处要看数字：兜底与无证据这两条路径的**模型调用都是 0 次**——不是「调了模型让它别乱说」，")
    lines.append("而是压根没进生成那一步；无效输入那几行的「步骤表新增 0 条」说明守门建图之前就拦下了。")
    lines.append("")
    lines.append("## 六、五条路径横向对照（第4节）")
    lines.append("")
    lines.append("```text")
    lines.append(summary_text)
    lines.append("```")
    lines.append("")
    lines.append("任务卡的三条完成标准各自对应一行：分支可达看「走到的分支」、每步可查看看步骤表、")
    lines.append("兜底看那条 `unknown` 的 0 次模型调用。")
    lines.append("")
    lines.append("## 七、换模型不换编排")
    lines.append("")
    lines.append("```text")
    lines.append(mock_text)
    lines.append("```")
    lines.append("")
    lines.append("同一套工作流、同一批资料，只把模型换成离线编排模型：路由结果、分支判定、兜底的")
    lines.append("0 次模型调用、步骤表结构全部一致——说明「谁编排」和「谁生成」是分开的两件事。")
    lines.append("")
    lines.append("## 八、边界核对")
    lines.append("")
    lines.append("下面每一条都由 Python 在跑工作流**之前**或**取证据过程中**拦下，")
    lines.append("报错文案直接说明被拒绝的原因。")
    for title, code in BOUNDARIES:
        lines.append("")
        lines.append(f"### {title}")
        lines.append("")
        lines.append("```text")
        lines.append(run_code(code))
        lines.append("```")
    lines.append("")
    lines.append("## 九、密钥与产物说明")
    lines.append("")
    lines.append("- 密钥写在项目根 `.env`，`.gitignore` 已排除，不进仓库，也不出现在任何输出里。")
    lines.append("- 本证据文档中的所有输出均经过 `sk-` 掩码处理。")
    lines.append("- 运行产物沿用 `.chroma_unit15/`（向量库）与 `.sqlite_unit15/`（FAQ 库），")
    lines.append("  与单元15/17 是同一份，本单元不新建库——换机器重跑 `/flow` 即可重建。")
    lines.append("- 想不调用 API 复现同一套流程，先 `LLM_PROVIDER=mock` 再跑：")
    lines.append("  摘要与润色换成离线模型（只摘录证据），路由、分支、兜底与步骤表完全一致。")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
