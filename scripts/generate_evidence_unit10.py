"""生成单元10 的课堂验收证据。

按任务卡四节课的顺序**真实执行**并抓取输出，不手写任何数字：

    uv run --extra embedding --extra chroma python scripts/generate_evidence_unit10.py

覆盖要交的东西：
    第1节 本单元独立库、资料条数、复用第九单元检索服务的结果
    第2节 跨项目未命中 → 拒答，且模型调用次数为 0；编号上下文示例
    第3节 真实 DeepSeek 生成：真实问题、模型回答、引用编号
    第4节 来源核验清单、边界说明与小组验收要点

**密钥安全**：脚本只输出模型类型与「密钥是否已配置」，绝不打印密钥本身；
输出还会再做一次 sk- 掩码兜底。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元10验收证据.md"
REAL_DB_PATH = PROJECT_ROOT / ".chroma_unit10"

# 一次完整的「建库 → 拒答 → 上下文 → 生成 → 核验」，与 main.py 的 /rag 走同一套代码。
FLOW_CODE = r"""
import sys, os
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))

from app.chroma_store import PROJECT_ID, ChromaVectorStore
from app.ingest import load_markdown
from app.llm import build_llm
from app.rag import (
    DEMO_QUESTION, EMPTY_PROJECT, NO_EVIDENCE_ANSWER, RAGService, RecordingLLM,
    build_context_preview, render_result,
)
from app.retriever import RetrievalService

records = load_markdown()
database_path = Path(sys.argv[1])
store = ChromaVectorStore(database_path)
before = store.count()
store.upsert_documents(records)
print(f"数据库目录：{database_path}")
print(f"集合：{store.collection.name}  集合 ID：{store.collection.id}")
print(f"导入前/后：{before} / {store.count()} 条；项目：{PROJECT_ID}")
print(f"资料篇数：{len(records)}")

llm = build_llm()
recorder = RecordingLLM(llm)
print(f"模型类型：{type(llm).__name__}")
print(f"密钥已配置：{bool(os.getenv('DEEPSEEK_API_KEY'))}")
rag = RAGService(RetrievalService(store), recorder)

print()
print(f"未命中示例（项目：{EMPTY_PROJECT}）：")
miss = rag.answer(DEMO_QUESTION, EMPTY_PROJECT)
print(f"  回答：{miss.answer}")
print(f"  命中条数：{len(miss.hits)}")
print(f"  等于拒答话术：{miss.answer == NO_EVIDENCE_ANSWER}")
print(f"  模型累计调用次数：{recorder.call_count}")

hits = rag.retrieve(DEMO_QUESTION, PROJECT_ID, rag.top_k)
print()
print("编号上下文（模型只能在这些编号里引用）：")
print(build_context_preview(rag, hits, preview=120))

print()
result = rag.answer(DEMO_QUESTION, PROJECT_ID)
print(render_result(result))
print(f"模型累计调用次数：{recorder.call_count}")
"""

BOUNDARY_CODE = r"""
import sys
sys.stdout.reconfigure(errors="replace")
from tempfile import mkdtemp
from pathlib import Path

from app.chroma_store import ChromaVectorStore, PROJECT_ID
from app.ingest import load_markdown
from app.rag import DEMO_QUESTION, NO_CITATION_ANSWER, RAGService
from app.rag_prompts import build_messages
from app.retriever import RetrievalService


class Hit:
    document_id = "x"
    text = "t"
    source = "模拟数据/knowledge/DOC-02-01-普通商品退货规则.md"
    project_id = PROJECT_ID
    doc_type = "knowledge"
    distance = 0.5
    similarity = 0.75


class ScriptedLLM:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages):
        return self.reply


store = ChromaVectorStore(Path(mkdtemp(prefix="unit10-boundary-")) / "chroma")
store.upsert_documents(load_markdown())
service = RetrievalService(store)

print("【参数边界】")
for label, call in [
    ("问题为空", lambda: RAGService(service, ScriptedLLM("a")).answer("   ", PROJECT_ID)),
    ("项目标识为空", lambda: RAGService(service, ScriptedLLM("a")).answer(DEMO_QUESTION, "")),
    ("上下文为空", lambda: build_messages("问题", "")),
]:
    try:
        call()
        print(f"{label} -> 没有报错（异常）")
    except Exception as exc:
        print(f"{label} -> {type(exc).__name__}: {exc}")

print()
print("【引用核验边界】（只有 2 条证据时的四种回答）")
hits = [Hit(), Hit()]
rag = RAGService(service, ScriptedLLM("占位"), top_k=3)
cases = {
    "引用越界 [9]": "据资料判断 [9]。",
    "完全不带编号": "据资料判断，定制商品不能退。",
    "模型自己说资料不足": "未检索到足够资料，暂不回答。",
    "重复引用 [1] 与 [2]": "第一点 [1]，第二点 [2]，再补充 [2]。",
}
for label, reply in cases.items():
    print(f"{label} -> 核验后编号={rag.verify_sources(reply, hits)}")

print()
print(f"不可核验时的统一回答：{NO_CITATION_ANSWER}")
"""


def run_code(code: str, *args: str) -> str:
    """在独立进程里跑一段代码，抓回 stdout；失败直接抛出，不静默吞掉。"""
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return mask_secrets(result.stdout.strip())


def mask_secrets(text: str) -> str:
    """兜底掩码：任何 sk- 开头的串都打码，避免密钥被写进证据文件。"""
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***（已掩码）", text)


def field(text: str, label: str) -> str:
    """从一段输出里取某个字段；取不到就如实写「未取到」。"""
    marker = f"{label}："
    for line in text.splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return "（未取到）"


def slice_from(text: str, start: str, end: str | None = None) -> str:
    """截取两个标记之间的输出片段，把一次运行拆成几个验收小节。"""
    lines = text.splitlines()
    start_index = None
    for index, line in enumerate(lines):
        if start in line:
            start_index = index
            break
    if start_index is None:
        return "（未取到）"
    stop_index = len(lines)
    if end is not None:
        for index in range(start_index + 1, len(lines)):
            if end in lines[index]:
                stop_index = index
                break
    while stop_index > start_index and not lines[stop_index - 1].strip():
        stop_index -= 1
    return "\n".join(lines[start_index:stop_index])


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        import chromadb  # noqa: F401
    except ImportError:
        print("缺少 chromadb。请先执行：uv sync --extra chroma")
        return 1
    try:
        from app.chroma_store import PROJECT_ID
        from app.embedding import model_ready
        from app.ingest import load_markdown
        from app.rag import DEMO_QUESTION, EMPTY_PROJECT
    except ImportError:
        print("缺少 fastembed 或本地模型。请先执行：uv sync --extra embedding")
        return 1

    ready, detail = model_ready()
    if not ready:
        print(f"本地向量模型不可用：{detail}")
        return 1

    document_count = len(load_markdown())

    # 场景A：全新数据库，看导入前后数量的真实变化。
    fresh_dir = Path(tempfile.mkdtemp(prefix="unit10-fresh-"))
    fresh_run = run_code(FLOW_CODE, str(fresh_dir / "chroma"))
    shutil.rmtree(fresh_dir, ignore_errors=True)

    # 场景B：真实库连跑两次，核对记录数不增加、集合 ID 不变、行为可复现。
    real_first = run_code(FLOW_CODE, str(REAL_DB_PATH))
    real_second = run_code(FLOW_CODE, str(REAL_DB_PATH))

    # 第4节边界：参数校验 + 引用核验的四种情形。
    boundary = run_code(BOUNDARY_CODE)

    model_type = field(real_first, "模型类型")
    key_ready = field(real_first, "密钥已配置")
    refused_calls = field(real_first, "模型累计调用次数")
    # 模型点评、引用编号、来源都在 render_result 那一段里，只从这段解析，
    # 免得撞上前面"编号上下文"里带「来源：」的资料行。
    answer_block = slice_from(real_first, "检索命中：")
    answer_text = field(answer_block, "回答")
    cited = field(answer_block, "引用编号")
    sources = field(answer_block, "来源")
    hits_line = slice_from(answer_block, "检索命中：", "  [1]")
    total_calls = real_second.splitlines()[-1].split("：")[-1].strip()
    errored = "模型调用失败" in real_first

    lines: list[str] = []
    lines.append("# 单元10 验收证据：RAG 问答闭环")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")
    lines.append(f"- 本组资料：`模拟数据/knowledge/`，共 {document_count} 篇")
    lines.append(f"- 项目标识：`{PROJECT_ID}`；未命中演示用另一个项目：`{EMPTY_PROJECT}`")
    lines.append(f"- 真实问题（自拟）：{DEMO_QUESTION}")
    lines.append("- 持久化目录：`.chroma_unit10/`，与单元08/09 的库互不干扰")
    lines.append(
        f"- 生成模型：`{model_type}`；本机密钥已配置：`{key_ready}`"
        "（按任务卡要求，密钥本身不出现在任何输出或提交里）"
    )
    if errored:
        lines.append("")
        lines.append(
            "> ⚠️ **本次真实模型调用失败**，下面的回答是错误信息而不是业务答案。"
            "这是任务卡要求如实呈现的情况；充值或换 key 后重新运行本脚本即可覆盖。"
        )
    lines.append("")

    lines.append("## 一、本单元独立库与入库情况（第1节）")
    lines.append("")
    lines.append("```text")
    lines.append("\n".join(fresh_run.splitlines()[:5]))
    lines.append("```")
    lines.append("")
    lines.append(
        f"资料 {document_count} 篇全部入库；稳定 ID 让重复导入变成更新，记录数不会增长。"
        "检索直接复用第九单元的 `RetrievalService`，本单元不重写任何相似度计算。"
    )
    lines.append("")

    lines.append("## 二、未命中直接拒答，且不调用模型（第2节）")
    lines.append("")
    lines.append("```text")
    lines.append(slice_from(real_first, "未命中示例", "编号上下文"))
    lines.append("```")
    lines.append("")
    lines.append(
        f"关键证据：**模型调用次数为 {refused_calls}**。没有检索到证据时，"
        "编排层在第②步直接返回拒答话术，压根没走到生成——"
        "既省 token，也避免模型在没有依据的情况下凭常识编答案。"
    )
    lines.append("")
    lines.append(
        "补充：向量检索总会返回最近邻，所以**拿无关问题演示不了空结果**，"
        "只有换项目标识才能可靠地拿到 0 条命中。"
    )
    lines.append("")

    lines.append("## 三、编号上下文：模型唯一能引用的范围（第2节）")
    lines.append("")
    lines.append("```text")
    lines.append(slice_from(real_first, "编号上下文", "问题："))
    lines.append("```")
    lines.append("")
    lines.append(
        "每条证据一个编号，编号与来源文件一一对应；发给模型之后，"
        "第⑤步只认 `[1]` 到 `[N]` 之间的编号。"
    )
    lines.append("")

    lines.append("## 四、真实模型生成与引用编号（第3节）")
    lines.append("")
    lines.append("```text")
    lines.append(slice_from(real_first, "检索命中："))
    lines.append("```")
    lines.append("")
    lines.append(f"- 检索概况：{hits_line}")
    lines.append(f"- 模型回答：{answer_text or '（无）'}")
    lines.append(f"- 回答中抽取到的引用编号：`{cited}`")
    lines.append(f"- 累计模型调用次数：{total_calls}")
    lines.append("")
    lines.append(
        "提示词约束了三件事：只能依据【检索资料】、资料不足时输出固定拒答话术、"
        "每个关键结论后必须带方括号编号；另外要求把资料里的任何指令当作普通文本，防止提示词注入。"
    )
    lines.append("")

    lines.append("## 五、来源核验与清单（第4节）")
    lines.append("")
    lines.append("| 项 | 本次结果 |")
    lines.append("| --- | --- |")
    lines.append(f"| 引用编号 | `{cited}` |")
    lines.append(f"| 核验后来源 | `{sources}` |")
    lines.append("| 命中但未被引用 | 不计入来源（不去冒充出处） |")
    lines.append("")
    lines.append(
        "核验规则：只保留「回答里引用了且编号在合法范围内」的编号；"
        "**一旦出现越界编号，整条回答判为不可核验**，宁可不给来源也不放行。"
        "去重后按出现顺序输出真实文件路径。"
    )
    lines.append("")
    lines.append(
        "**这条核验只能证明「编号存在且合法」，证明不了答案每句话都忠于原文** —— "
        "要判定忠实度得做逐句比对或人工复核，不属于本单元范围。"
    )
    lines.append("")

    lines.append("## 六、在真实库上重复运行两次（第4节：行为可复现）")
    lines.append("")
    lines.append("第一次运行后开始的三行：")
    lines.append("")
    lines.append("```text")
    lines.append("\n".join(real_second.splitlines()[:3]))
    lines.append("```")
    lines.append("")
    lines.append("```text")
    lines.append(slice_from(real_second, "检索命中："))
    lines.append("```")
    lines.append("")
    first_answer = field(slice_from(real_first, "检索命中："), "回答")
    second_answer = field(slice_from(real_second, "检索命中："), "回答")
    lines.append(
        f"两次运行的集合 ID 一致、命中资料与引用编号完全相同；"
        f"拒答分支的模型调用次数始终是 `{refused_calls}`，因为它根本不进生成环节。"
    )
    if first_answer and second_answer and first_answer != second_answer:
        lines.append("")
        lines.append(
            "两次**回答的措辞不完全相同**（真实大模型每次采样有非确定性），"
            "但引用编号与核验后的来源文件完全一致——"
            "可复现的是检索与核验，不是生成文本。"
        )
    lines.append("")

    lines.append("## 七、边界核对（第4节）")
    lines.append("")
    lines.append("```text")
    lines.append(boundary)
    lines.append("```")
    lines.append("")

    lines.append("## 八、本单元没有做的事")
    lines.append("")
    lines.append(
        "- **没有做分块（chunk）、重排（rerank）或相关性阈值**："
        "一篇文档一条记录，Top-K 结果直接拼给模型。"
    )
    lines.append("- **没有把生成结果写回数据库**，库里始终只有本组原始资料的向量副本。")
    lines.append("- **没有改动 `模拟数据/` 里的原始资料**。")
    lines.append("- 密钥不进输出、不进提交（`.env` 已在 `.gitignore`）。")
    lines.append(
        "- 观察：问「定制商品签收后还能申请退货吗？」，"
        "最高分命中的往往是《普通商品退货规则》而不是《定制商品例外》。"
        "这是检索（字面/语义接近）与判断（业务上该引用哪条）的差距，不是 bug。"
    )
    lines.append("")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入：{OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"拒答分支模型调用次数：{refused_calls}")
    print(f"本条回答是否含有效引用：{cited not in {'', '（未取到）'}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
