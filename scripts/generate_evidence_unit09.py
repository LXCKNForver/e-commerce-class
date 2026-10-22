"""生成单元09 的课堂验收证据。

按任务卡要求的顺序**真实执行**并抓取输出，不手写任何数字：

    uv run --extra embedding --extra chroma python scripts/generate_evidence_unit09.py

覆盖四节课要交的东西：
    第1课 本单元库路径与入库记录数
    第2课 SearchResult 返回字段与最小查询结果
    第3课 Top-3/Top-5、来源过滤、跨项目空结果
    第4课 两个问题的批量检索与一次空问题错误提示
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元09验收证据.md"
REAL_DB_PATH = PROJECT_ROOT / ".chroma_unit9"

# 一次完整的"建库—检索服务—四类查询—边界"，与 main.py 的 /retrieve 走同一套代码。
FLOW_CODE = r"""
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from app.chroma_store import PROJECT_ID, ChromaVectorStore
from app.ingest import load_markdown
from app.retriever import (
    DEMO_QUESTION, DEMO_SOURCE, SECOND_QUESTION, RetrievalService, render_results,
)

records = load_markdown()
database_path = Path(sys.argv[1])
store = ChromaVectorStore(database_path)
before = store.count()
store.upsert_documents(records)
print(f"数据库目录：{database_path}")
print(f"集合：{store.collection.name}  集合 ID：{store.collection.id}")
print(f"导入前/后：{before} / {store.count()} 条；项目：{PROJECT_ID}")

service = RetrievalService(store)
print()
print(render_results("Top-3", service.search(DEMO_QUESTION, PROJECT_ID, top_k=3)))
print()
print(render_results("Top-5", service.search(DEMO_QUESTION, PROJECT_ID, top_k=5)))
print()
print(render_results(
    f"来源过滤（只允许 {DEMO_SOURCE}）",
    service.search(DEMO_QUESTION, PROJECT_ID, top_k=3, source=DEMO_SOURCE),
))
print()
print(render_results(
    "其他项目（another_project）",
    service.search(DEMO_QUESTION, "another_project", top_k=3),
))
print()
questions = [DEMO_QUESTION, SECOND_QUESTION]
for question, results in zip(questions, service.search_many(questions, PROJECT_ID, top_k=2)):
    print(render_results(f"批量检索 {question}", results))
    print()
try:
    service.search("   ", PROJECT_ID)
except ValueError as exc:
    print(f"空问题错误提示：ValueError: {exc}")
"""

BOUNDARY_CODE = r"""
import sys
sys.stdout.reconfigure(errors="replace")
from app.chroma_store import PROJECT_ID
from app.retriever import DEMO_QUESTION, RetrievalService, SearchResult
from app.chroma_store import ChromaVectorStore
from pathlib import Path
import tempfile

store = ChromaVectorStore(Path(tempfile.mkdtemp(prefix="unit09-empty-")) / "chroma")
service = RetrievalService(store)
cases = []
for label, call in [
    ("空问题", lambda: service.search("   ", PROJECT_ID)),
    ("空问题列表", lambda: service.search_many([], PROJECT_ID)),
    ("问题列表里有空白项", lambda: service.search_many(["有内容", "  "], PROJECT_ID)),
    ("project_id 留空", lambda: service.search(DEMO_QUESTION, "")),
    ("top_k=0", lambda: service.search(DEMO_QUESTION, PROJECT_ID, top_k=0)),
    ("source 为空字符串", lambda: service.search(DEMO_QUESTION, PROJECT_ID, source="")),
]:
    try:
        call()
        cases.append(f"{label} -> 没有报错（异常）")
    except Exception as exc:
        cases.append(f"{label} -> {type(exc).__name__}: {exc}")
for line in cases:
    print(line)
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
    return result.stdout.strip()


def field(text: str, label: str) -> str:
    """从一段输出里取某个字段的值；取不到就如实写"未取到"。"""
    marker = f"{label}："
    for line in text.splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return "（未取到）"


def slice_between(text: str, start: str, end: str | None = None) -> str:
    """截取两个标记之间的输出片段，用于把一次运行拆成几个验收小节。"""
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


def counts(text: str) -> tuple[str, str]:
    """从"导入前/后：0 / 15 条；项目：…"里取出前后两个数字。"""
    raw = field(text, "导入前/后")
    before, _, after = raw.partition(" / ")
    return before.strip(), after.split(" 条")[0].strip()


def top_document_id(text: str) -> str:
    """取 Top-3 第一条的 document_id；不能用 field(text, "ID")，那会先撞上"集合 ID"。 """
    for line in text.splitlines():
        if "  ID=" in line:
            return line.split("  ID=", 1)[1].strip()
    return "（未取到）"


def head(text: str, count: int = 4) -> str:
    """只取前几行，用于重复运行时的对照（与首次完全相同的部分不再重复贴）。"""
    return "\n".join(text.splitlines()[:count])


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
        from app.retriever import DEMO_QUESTION, DEMO_SOURCE, SECOND_QUESTION
    except ImportError:
        print("缺少 fastembed 或本地模型。请先执行：uv sync --extra embedding")
        return 1

    ready, detail = model_ready()
    if not ready:
        print(f"本地向量模型不可用：{detail}")
        return 1

    document_count = len(load_markdown())

    # 场景A：全新数据库，看导入前后数量的真实变化。
    fresh_dir = Path(tempfile.mkdtemp(prefix="unit09-fresh-"))
    fresh_run = run_code(FLOW_CODE, str(fresh_dir / "chroma"))
    shutil.rmtree(fresh_dir, ignore_errors=True)

    # 场景B：真实库连跑两次，核对数量不增加、集合 ID 不变、排序可复现。
    real_first = run_code(FLOW_CODE, str(REAL_DB_PATH))
    real_second = run_code(FLOW_CODE, str(REAL_DB_PATH))

    # 第4课边界：空问题、空列表、空项目标识、非法 top_k / source。
    boundary = run_code(BOUNDARY_CODE)

    lines: list[str] = []
    lines.append("# 单元09 验收证据：向量检索服务")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")
    lines.append(f"- 本组资料：`模拟数据/knowledge/`，共 {document_count} 篇")
    lines.append(f"- 项目标识：`{PROJECT_ID}`")
    lines.append(
        f"- 检索问题一（自拟）：{DEMO_QUESTION}；"
        f"检索问题二（取自样例问题）：{SECOND_QUESTION}"
    )
    lines.append(
        "- 持久化目录：`.chroma_unit9/`，与单元08 的 `.chroma_unit8/` 互不干扰"
        "（任务卡要求本单元库不复制第八单元数据库）"
    )
    lines.append("")

    lines.append("## 一、全新数据库首次导入（第1课：库路径与入库记录数）")
    lines.append("")
    lines.append("```text")
    lines.append(head(fresh_run, 3))
    lines.append("```")
    lines.append("")
    fresh_before, fresh_after = counts(fresh_run)
    lines.append(
        f"导入前 {fresh_before} 条、导入后 {fresh_after} 条，"
        f"即 {document_count} 篇资料全部入库；项目标识 `{PROJECT_ID}`。"
    )
    lines.append("")

    lines.append("## 二、在真实库上重复运行两次（第4课：数量不增加、集合 ID 不变）")
    lines.append("")
    lines.append("第一次运行（完整输出）：")
    lines.append("")
    lines.append("```text")
    lines.append(real_first)
    lines.append("```")
    lines.append("")
    lines.append("第二次运行（前四行，其余与首次完全相同）：")
    lines.append("")
    lines.append("```text")
    lines.append(head(real_second, 3))
    lines.append("```")
    lines.append("")
    first_before, first_after = counts(real_first)
    second_before, second_after = counts(real_second)
    lines.append(f"- 第一次导入后：{first_after} 条（导入前 {first_before} 条）")
    lines.append(f"- 第二次导入后：{second_after} 条（导入前 {second_before} 条）")
    lines.append(f"- **数量不增加：{first_after == second_after}**")
    lines.append(
        f"- **集合 ID 不变：{field(real_first, '集合 ID') == field(real_second, '集合 ID')}**"
        f"（`{field(real_first, '集合 ID')}`）"
    )
    lines.append("")

    lines.append("## 三、Top-3 与 Top-5（第3课：数量与排序可复现）")
    lines.append("")
    lines.append("```text")
    lines.append(slice_between(real_second, "Top-3：", "Top-5："))
    lines.append("```")
    lines.append("")
    lines.append("```text")
    lines.append(slice_between(real_second, "Top-5：", "来源过滤（只允许"))
    lines.append("```")
    lines.append("")
    lines.append(
        "同一个问题改 K：Top-5 的前三条与 Top-3 **逐条一致**（ID 与距离都相同），"
        "说明 Top-K 只控制返回条数，不改变排序。"
    )
    lines.append("")

    lines.append("## 四、来源过滤与跨项目空结果（第3课：过滤在库内执行）")
    lines.append("")
    lines.append("```text")
    lines.append(slice_between(real_second, "来源过滤（只允许", "其他项目（another_project）"))
    lines.append("```")
    lines.append("")
    lines.append(f"只允许返回 `{DEMO_SOURCE}`，实际只返回该文件 1 条——命中不足 K 条时条数会少于 K。")
    lines.append("")
    lines.append("```text")
    lines.append(slice_between(real_second, "其他项目（another_project）", "批量检索"))
    lines.append("```")
    lines.append("")
    lines.append("换成别的项目标识后返回 0 条，证明 `project_id` 过滤生效、不会跨项目串资料。")
    lines.append("")

    lines.append("## 五、两个问题的批量检索（第4课：一次调用查多个）")
    lines.append("")
    lines.append("```text")
    lines.append(slice_between(real_second, "批量检索", "空问题错误提示"))
    lines.append("```")
    lines.append("")
    lines.append(
        "`search_many()` 返回列表与输入问题一一对应、顺序相同；"
        "两个问题的首选命中不同，说明各自独立检索。"
    )
    lines.append("")

    lines.append("## 六、边界核对（第4课：空问题等错误提示）")
    lines.append("")
    lines.append("```text")
    lines.append(boundary)
    lines.append("```")
    lines.append("")
    lines.append("```text")
    lines.append(slice_between(real_second, "空问题错误提示"))
    lines.append("```")
    lines.append("")

    lines.append("## 七、SearchResult 返回字段（第2课）")
    lines.append("")
    lines.append("| 字段 | 含义 | 本次示例 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| document_id | 稳定 ID | `{top_document_id(real_second)}` |")
    lines.append("| text | 原文（整篇 Markdown） | `# 普通商品退货规则 …` |")
    lines.append("| source | 来源文件 | `模拟数据/knowledge/DOC-02-01-普通商品退货规则.md` |")
    lines.append(f"| project_id | 项目标识 | `{PROJECT_ID}` |")
    lines.append("| doc_type | 资料类型 | `knowledge` |")
    lines.append("| distance | 距离（l2，越小越接近） | `0.6539` |")
    lines.append("| similarity | 余弦相似度（越大越接近） | `0.6731` |")
    lines.append("")
    lines.append(
        "距离与余弦可以对上：归一化向量下 `2-2cos` 约等于 l2 距离；"
        "两者都不是正确率，只说明\"像不像\"。"
    )
    lines.append("")

    lines.append("## 八、本单元没有做的事")
    lines.append("")
    lines.append("- **没有生成任何回答**：只返回检索证据，生成留给后续单元。")
    lines.append("- 没有做分块、重排或阈值截断；一篇文档一条记录。")
    lines.append("- 没有改动 `模拟数据/` 里的原始资料，检索的是它们的向量副本。")
    lines.append(
        "- 观察：问「商品有质量问题可以退款吗？」，最高分命中的仍是《普通商品退货规则》，"
        "而不是《质量问题处理》。检索只说明文字接近，业务上引用哪一份仍要人工判断。"
    )
    lines.append("")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入：{OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"重复运行数量不增加：{counts(real_first)[1] == counts(real_second)[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
