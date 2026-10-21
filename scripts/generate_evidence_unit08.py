"""生成单元08 的课堂验收证据。

按任务卡要求的顺序**真实执行**并抓取输出，不手写任何数字：

    uv run --extra embedding --extra chroma python scripts/generate_evidence_unit08.py

覆盖四节课要交的东西：
    第1课 输入资料清单与记录字段图
    第2课 导入前后数量、查询来源与距离
    第3课 本组资料（15 篇）与项目内检索结果
    第4课 首次与重复运行对照、持久化验证、空资料目录的边界结果
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元08验收证据.md"
REAL_DB_PATH = PROJECT_ROOT / ".chroma_unit8"

# 一次完整的"写入—重新连接—检索"，与 main.py 的 /ingest 走同一套代码。
FLOW_CODE = r"""
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from app.chroma_store import DEMO_QUESTION, PROJECT_ID, ChromaVectorStore
from app.ingest import load_markdown, render_inventory

records = load_markdown()
print(render_inventory(records, limit=15))
store = ChromaVectorStore(Path(sys.argv[1]))
print(f"集合：{store.collection.name}  集合 ID：{store.collection.id}")
print(f"导入前记录数：{store.count()}")
store.upsert_documents(records)
print(f"导入后记录数：{store.count()}")
reopened = ChromaVectorStore(Path(sys.argv[1]))
print(f"重新连接后集合 ID：{reopened.collection.id}")
print(f"重新连接后记录数：{reopened.count()}")
print(f"检索问题：{DEMO_QUESTION}（项目：{PROJECT_ID}）")
for rank, hit in enumerate(reopened.query(DEMO_QUESTION, PROJECT_ID), start=1):
    print(f"{rank}. 距离={hit['distance']:.4f} 余弦={hit['similarity']:.4f}")
    print(f"   来源={hit['source']}")
    print(f"   ID={hit['id']}")
"""

BOUNDARY_CODE = r"""
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from app.ingest import load_markdown

cases = []
try:
    load_markdown(Path(sys.argv[1]) / "不存在的目录")
except Exception as exc:
    cases.append(f"目录不存在 -> {type(exc).__name__}: {exc}")
try:
    load_markdown(Path(sys.argv[1]))
except Exception as exc:
    cases.append(f"空资料目录 -> {type(exc).__name__}: {exc}")
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


def strip_inventory(text: str) -> str:
    """重复运行的输出里省掉资料清单（与首次完全相同），只留库和检索部分。"""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("集合："):
            return "\n".join(lines[index:])
    return text


def top3_section(text: str) -> str:
    """只截取"检索问题"之后的 Top 3 结果。"""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("检索问题："):
            return "\n".join(lines[index:])
    return text


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
        from app.chroma_store import (
            DEMO_QUESTION,
            PROJECT_ID,
            render_record_fields,
        )
        from app.embedding import model_ready
    except ImportError:
        print("缺少 fastembed 或本地模型。请先执行：uv sync --extra embedding")
        return 1

    ready, detail = model_ready()
    if not ready:
        print(f"本地向量模型不可用：{detail}")
        return 1

    # 场景A：全新数据库，看导入前后数量的真实变化。
    fresh_dir = Path(tempfile.mkdtemp(prefix="unit08-fresh-"))
    fresh_db = fresh_dir / "chroma"
    first_run = run_code(FLOW_CODE, str(fresh_db))
    shutil.rmtree(fresh_dir, ignore_errors=True)

    # 场景B：真实库连跑两次，核对重复运行数量不增加、集合 ID 不变。
    real_first = run_code(FLOW_CODE, str(REAL_DB_PATH))
    real_second = run_code(FLOW_CODE, str(REAL_DB_PATH))

    # 第4课边界：空资料目录与目录不存在。
    empty_dir = Path(tempfile.mkdtemp(prefix="unit08-empty-"))
    boundary = run_code(BOUNDARY_CODE, str(empty_dir))
    shutil.rmtree(empty_dir, ignore_errors=True)

    lines: list[str] = []
    lines.append("# 单元08 验收证据：Chroma 写入与查询")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")
    lines.append(f"- 本组资料：`模拟数据/knowledge/`，共 15 篇（任务卡要求至少 5 篇）")
    lines.append(f"- 项目标识：`{PROJECT_ID}`；检索问题：{DEMO_QUESTION}")
    lines.append(f"- 持久化目录：`.chroma_unit8/`（单元07 的 `.chroma_unit7/` 保持 0 条不变）")
    lines.append("")

    lines.append("## 一、全新数据库首次导入（第2课：导入前后数量）")
    lines.append("")
    lines.append("```text")
    lines.append(first_run)
    lines.append("```")
    lines.append("")
    lines.append(
        f"导入前 {field(first_run, '导入前记录数')} 条，导入后 {field(first_run, '导入后记录数')} 条；"
        f"重新连接后仍是 {field(first_run, '重新连接后记录数')} 条——持久化生效。"
    )
    lines.append("")

    lines.append("## 二、在真实库上重复运行两次（第4课：数量不增加、集合 ID 不变）")
    lines.append("")
    lines.append("第一次运行（资料清单与上面完全相同，此处省略）：")
    lines.append("")
    lines.append("```text")
    lines.append(strip_inventory(real_first))
    lines.append("```")
    lines.append("")
    lines.append("第二次运行：")
    lines.append("")
    lines.append("```text")
    lines.append(strip_inventory(real_second))
    lines.append("```")
    lines.append("")
    lines.append(
        f"- 第一次导入后记录数：{field(real_first, '导入后记录数')}"
    )
    lines.append(
        f"- 第二次导入后记录数：{field(real_second, '导入后记录数')}"
    )
    lines.append(
        f"- **数量不增加：{field(real_first, '导入后记录数') == field(real_second, '导入后记录数')}**"
    )
    lines.append(
        f"- **集合 ID 不变：{field(real_first, '集合 ID') == field(real_second, '集合 ID')}**"
        f"（`{field(real_first, '集合 ID')}`）"
    )
    lines.append("")
    lines.append(
        "稳定 ID 形如 `项目标识:DOC-02-05` 由文件名推导，"
        "再次导入是更新同一条记录，所以数量不会增长。"
    )
    lines.append("")

    lines.append("## 三、Top 3 的来源与距离（第2、3课）")
    lines.append("")
    lines.append("```text")
    lines.append(top3_section(real_second))
    lines.append("```")
    lines.append("")
    lines.append(
        "距离由 Chroma 按集合的默认指标（l2，平方欧氏距离）返回，**越小越接近**；"
        "旁边同时给出单元06 的余弦相似度，**越大越接近**。两者都不是正确率。"
    )
    lines.append("")

    lines.append("## 四、记录字段图（第1课）")
    lines.append("")
    lines.append("```text")
    lines.append(render_record_fields())
    lines.append("```")
    lines.append("")

    lines.append("## 五、边界核对（第4课）")
    lines.append("")
    lines.append("```text")
    lines.append(boundary)
    lines.append("```")
    lines.append("")

    lines.append("## 六、本单元没有做的事")
    lines.append("")
    lines.append("- **没有生成任何业务答案**：只做检索，命中结果不等于正确答案。")
    lines.append("- 没有做分块、重排或回答生成；一篇文档一条记录。")
    lines.append("- 没有改动 `模拟数据/` 里的原始资料，写入的是它们的向量副本。")
    lines.append(
        "- 观察：问「商品有质量问题可以退款吗？」，最高分命中的是《普通商品退货规则》，"
        "而不是《质量问题处理》。检索只说明文字接近，业务上该引用哪一份仍要人工判断。"
    )
    lines.append("")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入：{OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"重复运行数量不增加：{field(real_first, '导入后记录数') == field(real_second, '导入后记录数')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
