"""生成单元07 的课堂验收证据。

按任务卡要求的顺序**真实执行**脚本并抓输出，不手写任何数字：

    uv run --extra embedding --extra chroma python scripts/generate_evidence_unit07.py
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元07验收证据.md"
SETUP = PROJECT_ROOT / "setup_chroma.py"
INSPECT = PROJECT_ROOT / "inspect_chroma.py"


def run_script(script: Path) -> str:
    """在独立进程里跑脚本，抓回 stdout；失败直接抛出，不静默吞掉。"""
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


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
        from app.vector_store import (
            DEMO_QUERY,
            TENANT_ID,
            InMemoryVectorStore,
            build_demo_records,
            render_field_diagram,
        )
    except ImportError:
        print("缺少 fastembed 或本地模型。请先执行：uv sync --extra embedding")
        return 1

    # 第3、4课：真实执行建库两次 + 只读查看一次。
    setup_first = run_script(SETUP)
    setup_second = run_script(SETUP)
    inspect_output = run_script(INSPECT)

    def field(text: str, label: str) -> str:
        for line in text.splitlines():
            if line.startswith(label):
                return line.split("：", 1)[1].strip()
        return "（未取到）"

    first_id = field(setup_first, "集合 ID")
    second_id = field(setup_second, "集合 ID")

    # 第1、2课：内存实现的精确检索基线。
    store = InMemoryVectorStore("after_sale_knowledge_base_demo")
    items = build_demo_records()
    store.add_many(items)
    hits = store.query(DEMO_QUERY, TENANT_ID, top_k=3)
    typed = store.query(DEMO_QUERY, TENANT_ID, top_k=2, doc_type="knowledge")

    lines: list[str] = []
    lines.append("# 单元07 验收证据：向量数据库选型与核心原理")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")

    lines.append("## 一、执行一次 `setup_chroma.py`（第3课）")
    lines.append("")
    lines.append("```text")
    lines.append(setup_first)
    lines.append("```")
    lines.append("")

    lines.append("## 二、重复执行 `setup_chroma.py`，核对集合 ID 不变（第4课）")
    lines.append("")
    lines.append("```text")
    lines.append(setup_second)
    lines.append("```")
    lines.append("")
    lines.append(f"- 第一次集合 ID：`{first_id}`")
    lines.append(f"- 第二次集合 ID：`{second_id}`")
    lines.append(f"- **两次一致：{first_id == second_id}**")
    lines.append("")

    lines.append("## 三、执行 `inspect_chroma.py` 只读查看（第3课）")
    lines.append("")
    lines.append("```text")
    lines.append(inspect_output)
    lines.append("```")
    lines.append("")

    lines.append("## 四、记录字段图（第1课）")
    lines.append("")
    lines.append("```text")
    lines.append(render_field_diagram())
    lines.append("```")
    lines.append("")

    lines.append("## 五、精确遍历基线：内存集合检索（第2课对照标准）")
    lines.append("")
    lines.append("```text")
    lines.append(
        f"集合：{store.collection_name}；维度：{store.dimension}；"
        f"记录数：{store.count()}"
        f"（知识文档 {store.count(doc_type='knowledge')} 条、"
        f"样例问题 {store.count(doc_type='question')} 条）"
    )
    lines.append(f"查询：{DEMO_QUERY}")
    for hit in hits:
        lines.append(
            f"  {hit.score:.4f}  {hit.record.document_id}  "
            f"类型={hit.record.doc_type}  来源={hit.record.source}"
        )
    lines.append(f"元数据过滤 doc_type=knowledge 后：")
    for hit in typed:
        lines.append(f"  {hit.score:.4f}  {hit.record.document_id}")
    lines.append(
        f"项目隔离（换成 other-project）：{len(store.query(DEMO_QUERY, 'other-project'))} 条"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "这里是全量遍历的**精确**结果，后面用 HNSW/IVF 换来的速度，"
        "都要拿它当标准答案去量 Recall@K。"
    )
    lines.append("")

    lines.append("## 六、本单元没有做的事")
    lines.append("")
    lines.append("- **没有往 Chroma 写入任何业务文档**，集合记录数为 0；写入与查询留给单元08。")
    lines.append("- 没有实现 HNSW / IVF，本单元只讲原理。")
    lines.append("- 没有做三库性能实测，因此选型决策卡里没有任何速度或召回排名。")
    lines.append("")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入：{OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"两次集合 ID 一致：{first_id == second_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
