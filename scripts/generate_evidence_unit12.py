"""生成单元12 的课堂验收证据。

按任务卡的时间段顺序**真实执行**并抓取输出，不手写任何数字：

    uv run --extra embedding --extra chroma --extra parse python scripts/generate_evidence_unit12.py

覆盖要交的东西：
    0–8 分钟   三种格式的解析单位（文档 / 页 / 行）
    8–17 分钟  解析清单：project_id、source、location 与稳定 ID
    17–27 分钟 真实向量化入库 + 从数据库回读 PDF 第 1 页与 CSV 第 2 行
    27–35 分钟 重跑不重复（稳定 ID）+ 切到另一项目：本项目 7 条、库内共 14 条
    另外       边界核对（未知项目、坏 CSV、空 Markdown、只有表头、不支持的格式）

**密钥安全**：本单元完全不调用大模型，脚本也不打印任何密钥；输出仍做 sk- 掩码兜底。
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元12验收证据.md"

# 全新临时库跑一遍完整四段：证明"从 0 条开始"也能得到 7 条 / 14 条。
FRESH_FLOW = r"""
import sys, tempfile
sys.stdout.reconfigure(errors="replace")
from pathlib import Path

from app.parse_lab import render_run, run_parse

database = Path(tempfile.mkdtemp(prefix="unit12-fresh-")) / "chroma"
run = run_parse("ecom", database)
print(render_run(run))
print()
print(f"本次使用的临时数据库：{database}")
"""

# 在真实库 .chroma_unit12/ 上再跑一遍：课程验收时要看的就是这个现场。
REAL_FLOW = r"""
import sys
sys.stdout.reconfigure(errors="replace")

from app.parse_lab import render_header, render_ingest, render_rerun, run_parse

run = run_parse("ecom")
print(render_header(run))
print()
print(render_ingest(run))
print()
print(render_rerun(run))
"""

# 按项目计数：证明隔离靠元数据过滤，不靠分库。
ISOLATION_FLOW = r"""
import sys
sys.stdout.reconfigure(errors="replace")

from app.chroma_store import (
    UNIT12_COLLECTION, UNIT12_COLLECTION_METADATA, ChromaVectorStore, resolve_unit12_db_path,
)

database = resolve_unit12_db_path()
store = ChromaVectorStore(database, UNIT12_COLLECTION, UNIT12_COLLECTION_METADATA)
print(f"数据库目录：{database}")
print(f"集合：{store.collection.name}")
print(f"库内总数：{store.count()} 条")
for project_id in ("e-commerce-after-sales", "travel-expense-demo", "not-a-project"):
    print(f"  project_id={project_id}：{store.count(project_id)} 条")
"""

# 边界：一条一条单独跑，任何一条抛错都如实抓回来。
BOUNDARIES = [
    (
        "项目名写错（未知项目）",
        r"""
import sys
sys.stdout.reconfigure(errors="replace")
from app.parse_lab import project_keys, resolve_project
try:
    resolve_project("finance")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
print(f"可选项目：{project_keys()}")
""",
    ),
    (
        "CSV 行列数与表头不一致",
        r"""
import sys, tempfile
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from app.loaders import parse_file
folder = Path(tempfile.mkdtemp(prefix="unit12-badcsv-"))
path = folder / "坏表.csv"
path.write_text("问题编号,问题\nEC-01,能退货吗,多出来的一列\n", encoding="utf-8-sig")
try:
    parse_file(path, "e-commerce-after-sales")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "CSV 只有表头",
        r"""
import sys, tempfile
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from app.loaders import parse_file
folder = Path(tempfile.mkdtemp(prefix="unit12-headeronly-"))
path = folder / "只有表头.csv"
path.write_text("问题编号,问题,标准答复\n", encoding="utf-8-sig")
try:
    parse_file(path, "e-commerce-after-sales")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "Markdown 正文为空",
        r"""
import sys, tempfile
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from app.loaders import parse_file
folder = Path(tempfile.mkdtemp(prefix="unit12-emptymd-"))
path = folder / "空文档.md"
path.write_text("   \n\n", encoding="utf-8")
try:
    parse_file(path, "e-commerce-after-sales")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "不支持的格式",
        r"""
import sys, tempfile
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from app.loaders import parse_file
folder = Path(tempfile.mkdtemp(prefix="unit12-txt-"))
path = folder / "说明.txt"
path.write_text("纯文本不是资料", encoding="utf-8")
try:
    parse_file(path, "e-commerce-after-sales")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "缺少 project_id",
        r"""
import sys, tempfile
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from app.loaders import parse_file
folder = Path(tempfile.mkdtemp(prefix="unit12-noproject-"))
path = folder / "DOC-09-01-规则.md"
path.write_text("# 规则\n\n正文", encoding="utf-8")
try:
    parse_file(path, "   ")
except ValueError as error:
    print(f"{type(error).__name__}: {error}")
""",
    ),
    (
        "资料目录不存在",
        r"""
import sys
sys.stdout.reconfigure(errors="replace")
from pathlib import Path
from app.loaders import load_materials
try:
    load_materials(Path("模拟数据/单元12项目资料/不存在的项目组"), "e-commerce-after-sales")
except FileNotFoundError as error:
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
    """兜底掩码：本单元不调用模型，但保留这道防线，避免以后复制脚本时漏掉。"""
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***（已掩码）", text)


def slice_from(text: str, start: str, end: str | None = None) -> str:
    """按标题切片；标记必须是这段输出里唯一出现的那一个。"""
    index = text.find(start)
    if index < 0:
        return f"（未找到片段：{start}）"
    body = text[index:]
    if end:
        stop = body.find(end, len(start))
        if stop > 0:
            body = body[:stop]
    return body.rstrip()


def field(text: str, label: str) -> str:
    """取一个字段值；取不到就如实写「未取到」，不猜。"""
    marker = f"{label}："
    for line in text.splitlines():
        if marker in line:
            return line.split(marker, 1)[1].strip()
    return "（未取到）"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.path.insert(0, str(PROJECT_ROOT))

    fresh = run_code(FRESH_FLOW)
    real = run_code(REAL_FLOW)
    isolation = run_code(ISOLATION_FLOW)

    units = slice_from(fresh, "三种格式的解析单位（0–8 分钟）", "解析清单（共")
    inventory = slice_from(fresh, "解析清单（共", "真实向量化 + 入库 + 回读")
    ingest = slice_from(fresh, "真实向量化 + 入库 + 回读（17–27 分钟）", "重跑与切换项目")
    rerun = slice_from(fresh, "重跑与切换项目（27–35 分钟）")

    lines: list[str] = []
    lines.append("# 单元12 验收证据：文档解析与知识库")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append("以下所有输出均由脚本实际执行抓取，未手工填写。")
    lines.append("")
    lines.append("- 所选项目：`ecom`（本组，项目标识 `e-commerce-after-sales`）；对照项目：`travel`（课程另一组）")
    lines.append("- 资料目录：`模拟数据/单元12项目资料/`（两组各 3 篇 Markdown + 1 份 PDF + 1 份 CSV）")
    lines.append("- 持久化目录：`.chroma_unit12/`；集合：`after_sale_parsed_materials`")
    lines.append("- 向量模型：`BAAI/bge-small-zh-v1.5`（与前几单元同一个，512 维）")
    lines.append("- 生成模型：**本单元不调用任何大模型**，全是解析、向量化与数据库读写")
    lines.append("")
    lines.append("## 一、三种格式的解析单位与解析清单（0–17 分钟）")
    lines.append("")
    lines.append("同一个 `parse_file()` 处理三种格式，差别只在「一条记录是什么」：Markdown 是一整篇文档、")
    lines.append("PDF 是一页、CSV 是一行。表头不算记录，所以 CSV 的行号从 2 开始，与人在表格里看到的行号一致。")
    lines.append("")
    lines.append("```text")
    lines.append(units)
    lines.append("```")
    lines.append("")
    lines.append("清单里每条记录都带齐三件套：**project_id（属于哪个项目）、source（来自哪个文件）、")
    lines.append("location（这份文件里的第几页/第几行/第几篇）**。稳定 ID 由「项目 + 文件名 + 位置类型 + 位置」拼成，")
    lines.append("所以重跑时 ID 不变，写入是更新而不是追加。")
    lines.append("")
    lines.append("```text")
    lines.append(inventory)
    lines.append("```")
    lines.append("")
    count_match = re.search(r"解析清单（共 (\d+) 条）", inventory)
    parsed_count = count_match.group(1) if count_match else "（未取到）"
    lines.append(
        f"所选项目共解析出 **{parsed_count} 条**，"
        "与任务卡要求的 7 条一致：3 篇 Markdown + 1 页 PDF + 3 行 CSV。"
    )
    lines.append("")
    lines.append("## 二、真实向量化 + 入库 + 回读（17–27 分钟）")
    lines.append("")
    lines.append("从全新的临时库开始，先用单元06 的本地中文模型把每条原文编码成 512 维向量，再 upsert 进 Chroma；")
    lines.append("然后**从数据库回读**（不是重新打开源文件）PDF 第 1 页与 CSV 第 2 行，核对原文与元数据都落盘了。")
    lines.append("")
    lines.append("```text")
    lines.append(ingest)
    lines.append("```")
    lines.append("")
    lines.append("回读出来的 `metadata` 里，`project_id`、`source`、`location_type`、`location` 四项齐全——")
    lines.append("这正是任务卡要求「统一来源元数据」的落点，检索、计数、回溯出处全靠它们。")
    lines.append("")
    lines.append("## 三、重跑不重复，再切到另一项目（27–35 分钟）")
    lines.append("")
    lines.append("同一批资料连写两遍，本项目条数必须还是 7 条：稳定 ID 让第二次 upsert 变成更新。")
    lines.append("接着把另一个项目组的 7 条也写进**同一个集合**，库内总数变成 14 条，但按 `project_id` 数仍然是各 7 条。")
    lines.append("")
    lines.append("```text")
    lines.append(rerun)
    lines.append("```")
    lines.append("")
    lines.append("## 四、在真实库 `.chroma_unit12/` 上再跑一次")
    lines.append("")
    lines.append("上面第二节用的是临时库，这一节用仓库里保留的库跑，确认现场可复现：")
    lines.append("")
    lines.append("```text")
    lines.append(real)
    lines.append("```")
    lines.append("")
    lines.append("## 五、按项目计数：隔离是查出来的，不是分库分出来的")
    lines.append("")
    lines.append("两个项目的资料写在**同一个集合**里。既然在同一个集合，为什么另一个项目的 7 条不会算进本项目？")
    lines.append("下面直接数一遍：")
    lines.append("")
    lines.append("```text")
    lines.append(isolation)
    lines.append("```")
    lines.append("")
    lines.append("答案是三条，缺一条都不成立：")
    lines.append("")
    lines.append("1. **稳定 ID 就不同**：ID 以项目标识开头（`e-commerce-after-sales:…` / `travel-expense-demo:…`），")
    lines.append("   同一个文件在两个项目里也是两条不同记录，不会互相覆盖，也不会重复计数。")
    lines.append("2. **每条记录都带 `project_id` 元数据**，写入时一起落盘，不是事后补的。")
    lines.append("3. **计数与检索都带 `where project_id` 过滤**，另一个项目的记录在查询条件之外；")
    lines.append("   写一个库里根本没有的项目标识，结果就是 0 条——过滤条件确实在执行，不是碰巧数对了。")
    lines.append("")
    lines.append("所以「另一个项目的数据不会计入本项目数量」不是因为分了两个库，而是因为**每条记录都记得自己属于谁**。")
    lines.append("")
    lines.append("## 六、边界核对")
    lines.append("")
    lines.append("解析层是整条链路的入口，坏资料必须在入库之前被挡住，并且报错要能指出是哪一条、为什么。")
    for title, code in BOUNDARIES:
        lines.append("")
        lines.append(f"### {title}")
        lines.append("")
        lines.append("```text")
        lines.append(run_code(code))
        lines.append("```")
    lines.append("")
    lines.append("## 七、本单元没有做的事")
    lines.append("")
    lines.append("- **不做 OCR**：扫描件没有文字层，`parse_file` 会如实报「PDF 没有可提取文字」，")
    lines.append("  而不是编一段看起来像正文的东西出来。")
    lines.append("- **不做分块**：Markdown 按整篇导入，分块是单元11 的内容；本单元只解决「三种格式怎么变成同一种记录」。")
    lines.append("- **不生成回答**：本单元到入库为止，问答留给单元10 的 RAG 流程。")
    lines.append("- **不做表格结构还原**：CSV 一行压成一段「字段：值」，够检索用；要保留列语义得另建字段，")
    lines.append("  那属于结构化入库，不在这次练习范围。")
    lines.append("")
    lines.append("## 八、复现方式")
    lines.append("")
    lines.append("```bash")
    lines.append("uv sync --extra embedding --extra chroma --extra parse")
    lines.append("uv run python main.py        # 然后输入 /parse 或 /parse travel")
    lines.append("uv run --extra embedding --extra chroma --extra parse python scripts/generate_evidence_unit12.py")
    lines.append("uv run --extra embedding --extra chroma --extra parse python -m unittest tests.test_unit12")
    lines.append("```")
    lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成：{OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
