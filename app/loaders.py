"""单元12：把 Markdown、PDF、CSV 三种格式解析成同一种知识记录。

三种格式在磁盘上的"最小单位"完全不同：Markdown 是一整篇文档、PDF 是页、
CSV 是数据行。解析层要做的就是把它们压平成同一条记录，统一带上
**项目标识（project_id）、来源（source）、位置（location）** 三件套，
后面入库、检索、回读都只认这套字段。

格式差异只在这里处理一次：
- `.md`  → 一篇一条，`location_type=document`，`location=1`
- `.pdf` → 逐页一条，`location_type=page`，`location=页码`（从 1 开始）
- `.csv` → 一行一条，`location_type=row`，`location=表格行号`（从 2 开始，与人在表格里
  看到的行号一致；表头不算记录）

PDF 需要 `pypdf`：`uv sync --extra parse`

    uv run --extra parse python -c "from app.loaders import load_materials; print(len(load_materials(__import__('pathlib').Path('模拟数据/单元12项目资料/电商售后服务智能体项目组'), 'e-commerce-after-sales')[0]))"
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from app.data_store import PROJECT_ROOT

# 本单元资料放在这两组目录里：一组是本组电商售后，一组是课程另一个项目（差旅报销）。
# 两组共用同一套解析代码，只靠 project_id 区分——这正是"项目隔离"要演示的。
MATERIALS_DIR = PROJECT_ROOT / "模拟数据" / "单元12项目资料"


@dataclass(frozen=True)
class KnowledgeRecord:
    """一条解析产物：身份 + 原文 + 三件套（项目、来源、位置）。"""

    document_id: str
    text: str
    source: str
    file_type: str
    location_type: str
    location: int
    project_id: str

    @property
    def metadata(self) -> dict[str, object]:
        """写进 Chroma 的元数据：来源、格式、位置类型、位置、项目、资料类型。

        `doc_type` 沿用前几单元的字段名，这里直接取格式，方便按格式过滤回读。
        """
        return {
            "source": self.source,
            "file_type": self.file_type,
            "location_type": self.location_type,
            "location": self.location,
            "project_id": self.project_id,
            "doc_type": self.file_type,
        }


def parse_file(path: Path, project_id: str, root: Path | None = None) -> list[KnowledgeRecord]:
    """同一个函数处理三种格式，输出统一的来源与位置字段。

    root 用来把 source 记成"相对项目根目录"的路径；不传则退回文件名，
    解析逻辑不受影响，只是证据里显示长短不同。
    """
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project_id 不能为空，解析结果必须能追溯到项目")

    path = Path(path)
    suffix = path.suffix.lower()
    parts: list[tuple[str, str, int]] = []

    if suffix == ".md":
        # 课程样例较短，Markdown 按整个文件导入；本单元不重复讲分块（那是单元11 的事）。
        body = path.read_text(encoding="utf-8-sig").strip()
        if not body:
            raise ValueError("Markdown 正文为空")
        parts.append((body, "document", 1))
    elif suffix == ".pdf":
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ImportError as exc:  # pragma: no cover - 依赖缺失时才会走这里
            raise RuntimeError("缺少 pypdf。请执行：uv sync --extra parse") from exc
        try:
            pages = list(PdfReader(str(path)).pages)
        except PdfReadError as exc:
            raise ValueError(f"PDF 无法读取：{exc}") from exc
        # PDF 逐页提取可复制文字；扫描图片没有文字层，需要先做 OCR，
        # 所以"解析出 0 页"必须当成错误报出来，不能静默跳过。
        for page_number, page in enumerate(pages, start=1):
            body = (page.extract_text() or "").strip()
            if body:
                parts.append((body, "page", page_number))
        if not parts:
            raise ValueError("PDF 没有可提取文字；扫描件需先做 OCR")
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or any(not name for name in reader.fieldnames):
                raise ValueError("CSV 缺少有效表头")
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"CSV 第 {row_number} 行列数与表头不一致")
                # 字段名也要入文本，避免单独的编号或数值丢掉语义
                # （只存"EC-FAQ-01"四个字符，检索时它跟任何问题都不像）。
                body = "；".join(
                    f"{key}：{value.strip()}"
                    for key, value in row.items()
                    if value is not None and str(value).strip()
                )
                if body:
                    parts.append((body, "row", row_number))
        if not parts:
            raise ValueError("CSV 没有有效数据行")
    else:
        raise ValueError(f"不支持的文件格式：{suffix or '无扩展名'}")

    source = _relative_source(path, root)

    # 稳定 ID = 项目 + 文件名 + 位置类型 + 位置：
    # 再次运行解析同一个文件时 ID 不变，upsert 只是更新，不会追加第二条。
    return [
        KnowledgeRecord(
            document_id=f"{project_id}:{path.name}:{kind}:{location}",
            text=body,
            source=source,
            file_type=suffix.lstrip("."),
            location_type=kind,
            location=location,
            project_id=project_id,
        )
        for body, kind, location in parts
    ]


def _relative_source(path: Path, root: Path | None) -> str:
    """把文件路径记成相对项目根目录的形式，证据里一眼能看出资料在哪。"""
    base = Path(root) if root is not None else PROJECT_ROOT
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def load_materials(
    folder: Path, project_id: str, root: Path | None = None
) -> tuple[list[KnowledgeRecord], list[tuple[str, str]]]:
    """逐文件解析整个目录；坏文件只记录原因，不妨碍其他资料入库。"""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"项目资料目录不存在：{folder}")

    records: list[KnowledgeRecord] = []
    failures: list[tuple[str, str]] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        try:
            records.extend(parse_file(path, project_id, root))
        except (OSError, UnicodeError, ValueError, csv.Error, RuntimeError) as error:
            failures.append((path.name, str(error)))
    return records, failures


# 0–8 分钟要交的对照：三种格式各自的解析单位。
PARSE_UNITS = [
    ("md", "文档（整篇）", "一整篇 Markdown 一条记录", "课程样例短，本单元不再讲分块"),
    ("pdf", "页", "每一页各一条记录", "按可复制文字提取，扫描件无文字层需先 OCR"),
    ("csv", "数据行", "每个数据行一条记录", "表头不算记录，行号从 2 起，与表格行号一致"),
]

UNITS_HEADER = "格式\t解析单位\t一条记录是什么\t注意事项"


def render_parse_units(records: list[KnowledgeRecord]) -> str:
    """按格式统计实际解析出的条数，把"解析单位"这件事讲成一张表。"""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.file_type] = counts.get(record.file_type, 0) + 1

    lines = [
        "三种格式的解析单位（0–8 分钟）",
        UNITS_HEADER,
    ]
    for suffix, unit, meaning, note in PARSE_UNITS:
        lines.append(f".{suffix}\t{unit}\t{meaning}\t{note}")
    lines.append("")
    lines.append("本项目实际解析条数：" + "；".join(
        f".{suffix} {counts.get(suffix, 0)} 条" for suffix, _, _, _ in PARSE_UNITS
    ))
    return "\n".join(lines)


def render_parse_inventory(
    records: list[KnowledgeRecord],
    failures: list[tuple[str, str]],
    title: str = "解析清单",
) -> str:
    """8–17 分钟要交的清单：每条记录的项目、来源、位置与稳定 ID。"""
    lines = [
        f"{title}（共 {len(records)} 条）",
        "序号\t格式\t文件\tlocation_type=location\t稳定 ID\t正文字数",
    ]
    for index, record in enumerate(records, start=1):
        filename = record.source.rsplit("/", 1)[-1]
        lines.append(
            f"{index}\t.{record.file_type}\t{filename}\t"
            f"{record.location_type}={record.location}\t"
            f"{record.document_id}\t{len(record.text)}"
        )
    for filename, reason in failures:
        lines.append(f"解析失败\t{filename}\t原因：{reason}")
    return "\n".join(lines)


def render_record_explanation(record: KnowledgeRecord) -> str:
    """回读一条记录时顺带解释三个字段的含义，用于 17–27 分钟的输出。"""
    return (
        f"  project_id：{record.project_id}（属于哪个项目，查询时按它隔离）\n"
        f"  source：{record.source}（来自哪个文件，回读后据此核对出处）\n"
        f"  location：{record.location_type}={record.location}"
        f"（在这份文件里的第几页/第几行/第几篇）"
    )
