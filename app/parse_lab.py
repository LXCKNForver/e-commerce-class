"""单元12 的编排层：解析 → 入库 → 回读 → 重跑 → 切换项目。

解析的格式差异在 `app/loaders.py`，数据库读写在 `app/chroma_store.py`，
这里只负责把任务卡四段时间串成一条可复现的流水线，并把每一步打印出来。

任务卡要求的两个数字写在这里，别在别处再写一遍：
    所选项目解析 **7 条**（3 篇 Markdown + 1 页 PDF + 3 行 CSV）
    两个项目都入库后 **库内共 14 条**，但按 project_id 数仍然是各 7 条

    uv run --extra embedding --extra chroma --extra parse python -c "from app.parse_lab import run_parse; print(len(run_parse('ecom').records))"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.chroma_store import (
    UNIT12_COLLECTION,
    UNIT12_COLLECTION_METADATA,
    ChromaVectorStore,
    resolve_unit12_db_path,
)
from app.loaders import (
    MATERIALS_DIR,
    KnowledgeRecord,
    load_materials,
    render_parse_inventory,
    render_parse_units,
    render_record_explanation,
)


@dataclass(frozen=True)
class ProjectSpec:
    """一个可选的项目资料组：命令行参数 → 项目标识 + 资料目录。"""

    key: str
    project_id: str
    label: str
    folder: Path


# 学生练习原话是 `main.py --project ecom` / `--project travel`，这里沿用同样的两个键。
PROJECTS: dict[str, ProjectSpec] = {
    "ecom": ProjectSpec(
        key="ecom",
        project_id="e-commerce-after-sales",
        label="电商售后服务智能体",
        folder=MATERIALS_DIR / "电商售后服务智能体项目组",
    ),
    "travel": ProjectSpec(
        key="travel",
        project_id="travel-expense-demo",
        label="差旅与报销审核助手",
        folder=MATERIALS_DIR / "差旅与报销审核助手项目组",
    ),
}
DEFAULT_PROJECT = "ecom"
# 完成标准：所选项目解析 7 条；两个项目都入库后库内共 14 条。
EXPECTED_RECORDS = 7
EXPECTED_TOTAL = 14


def project_keys() -> str:
    return " / ".join(PROJECTS)


def resolve_project(key: str | None) -> ProjectSpec:
    """把命令行参数换成项目；不认识的参数直接给出可选值，不要静默回退。"""
    chosen = (key or DEFAULT_PROJECT).strip().lower()
    if chosen not in PROJECTS:
        raise ValueError(f"未知项目：{chosen}；可选：{project_keys()}")
    return PROJECTS[chosen]


@dataclass
class ParseRun:
    """一次完整演练的结果，字段全部来自真实运行，渲染函数只读不写。"""

    project: ProjectSpec
    database_path: Path
    collection_name: str
    records: list[KnowledgeRecord] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)
    before_count: int = 0
    after_count: int = 0
    reread: list[dict] = field(default_factory=list)
    rerun_count: int = 0
    other_project: ProjectSpec | None = None
    other_records: int = 0
    after_other_count: int = 0
    total_count: int = 0
    # 课程另一个项目组的资料目录可能不在仓库里（例如被人从工作区删掉）：
    # 这时不整条流水线崩掉，而是如实记下"缺对照项目"，让用户看得见原因。
    other_missing: bool = False

    @property
    def stable(self) -> bool:
        """重跑后本项目条数没变 = 稳定 ID 生效（是更新，不是追加）。"""
        return self.rerun_count == self.after_count


def run_parse(
    project_key: str | None = None,
    database_path: Path | None = None,
    with_other_project: bool = True,
) -> ParseRun:
    """跑完四段时间的全部动作，返回可渲染的结果对象。"""
    project = resolve_project(project_key)
    db_path = Path(database_path) if database_path else resolve_unit12_db_path()
    store = ChromaVectorStore(db_path, UNIT12_COLLECTION, UNIT12_COLLECTION_METADATA)

    # 8–17 分钟：解析所选项目的资料，坏文件单独记录原因。
    records, failures = load_materials(project.folder, project.project_id)
    if not records:
        raise RuntimeError(f"没有可入库的文本，请检查资料目录与解析失败信息：{project.folder}")

    run = ParseRun(
        project=project,
        database_path=db_path,
        collection_name=store.collection.name,
        records=records,
        failures=failures,
    )

    # 17–27 分钟：真实向量化 + 入库。入库前后都只数"本项目"的条数。
    run.before_count = store.count(project.project_id)
    store.upsert_documents(records)
    run.after_count = store.count(project.project_id)

    # 回读 PDF 第 1 页与 CSV 第 2 行：从数据库里读，不重新打开源文件。
    for file_type, location in (("pdf", 1), ("csv", 2)):
        example = next(
            (
                item
                for item in records
                if item.file_type == file_type and item.location == location
            ),
            None,
        )
        if example is None:
            continue
        saved = store.read_record(example.document_id)
        run.reread.append({**saved, "record": example})

    # 27–35 分钟：重跑一次，靠稳定 ID 验证条数不增长。
    store.upsert_documents(records)
    run.rerun_count = store.count(project.project_id)

    # 再切到另一个项目：库里变成 14 条，但本项目仍然只有 7 条。
    if with_other_project:
        other = next(spec for spec in PROJECTS.values() if spec.key != project.key)
        if not other.folder.is_dir():
            # 对照项目的资料不在工作区里：如实记录，不伪造那 7 条。
            run.other_missing = True
            run.failures = failures + [
                (f"{other.key}/", f"对照项目资料目录不存在：{other.folder}")
            ]
            run.after_other_count = run.rerun_count
            run.total_count = store.count()
        else:
            other_records, other_failures = load_materials(other.folder, other.project_id)
            if other_records:
                store.upsert_documents(other_records)
            run.other_records = len(other_records)
            run.failures = failures + [
                (f"{other.key}/{name}", reason) for name, reason in other_failures
            ]
            run.after_other_count = store.count(project.project_id)
            run.total_count = store.count()
        run.other_project = other
    else:
        run.after_other_count = run.rerun_count
        run.total_count = store.count()
    return run


# ------------------------------------------------------------------ 渲染


def render_header(run: ParseRun) -> str:
    return "\n".join(
        [
            f"项目：{run.project.key} → {run.project.project_id}（{run.project.label}）",
            f"资料目录：{_relative(run.project.folder)}",
            f"数据库目录：{_relative(run.database_path)}",
            f"集合：{run.collection_name}",
        ]
    )


def render_units(run: ParseRun) -> str:
    return render_parse_units(run.records)


def render_inventory(run: ParseRun) -> str:
    return render_parse_inventory(run.records, run.failures)


def render_ingest(run: ParseRun) -> str:
    lines = [
        "真实向量化 + 入库 + 回读（17–27 分钟）",
        f"入库前（本项目）：{run.before_count} 条；入库后（本项目）：{run.after_count} 条",
        f"本次解析 {len(run.records)} 条"
        + (f"（完成标准要求 {EXPECTED_RECORDS} 条：一致）"
           if len(run.records) == EXPECTED_RECORDS else
           f"（完成标准要求 {EXPECTED_RECORDS} 条：不一致，请检查资料目录）"),
    ]
    for item in run.reread:
        record: KnowledgeRecord = item["record"]
        kind = "第 1 页" if record.file_type == "pdf" else "第 2 行"
        lines.append("")
        lines.append(f"回读 {record.file_type.upper()} {kind}：{item['id']}")
        lines.append(f"  元数据：{item['metadata']}")
        lines.append(f"  正文：{_preview(item['text'])}")
        lines.append(render_record_explanation(record))
    return "\n".join(lines)


def render_rerun(run: ParseRun) -> str:
    lines = [
        "重跑与切换项目（27–35 分钟）",
        f"第二次入库后（本项目）：{run.rerun_count} 条"
        f"（与第一次一致：{run.stable}）",
        "  稳定 ID = 项目 + 文件名 + 位置类型 + 位置；再次解析同一个文件 ID 不变，",
        "  所以 upsert 是更新同一条，不会追加第二条——这就是「重跑不重复」。",
    ]
    if run.other_project is not None and run.other_missing:
        lines.extend(
            [
                "",
                f"对照项目（{run.other_project.key} → {run.other_project.project_id}）"
                f"的资料目录不在仓库里：{run.other_project.folder}",
                "已跳过跨项目对照，不伪造另一组资料；库内总数仍为本项目自己的 "
                f"{run.total_count} 条。",
                "想恢复“两个项目共 14 条”的对照，把该目录放回仓库里再跑一次即可。",
            ]
        )
    elif run.other_project is not None:
        lines.extend(
            [
                "",
                f"再解析并入库另一个项目（{run.other_project.key} → "
                f"{run.other_project.project_id}）：{run.other_records} 条",
                f"此时本项目条数：{run.after_other_count} 条；库内总数：{run.total_count} 条"
                + (f"（完成标准要求 {EXPECTED_TOTAL} 条：一致）"
                   if run.total_count == EXPECTED_TOTAL else
                   f"（完成标准要求 {EXPECTED_TOTAL} 条：不一致）"),
                "",
                "为什么另一个项目的数据不计入本项目数量：",
                "  1. 每条记录的 project_id 不同，稳定 ID 也带项目前缀，根本不是同一条；",
                "  2. 计数与检索都用 where project_id 过滤，另一个项目的记录被挡在条件外；",
                "  3. 隔离是**靠元数据查询时做的**，不是靠分库做的——两个项目就写在同一个集合里。",
            ]
        )
    return "\n".join(lines)


def render_run(run: ParseRun) -> str:
    """一次演练的完整输出，与 main.py 的 /parse 逐字一致。"""
    blocks = [
        "=== 单元12：文档解析与知识库 ===",
        render_header(run),
        "",
        render_units(run),
        "",
        render_inventory(run),
        "",
        render_ingest(run),
        "",
        render_rerun(run),
    ]
    return "\n".join(blocks)


def _relative(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(MATERIALS_DIR.parents[1]).as_posix()
    except ValueError:
        return str(path)


def _preview(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "……"
