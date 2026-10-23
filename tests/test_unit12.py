"""单元12 验收测试：Markdown / PDF / CSV → 同一种知识记录 → 入库回读 → 重跑稳定。

解析层的用例全部用临时目录里的**自造小文件**，不依赖课程资料，也永远运行；
PDF 用例需要 pypdf（`uv sync --extra parse`），缺了就跳过并说明原因。
真实入库的用例需要 embedding + chroma，缺依赖时同样跳过——不判失败。
调用 DeepSeek 的地方一处都没有：本单元从头到尾不碰模型。
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from app.loaders import (
    KnowledgeRecord,
    MATERIALS_DIR,
    load_materials,
    parse_file,
    render_parse_inventory,
    render_parse_units,
)
from app.parse_lab import (
    DEFAULT_PROJECT,
    EXPECTED_RECORDS,
    EXPECTED_TOTAL,
    PROJECTS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _pdf_available() -> bool:
    try:
        import pypdf  # noqa: F401
    except ImportError:
        return False
    return True


def _chroma_available() -> bool:
    try:
        import chromadb  # noqa: F401
        import fastembed  # noqa: F401
    except ImportError:
        return False
    return True


PDF_AVAILABLE = _pdf_available()
CHROMA_AVAILABLE = _chroma_available()


def write_markdown(folder: Path, name: str = "DOC-09-01-测试规则.md") -> Path:
    path = folder / name
    path.write_text("# 测试规则\n\n签收后 7 天内可申请退货评估。\n", encoding="utf-8")
    return path


def write_csv(folder: Path, name: str = "常见问题.csv") -> Path:
    path = folder / name
    path.write_text(
        "问题编号,问题,标准答复\n"
        "T-01,能退货吗,7 天内未使用可申请\n"
        "T-02,运费谁承担,按责任方判定\n",
        encoding="utf-8-sig",
    )
    return path


class ParseFileTest(unittest.TestCase):
    """三种格式各自的解析单位：文档 / 页 / 行。"""

    def setUp(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="parse-unit12-"))

    def tearDown(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_markdown_one_record_per_document(self) -> None:
        path = write_markdown(self._dir)
        records = parse_file(path, "e-commerce-after-sales")
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.location_type, "document")
        self.assertEqual(record.location, 1)
        self.assertEqual(record.file_type, "md")
        self.assertTrue(record.text.startswith("# 测试规则"))

    def test_ids_are_stable_across_runs(self) -> None:
        path = write_markdown(self._dir)
        first = parse_file(path, "e-commerce-after-sales")
        second = parse_file(path, "e-commerce-after-sales")
        self.assertEqual([r.document_id for r in first], [r.document_id for r in second])
        self.assertEqual(
            first[0].document_id,
            "e-commerce-after-sales:DOC-09-01-测试规则.md:document:1",
        )

    def test_project_prefix_separates_two_projects(self) -> None:
        path = write_markdown(self._dir)
        mine = parse_file(path, "e-commerce-after-sales")[0]
        other = parse_file(path, "travel-expense-demo")[0]
        self.assertNotEqual(mine.document_id, other.document_id)
        self.assertTrue(mine.document_id.startswith("e-commerce-after-sales:"))

    def test_csv_one_record_per_data_row_starting_at_two(self) -> None:
        records = parse_file(write_csv(self._dir), "e-commerce-after-sales")
        self.assertEqual(len(records), 2)
        self.assertEqual([r.location for r in records], [2, 3])
        self.assertTrue(all(r.location_type == "row" for r in records))
        # 表头不算记录，但字段名要进正文，否则单独一行数字没有语义。
        self.assertIn("问题：能退货吗", records[0].text)
        self.assertIn("标准答复：7 天内未使用可申请", records[0].text)

    def test_csv_column_mismatch_is_reported(self) -> None:
        path = self._dir / "坏表.csv"
        path.write_text("编号,问题\n1,能退货吗,多出来的一列\n", encoding="utf-8-sig")
        with self.assertRaises(ValueError):
            parse_file(path, "e-commerce-after-sales")

    def test_csv_header_only_is_reported(self) -> None:
        path = self._dir / "只有表头.csv"
        path.write_text("编号,问题\n", encoding="utf-8-sig")
        with self.assertRaises(ValueError):
            parse_file(path, "e-commerce-after-sales")

    def test_empty_markdown_is_reported(self) -> None:
        path = self._dir / "空.md"
        path.write_text("   \n\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            parse_file(path, "e-commerce-after-sales")

    def test_unsupported_suffix_is_reported(self) -> None:
        path = self._dir / "说明.txt"
        path.write_text("纯文本", encoding="utf-8")
        with self.assertRaises(ValueError):
            parse_file(path, "e-commerce-after-sales")

    def test_empty_project_id_is_rejected(self) -> None:
        path = write_markdown(self._dir)
        with self.assertRaises(ValueError):
            parse_file(path, "  ")

    @unittest.skipUnless(PDF_AVAILABLE, "未安装 pypdf，跳过 PDF 用例（uv sync --extra parse）")
    def test_pdf_one_record_per_page(self) -> None:
        path = MATERIALS_DIR / "电商售后服务智能体项目组" / "售后受理材料清单.pdf"
        if not path.is_file():
            self.skipTest(f"缺少课程 PDF：{path}")
        records = parse_file(path, "e-commerce-after-sales")
        self.assertGreaterEqual(len(records), 1)
        self.assertEqual(records[0].location_type, "page")
        self.assertEqual(records[0].location, 1)
        self.assertEqual(records[0].file_type, "pdf")
        self.assertIn("售后受理材料清单", records[0].text)


class MetadataTest(unittest.TestCase):
    def test_metadata_carries_project_source_and_location(self) -> None:
        record = KnowledgeRecord(
            document_id="e-commerce-after-sales:常见售后问题.csv:row:2",
            text="问题：能退货吗",
            source="模拟数据/单元12项目资料/电商售后服务智能体项目组/常见售后问题.csv",
            file_type="csv",
            location_type="row",
            location=2,
            project_id="e-commerce-after-sales",
        )
        metadata = record.metadata
        self.assertEqual(metadata["project_id"], "e-commerce-after-sales")
        self.assertEqual(metadata["location_type"], "row")
        self.assertEqual(metadata["location"], 2)
        self.assertEqual(metadata["doc_type"], "csv")
        self.assertIn("常见售后问题.csv", metadata["source"])


class LoadMaterialsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="materials-unit12-"))

    def tearDown(self) -> None:
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_bad_file_does_not_block_good_ones(self) -> None:
        write_markdown(self._dir)
        (self._dir / "说明.txt").write_text("不是资料", encoding="utf-8")
        records, failures = load_materials(self._dir, "e-commerce-after-sales")
        self.assertEqual(len(records), 1)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], "说明.txt")

    def test_missing_folder_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_materials(self._dir / "不存在", "e-commerce-after-sales")

    def test_inventory_and_unit_table_render(self) -> None:
        write_markdown(self._dir)
        records, _ = load_materials(self._dir, "e-commerce-after-sales")
        inventory = render_parse_inventory(records, [])
        self.assertIn("共 1 条", inventory)
        self.assertIn("document=1", inventory)
        self.assertIn("DOC-09-01-测试规则.md", inventory)
        units = render_parse_units(records)
        self.assertIn("文档（整篇）", units)
        self.assertIn(".md 1 条", units)


class ProjectSpecTest(unittest.TestCase):
    def test_default_project_is_our_own(self) -> None:
        from app.parse_lab import resolve_project

        self.assertEqual(DEFAULT_PROJECT, "ecom")
        self.assertEqual(resolve_project(None).project_id, "e-commerce-after-sales")
        self.assertEqual(resolve_project("ECOM").project_id, "e-commerce-after-sales")

    def test_unknown_project_lists_choices(self) -> None:
        from app.parse_lab import resolve_project

        with self.assertRaises(ValueError) as ctx:
            resolve_project("finance")
        self.assertIn("ecom", str(ctx.exception))
        self.assertIn("travel", str(ctx.exception))

    def test_two_projects_have_distinct_ids(self) -> None:
        ids = {spec.project_id for spec in PROJECTS.values()}
        self.assertEqual(len(ids), len(PROJECTS))


class RealMaterialsTest(unittest.TestCase):
    """课程资料本身的解析结果：所选项目 7 条，两个项目合计 14 条。"""

    def test_each_project_parses_seven_records(self) -> None:
        for key, spec in PROJECTS.items():
            if not spec.folder.is_dir():
                self.skipTest(f"缺少资料目录：{spec.folder}")
        for key, spec in PROJECTS.items():
            records, failures = load_materials(spec.folder, spec.project_id)
            self.assertEqual(failures, [], f"{key} 出现解析失败：{failures}")
            self.assertEqual(len(records), EXPECTED_RECORDS, f"{key} 应解析 7 条")

    @unittest.skipUnless(PDF_AVAILABLE, "未安装 pypdf，跳过 PDF 用例")
    def test_file_types_cover_md_pdf_csv(self) -> None:
        spec = PROJECTS[DEFAULT_PROJECT]
        records, _ = load_materials(spec.folder, spec.project_id)
        self.assertEqual({r.file_type for r in records}, {"md", "pdf", "csv"})


@unittest.skipUnless(CHROMA_AVAILABLE, "未安装 chromadb / fastembed，跳过真实入库用例")
class ChromaIntegrationTest(unittest.TestCase):
    """真实入库：7 条、回读两条、重跑不重复、换项目后总数 14 条。"""

    @classmethod
    def setUpClass(cls) -> None:
        from app.parse_lab import run_parse

        cls._dir = Path(tempfile.mkdtemp(prefix="chroma-unit12-cls-"))
        cls.run_result = run_parse(DEFAULT_PROJECT, cls._dir, with_other_project=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._dir, ignore_errors=True)

    def test_seven_records_written(self) -> None:
        self.assertEqual(self.run_result.before_count, 0)
        self.assertEqual(self.run_result.after_count, EXPECTED_RECORDS)

    def test_rerun_does_not_grow(self) -> None:
        self.assertTrue(self.run_result.stable)
        self.assertEqual(self.run_result.rerun_count, EXPECTED_RECORDS)

    def test_reread_pdf_page_and_csv_row(self) -> None:
        kinds = [item["record"].file_type for item in self.run_result.reread]
        self.assertEqual(kinds, ["pdf", "csv"])
        pdf_item = self.run_result.reread[0]
        self.assertEqual(pdf_item["metadata"]["location_type"], "page")
        self.assertEqual(pdf_item["metadata"]["location"], 1)
        self.assertIn("售后受理材料清单", pdf_item["text"])
        csv_item = self.run_result.reread[1]
        self.assertEqual(csv_item["metadata"]["location_type"], "row")
        self.assertEqual(csv_item["metadata"]["location"], 2)
        self.assertIn("EC-FAQ-01", csv_item["text"])

    def test_other_project_does_not_count_into_mine(self) -> None:
        run = self.run_result
        self.assertEqual(run.after_other_count, EXPECTED_RECORDS)
        self.assertEqual(run.total_count, EXPECTED_TOTAL)
        self.assertNotEqual(run.other_project.project_id, run.project.project_id)

    def test_render_contains_the_required_numbers(self) -> None:
        from app.parse_lab import render_run

        text = render_run(self.run_result)
        self.assertIn("入库后（本项目）：7 条", text)
        self.assertIn("库内总数：14 条", text)
        self.assertIn("page=1", text)
        self.assertIn("row=2", text)


if __name__ == "__main__":
    unittest.main()
