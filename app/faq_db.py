"""单元15：把本组 FAQ CSV 同步成"结构化记录"，供只读查询工具按编号精确命中。

单元12 做过一次"多格式 → 同一种知识记录"：那时 CSV 的每一行也被切进向量库，
和 Markdown、PDF 一样靠**相似度**被找到。本单元把它换一种装法——同一份原始 CSV
再同步一份进 SQLite，按**编号**精确查：

    原始资料（常见售后问题.csv）
        ├── 非结构化：逐行进 Chroma，靠向量相似度"模糊"命中（app/loaders.py）
        └── 结构化  ：整表进 SQLite，靠 question_id "精确"命中（本模块）

所以本模块刻意**不改原始资料**：只从 CSV 读出内容，写进本单元自己目录下的
SQLite。切换项目组时清空本表重灌，避免上一组遗留的编号被下一组查到。

工具只被允许执行固定的参数化 SQL（`SELECT ... WHERE question_id = ?`），
问号只传值，模型拿不到任何改写语句的入口；连库还用只读 URI 打开，
从根上堵住"顺手改一条记录"。

    uv run python -c "from app.faq_db import prepare_faq_db as p; print(p(__import__('pathlib').Path('模拟数据/单元12项目资料/电商售后服务智能体项目组/常见售后问题.csv'), __import__('pathlib').Path('.sqlite_unit15/faq.db')))"
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

# 表名固定，DELETE 只针对这张表——别的地方万一也有表，不受影响。
FAQ_TABLE = "faq"
# CSV 表头必须有这四列，缺一列就直接报错：宁可课堂上看一眼错在哪，
# 也不要让工具查出一个"有编号没内容"的空答复。
REQUIRED_FIELDS = ("问题编号", "问题", "标准答复", "来源编号")
# 固定的只读查询：SELECT 字段写死、WHERE 用问号占位，没有任何拼接。
SELECT_BY_ID = f"SELECT question_id, question, answer, source_id FROM {FAQ_TABLE} WHERE question_id = ?"


def read_faq_rows(faq_file: Path) -> list[dict[str, str]]:
    """读出 CSV 全部行；字段缺失直接报错，不做"缺哪一列就填空"的静默处理。"""
    if not Path(faq_file).is_file():
        raise FileNotFoundError(f"FAQ 文件不存在：{faq_file}")
    with Path(faq_file).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not set(REQUIRED_FIELDS).issubset(rows[0]):
        raise ValueError(
            f"{Path(faq_file).name} 缺少 FAQ 必需字段：需要 {list(REQUIRED_FIELDS)}"
        )
    return rows


def prepare_faq_db(faq_file: Path, db_path: Path) -> Path:
    """把本组 FAQ 同步到 SQLite，返回建好的库路径（供只读工具打开）。

    这里**必须**显式 `conn.close()`：Windows 上句柄不释放，紧接着删临时目录
    会 PermissionError（单元13 的日志库踩过同一个坑）。
    """
    rows = read_faq_rows(faq_file)
    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    try:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {FAQ_TABLE} (
                question_id TEXT PRIMARY KEY,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                source_id   TEXT NOT NULL
            )
            """
        )
        connection.execute(f"DELETE FROM {FAQ_TABLE}")
        connection.executemany(
            f"""
            INSERT INTO {FAQ_TABLE} (question_id, question, answer, source_id)
            VALUES (:问题编号, :问题, :标准答复, :来源编号)
            ON CONFLICT(question_id) DO UPDATE SET
                question=excluded.question,
                answer=excluded.answer,
                source_id=excluded.source_id
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()
    return target


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """以只读模式打开库；调用方负责关闭（本文件里都是 `finally: conn.close()`）。"""
    target = Path(db_path)
    if not target.is_file():
        raise FileNotFoundError(f"SQLite 库不存在：{target}")
    return sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)


def fetch_faq(db_path: Path, question_id: str) -> dict | None:
    """按编号取一行；没有就返回 None，让工具如实回答"未找到"。"""
    connection = connect_readonly(db_path)
    try:
        connection.row_factory = sqlite3.Row
        row = connection.execute(SELECT_BY_ID, (question_id,)).fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()


def count_faq(db_path: Path) -> int:
    """库里现有 FAQ 条数：切换项目组时用来确认"只剩本组这些"。"""
    connection = connect_readonly(db_path)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {FAQ_TABLE}").fetchone()[0])
    finally:
        connection.close()
