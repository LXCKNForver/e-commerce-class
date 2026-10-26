"""单元13：最小监控落库——每次问答的可复核证据，全部存在本地 SQLite。

为什么不只打印到控制台：评估要能**课后复核**。哪一题答得差、当时检索到了
哪几条证据、花了多少秒、用户标记了什么反馈，都必须在会话结束后还能查出来。
所以一题一行写进 `rag_logs` 表，反馈与汇总也从表里算，不从内存里算。

    uv run python -c "from app.monitoring import LogStore; from pathlib import Path; print(LogStore(Path('rag_monitor.sqlite3')).summary())"
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# 每行 rag_logs 的固定列：谁在什么时候问了什么、检索到什么、答了什么、
# 两类指标各多少分、耗时几秒、用户反馈是什么。评估闭环的证据全在这十列里。
SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    case_id TEXT NOT NULL,
    query TEXT NOT NULL,
    retrieved_json TEXT NOT NULL,
    answer TEXT NOT NULL,
    precision_at_3 REAL NOT NULL,
    recall_at_3 REAL NOT NULL,
    ragas_json TEXT NOT NULL,
    latency_seconds REAL NOT NULL,
    user_feedback TEXT
)
"""


@contextmanager
def _connect(path: Path) -> Iterator[sqlite3.Connection]:
    """打开连接并在退出时**一定关闭**句柄。

    `with sqlite3.connect(...)` 只管提交事务、**不会关闭连接**；Windows 上
    句柄不释放，紧接着删临时目录就会 PermissionError（测试清场会失败）。
    """
    conn = sqlite3.connect(path)
    try:
        with conn:  # 正常退出提交、异常退出回滚
            yield conn
    finally:
        conn.close()


class LogStore:
    """管理本单元的 SQLite 评估日志和用户反馈。"""

    def __init__(self, path: Path):
        """输入数据库文件路径；日志表不存在则当场创建。"""
        self.path = Path(path)
        with _connect(self.path) as conn:
            conn.execute(SCHEMA)

    def record(
        self,
        case_id: str,
        query: str,
        hits: list[dict],
        answer: str,
        precision: float,
        recall: float,
        ragas: dict,
        seconds: float,
    ) -> int:
        """保存一题的完整评估记录，返回新记录的整数 ID。

        检索命中（原文、来源、距离）整包存进 retrieved_json，
        RAGAS 三个分数存进 ragas_json——指标失败时如实存 error，不写假分数。
        """
        with _connect(self.path) as conn:
            cursor = conn.execute(
                """INSERT INTO rag_logs
                (case_id, query, retrieved_json, answer, precision_at_3, recall_at_3,
                 ragas_json, latency_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    case_id,
                    query,
                    json.dumps(hits, ensure_ascii=False),
                    answer,
                    precision,
                    recall,
                    json.dumps(ragas, ensure_ascii=False),
                    seconds,
                ),
            )
            return int(cursor.lastrowid)

    def set_feedback(self, log_id: int, feedback: str) -> None:
        """为指定日志记录写入用户反馈；只接受 good 或 bad，ID 不存在就报错。"""
        if feedback not in {"good", "bad"}:
            raise ValueError(f"反馈只能是 good 或 bad，收到：{feedback}")
        with _connect(self.path) as conn:
            cursor = conn.execute(
                "UPDATE rag_logs SET user_feedback=? WHERE id=?", (feedback, log_id)
            )
            if cursor.rowcount != 1:
                raise KeyError(f"日志ID不存在：{log_id}")

    def rows(self, limit: int = 5) -> list[dict]:
        """按写入倒序取最近的几行，渲染日志表截图内容时用。"""
        with _connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM rag_logs ORDER BY id DESC LIMIT ?", (limit,)
                )
            ]

    def summary(self) -> dict:
        """汇总已落库的评估结果，避免只凭单题判断系统质量。

        人工指标（precision/recall）直接对列求均值；RAGAS 三个分数存在
        ragas_json 里，逐行取出来过滤 None（评分失败的行不进均值）。
        尚无记录时均值均为 None，反馈计数为空字典。
        """
        with _connect(self.path) as conn:
            count, precision, recall, latency = conn.execute(
                """SELECT COUNT(*), AVG(precision_at_3), AVG(recall_at_3),
                          AVG(latency_seconds) FROM rag_logs"""
            ).fetchone()
            feedback = dict(
                conn.execute(
                    """SELECT user_feedback, COUNT(*) FROM rag_logs
                    WHERE user_feedback IS NOT NULL GROUP BY user_feedback"""
                ).fetchall()
            )
            ragas_values = [json.loads(row[0]) for row in conn.execute("SELECT ragas_json FROM rag_logs")]

        def ragas_average(key: str) -> float | None:
            values = [item[key] for item in ragas_values if isinstance(item, dict) and isinstance(item.get(key), (int, float))]
            return round(sum(values) / len(values), 3) if values else None

        return {
            "questions": count,
            "avg_precision_at_3": round(precision, 3) if precision is not None else None,
            "avg_recall_at_3": round(recall, 3) if recall is not None else None,
            "avg_context_precision": ragas_average("context_precision"),
            "avg_context_recall": ragas_average("context_recall"),
            "avg_faithfulness": ragas_average("faithfulness"),
            "avg_latency_seconds": round(latency, 2) if latency is not None else None,
            "feedback": feedback,
        }


def render_rows(rows: list[dict]) -> str:
    """把 rag_logs 的最近几行渲染成可贴进验收证据的文本。"""
    lines = [f"rag_logs 最近 {len(rows)} 条（按日志ID倒序）："]
    for row in rows:
        hits = json.loads(row["retrieved_json"])
        ragas = json.loads(row["ragas_json"])
        lines.append(
            f"  日志ID={row['id']}  题目={row['case_id']}  时间={row['created_at']}"
        )
        lines.append(
            f"    Precision@3={row['precision_at_3']:.2f}  Recall@3={row['recall_at_3']:.2f}"
            f"  耗时={row['latency_seconds']:.2f}秒  反馈={row['user_feedback'] or '未标记'}"
        )
        lines.append(f"    检索来源：{[hit['source'] for hit in hits]}")
        lines.append(f"    RAGAS：{ragas}")
    return "\n".join(lines)


def render_summary(summary: dict) -> str:
    """把 summary() 的结果渲染成可贴进验收证据的文本。"""
    return "\n".join(
        [
            "日志汇总（来自 rag_logs 表）",
            f"  已评估题数：{summary['questions']}",
            f"  平均 Precision@3：{summary['avg_precision_at_3']}",
            f"  平均 Recall@3：{summary['avg_recall_at_3']}",
            f"  平均 RAGAS Context Precision：{summary['avg_context_precision']}",
            f"  平均 RAGAS Context Recall：{summary['avg_context_recall']}",
            f"  平均 RAGAS Faithfulness：{summary['avg_faithfulness']}",
            f"  平均耗时：{summary['avg_latency_seconds']} 秒",
            f"  用户反馈：{summary['feedback'] or '暂无'}",
        ]
    )
