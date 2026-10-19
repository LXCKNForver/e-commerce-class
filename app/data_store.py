# app/data_store.py
"""只读订单数据仓库，连接仓库自带的 模拟数据/records.json。

仓库里 `模拟数据/` 是课程提供的虚构数据，`验证数据.py` 会对它做校验，
因此这里是唯一数据源，不另外维护第二份订单表。

单元03 任务卡要求"真实 CSV 数据调用入口"，所以同时提供 `export_csv()`：
把同一份订单导出成 CSV 供 Prompt 对比实验使用，两种格式永远来自同一份 JSON，
不会出现两套数据算出不同结论的情况。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "模拟数据"
RECORDS_JSON = DATA_DIR / "records.json"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"

# CSV 列顺序固定，导出与回读都用它，避免列漂移。
CSV_FIELDS = [
    "order_id",
    "customer_id",
    "item_category",
    "item_name",
    "order_status",
    "logistics_status",
    "delivered_on",
    "used_or_opened",
    "invoice_status",
    "amount_yuan",
    "after_sale_status",
]


class OrderRepository:
    """只读订单数据仓库。

    数据全部来自课程虚构样例，任何工具都只读不写。
    """

    def __init__(self, records_path: Path | None = None):
        self.records_path = records_path or RECORDS_JSON
        if not self.records_path.is_file():
            raise FileNotFoundError(f"未找到订单数据文件：{self.records_path}")

        payload = json.loads(self.records_path.read_text(encoding="utf-8"))
        data = payload["data"]

        # as_of 是算签收天数的基准日，必须取自数据文件而不是写死在代码里。
        self.as_of: str = data["as_of"]
        self.knowledge_count: int = payload.get("knowledge_count", 0)
        self.fictional: bool = payload.get("fictional", False)

        self.orders: dict[str, dict[str, Any]] = {}
        for row in data["orders"]:
            row = dict(row)
            row["used_or_opened"] = bool(row["used_or_opened"])
            row["amount_yuan"] = int(row["amount_yuan"])
            self.orders[row["order_id"]] = row

    def get_order(self, order_id: str) -> dict[str, Any] | None:
        return self.orders.get(order_id)

    def all_orders(self) -> list[dict[str, Any]]:
        return list(self.orders.values())

    def export_csv(self, target: Path | None = None) -> Path:
        """导出为 CSV（单元03 的"真实 CSV 调用入口"用同一份数据）。"""
        target = target or (DATA_DIR / "records.csv")
        with open(target, mode="w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for order in self.orders.values():
                writer.writerow(order)
        return target

    def load_csv(self, source: Path) -> list[dict[str, Any]]:
        """从 CSV 回读订单，用于核对导出结果与 JSON 是否一致。"""
        rows: list[dict[str, Any]] = []
        with open(source, mode="r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                row["used_or_opened"] = row["used_or_opened"].strip().lower() == "true"
                row["amount_yuan"] = int(row["amount_yuan"])
                row["delivered_on"] = row["delivered_on"] or None
                rows.append(row)
        return rows