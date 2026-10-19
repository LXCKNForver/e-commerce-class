import unittest

from app.data_store import OrderRepository
from app.tools import query_order, check_return_eligibility


class TestOrderDataConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = OrderRepository()

    def test_so2001_matches_records(self):
        order = self.repo.get_order("SO-2001")
        self.assertIsNotNone(order)
        self.assertEqual(order["order_status"], "已签收")
        self.assertEqual(order["logistics_status"], "已签收")
        self.assertEqual(order["delivered_on"], "2026-10-12")
        self.assertFalse(order["used_or_opened"])

    def test_unknown_order_returns_none(self):
        self.assertIsNone(self.repo.get_order("SO-9999"))

    def test_query_order_tool(self):
        result = query_order("SO-2001")
        self.assertTrue(result["ok"])
        self.assertEqual(result["order"]["order_id"], "SO-2001")

    def test_custom_item_not_normal_return(self):
        result = check_return_eligibility("SO-2004", "质量问题")
        self.assertTrue(result["ok"])
        self.assertFalse(result["eligible"])
        self.assertTrue(result["manual"])


if __name__ == "__main__":
    unittest.main()