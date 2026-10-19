import os
import unittest
from unittest.mock import patch

from app.llm import MockLLM, build_llm


class TestUnit01(unittest.TestCase):
    def test_mock_llm_returns_answer(self):
        answer = MockLLM().chat([{"role": "user", "content": "测试问题"}])
        self.assertIn("测试问题", answer)

    def test_default_provider_is_mock(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(build_llm(), MockLLM)

    def test_empty_messages_are_rejected(self):
        with self.assertRaises(ValueError):
            MockLLM().chat([])


if __name__ == "__main__":
    unittest.main()
