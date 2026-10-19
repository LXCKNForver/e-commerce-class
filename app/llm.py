from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass
class MockLLM:
    """课堂离线模型，保留未来替换真实 API 的统一接口。"""

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            raise ValueError("messages 不能为空")
        prompt = messages[-1]["content"]
        return f"这是离线 Mock 回答。已收到你的问题：{prompt}"


def build_llm():
    """根据环境变量选择模型，默认使用 Mock。"""
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "deepseek":
        from .deepseek import DeepSeekLLM

        return DeepSeekLLM()
    if provider != "mock":
        raise ValueError("LLM_PROVIDER 只能是 mock 或 deepseek")
    return MockLLM()
