from __future__ import annotations

from collections.abc import Iterator
import os


class DeepSeekConfigError(RuntimeError):
    pass


class DeepSeekLLM:
    """可选的真实模型客户端；只有选择 deepseek 模式时才导入 openai。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "https://api.deepseek.com",
    ):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise DeepSeekConfigError(
                "请先设置DEEPSEEK_API_KEY，或使用默认的Mock模式"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DeepSeekConfigError(
                "DeepSeek模式需要安装openai：pip install -r requirements-deepseek.txt"
            ) from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def chat(self, messages: list[dict[str, str]]) -> str:
        return "".join(self.stream(messages))

    def complete(self, messages: list[dict], tools: list[dict] | None = None):
        """非流式调用；传入 tools 时启用 Function Calling，返回带 tool_calls 的消息对象。"""
        if not messages:
            raise ValueError("messages不能为空")

        kwargs: dict = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not messages:
            raise ValueError("messages不能为空")

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content