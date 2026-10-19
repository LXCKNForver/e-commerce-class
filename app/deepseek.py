from __future__ import annotations

import os


class DeepSeekConfigError(RuntimeError):
    pass


class DeepSeekLLM:
    """可选的真实模型客户端，仅在 DeepSeek 模式下导入 openai。"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise DeepSeekConfigError(
                "请先设置 DEEPSEEK_API_KEY，或将 LLM_PROVIDER 改为 mock"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise DeepSeekConfigError(
                "DeepSeek 模式需要先按 README 安装 requirements-deepseek.txt"
            ) from exc

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            raise ValueError("messages 不能为空")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content or ""
