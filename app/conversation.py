"""单元02：会话对象。

职责单一：只负责保存、裁剪和读取 user/assistant 消息，
不关心模型怎么调用、结果怎么显示，便于在第 06—14 单元接入 RAG 后仍然复用。

裁剪策略：history 超过 max_messages 条时丢掉最早的一轮问答，
保证多轮上下文可用，同时不会让请求体无限增长。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Conversation:
    history: list[dict[str, str]] = field(default_factory=list)
    max_messages: int = 20

    def add(self, role: str, content: str) -> None:
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError("角色不合法")
        if not content.strip():
            raise ValueError("消息内容不能为空")
        self.history.append({"role": role, "content": content})
        self._trim()

    def add_pair(self, question: str, answer: str) -> None:
        """一次写入一整轮问答，保持 user -> assistant 的顺序。"""
        self.add("user", question)
        self.add("assistant", answer)

    def _trim(self) -> None:
        if self.max_messages <= 0 or len(self.history) <= self.max_messages:
            return
        overflow = len(self.history) - self.max_messages
        # 成对丢弃，避免把某轮问答切成半个上下文。
        overflow += overflow % 2
        self.history = self.history[overflow:]

    @property
    def turns(self) -> int:
        """已完成的问答轮数。"""
        return sum(1 for item in self.history if item["role"] == "user")

    @property
    def last_user_message(self) -> str | None:
        for item in reversed(self.history):
            if item["role"] == "user":
                return item["content"]
        return None

    def clear(self) -> None:
        self.history.clear()

    def render(self) -> str:
        """把历史渲染成课堂截图友好的文本。"""
        if not self.history:
            return "当前没有历史消息。"
        return "\n".join(
            f"{index}. {item['role']}: {item['content']}"
            for index, item in enumerate(self.history, start=1)
        )

    def replace_history(self, messages: list[dict[str, Any]]) -> None:
        """用外部消息列表覆盖当前 history（工具闭环会产生 tool 消息）。"""
        self.history = list(messages)
