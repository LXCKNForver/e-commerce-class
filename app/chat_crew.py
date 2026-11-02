"""单元22：AutoGen 实战 —— 三个角色在同一间**群聊**里轮流发言。

前几个单元的协作方式都是"一条流水线"：谁在什么时候说话由程序写死，上一环的产物
直接当下一环的输入。单元21 换成了黑板，但顺序仍是固定的五步。本单元换的是**群聊**：

    发起者（user，不调模型）提交「用户任务 + 平台工具证据」
        │
        ▼   GroupChat：消息按顺序追加，后发言的人能看到前面所有人的原话
    ① 规划员 planner  ── 拆步骤，只出规划
    ② 答复执行员 executor ── 写面向用户的答复
    ③ 评审员 reviewer ── 只出结论：{"passed": bool, "reason": str}
        通过 → 收尾，草稿成为正式答复
        不通过 → 群聊继续下一轮，执行员看到复核意见再改一版
                （直到轮次或成本用完；仍未通过就转人工）
        │
        ▼   GroupChatManager：决定下一位谁发言、什么时候停（自己不调模型）

四个完成标准的落点：

1. **至少两个角色参与**：三个角色 + 发起者；`Collaboration.participants` 是跑完后
   实际发过言的角色（去重），不是配置里写了几个。
2. **消息内容可查看**：每条消息都留发送者、正文、估算成本与"第几位发言"，
   `render_transcript()` 全文打印，`GroupChat.messages` 就是这次协作的完整记录。
3. **轮次和成本受控**：`GroupChat.max_round` 是**消息条数**硬上限（含发起者那条），
   `token_budget` 是成本上限（估算 token，超了就不再让下一个角色开口）。
   两条上限都是 Python 判断，不靠模型自觉；触发时状态是"未交付"，不假装答完了。
4. **结果由复核角色确认**：执行员写的是**草稿**，只有评审员给出 `passed=True`，
   `Collaboration.answer` 才有值。复核没轮到、复核不通过，答复一律为空。

关于 AutoGen：任务卡写的是"用 AutoGen 或**等价轻量实现**"。AutoGen 0.2 的
`ConversableAgent` / `GroupChat` / `GroupChatManager` 三个对象在本模块里一一对应
（`ChatAgent` / `GroupChat` / `GroupChatManager`），行为也一样：轮流发言、消息累积、
轮次上限要额外计入发起者那条消息、群聊管理器本身不调模型。差别只在于：
**没有引入 pyautogen 这个依赖**（它会带进 openai/docker/flaml 一整套），因此课堂上
不必为一个演示新增依赖，也不会把 `langchain-core 1.7` 的依赖树搞乱。取舍写在这里，
而不是让读者以为"这就是 AutoGen 本体"。

    uv run --extra embedding --extra chroma --extra langchain python -m app.chat_lab
"""

from __future__ import annotations

import json
import math
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.plan_solver import build_plan_model, validate_goal
from app.toolkit import TOOL_FAQ, TOOL_SEARCH

# ---------------------------------------------------------------- ① 常量

ROLE_PLANNER = "planner"
ROLE_EXECUTOR = "executor"
ROLE_REVIEWER = "reviewer"
# 发起者只提交任务与证据，自己不发言、不调模型——它不是第四个"智能体"。
ROLE_INITIATOR = "user"

ROLE_ORDER = (ROLE_PLANNER, ROLE_EXECUTOR, ROLE_REVIEWER)

# 角色标记写在 system 消息的最前面：离线剧本模型靠它认出"此刻轮到谁"。
# 用「【角色：xx】」这种整段标记，而不是裸的"规划员"——群聊记录里难免出现这些词，
# 只按关键词匹配会把别人的发言当成自己的角色。
TAG_PLANNER = "【角色：规划员】"
TAG_EXECUTOR = "【角色：答复执行员】"
TAG_REVIEWER = "【角色：评审员】"
ROLE_TAGS: dict[str, str] = {
    ROLE_PLANNER: TAG_PLANNER,
    ROLE_EXECUTOR: TAG_EXECUTOR,
    ROLE_REVIEWER: TAG_REVIEWER,
}

# 轮次：任务卡要求"轮次受控"。3 个角色各发言一次 = 3 轮，再往上要显式申请。
DEFAULT_ROUNDS = 3
MAX_ROUNDS = 6
# 成本：任务卡要求"成本受控"。按字符估算 token，不引 tokenizer（不为估算新增依赖）。
DEFAULT_TOKEN_BUDGET = 4000
CHARS_PER_TOKEN = 1.6

# 四种终止原因：一句话说清"为什么停"，不许合并成"结束了"。
STOP_REVIEWED = "reviewed"
STOP_REJECTED = "review_rejected"
STOP_MAX_ROUND = "max_round"
STOP_BUDGET = "token_budget"

STOP_NOTES: dict[str, str] = {
    STOP_REVIEWED: "复核角色已发言并通过：执行员的草稿成为正式答复",
    STOP_REJECTED: "复核角色最后一次发言不通过，且轮次已用完：答复留空，转人工复核",
    STOP_MAX_ROUND: "达到轮次上限：群聊停在 max_round 条消息，复核角色还没轮到",
    STOP_BUDGET: "达到成本上限：估算 token 已超预算，不再让下一位角色开口",
}

# 四种状态：交付 / 未交付（复核不通过）/ 未交付（轮次用完）/ 未交付（成本用完）。
STATUS_DELIVERED = "复核通过，已交付"
STATUS_REJECTED = "复核未通过，未交付（转人工复核）"
STATUS_MAX_ROUND = "达到轮次上限，复核未完成（未交付）"
STATUS_BUDGET = "达到成本上限，协作中止（未交付）"

STATUS_OF_STOP: dict[str, str] = {
    STOP_REVIEWED: STATUS_DELIVERED,
    STOP_REJECTED: STATUS_REJECTED,
    STOP_MAX_ROUND: STATUS_MAX_ROUND,
    STOP_BUDGET: STATUS_BUDGET,
}

# 提示词里两段固定文字：离线剧本模型据此切出证据 JSON。
EVIDENCE_START = "平台工具证据（Python 调用工具取出，不是模型生成）："
TRANSCRIPT_START = "群聊记录（同一间群聊，按发言顺序）："

OFFLINE_MODEL_NAME = "scripted-groupchat-model"
MESSAGE_PREVIEW = 160
TRANSCRIPT_WIDTH = 76

# 证据收集：编号紧贴中文时 `\b` 匹配不到，用前后否定环视（与单元15 的写法一致）。
FAQ_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9])EC-FAQ-\d{2}(?![A-Za-z0-9])")
# 短检索词比整句更不容易稀释相关性：按业务词挑一句固定查询。
KNOWLEDGE_QUERIES: tuple[tuple[str, str], ...] = (
    ("退货", "普通商品退货需要哪些材料，时限是多久"),
    ("质量问题", "商品有质量问题要提供什么材料"),
    ("物流", "物流轨迹长期没有更新怎么处理"),
)
FALLBACK_QUERY_CHARS = 40


class ChatProtocolError(ValueError):
    """某个角色没按约定输出（空内容 / 复核结论不是 JSON）：不猜、不降级，直接报错。"""


# ---------------------------------------------------------------- ② 角色契约


@dataclass(frozen=True)
class AgentSpec:
    """一个群聊角色的契约：职责、输入、输出、不许做什么。

    这张表是第1节要交的证据；"不许做什么"和"输出"一样重要——执行员不许自己放行，
    评审员不许代写答复。
    """

    role: str
    name: str
    duty: str
    inputs: str
    outputs: str
    forbidden: str
    tag: str


AGENTS: tuple[AgentSpec, ...] = (
    AgentSpec(
        ROLE_PLANNER,
        "规划员",
        "把用户任务拆成可核对的步骤，并指出执行时要覆盖哪几段",
        "用户任务 + 平台工具证据（群聊第 0 条）",
        "一段规划：先做什么、按什么顺序、要覆盖哪几段",
        "不写面向用户的答复、不调工具、不替评审员下结论",
        TAG_PLANNER,
    ),
    AgentSpec(
        ROLE_EXECUTOR,
        "答复执行员",
        "依据证据与规划写出面向用户的草稿",
        "用户任务 + 工具证据 + 规划员的发言",
        "一版答复草稿（**还没交付**）",
        "不编造证据里没有的时限或承诺、不自行放行",
        TAG_EXECUTOR,
    ),
    AgentSpec(
        ROLE_REVIEWER,
        "评审员",
        "独立核对草稿是否忠于证据、是否说明缺口",
        "用户任务 + 工具证据 + 前两位的发言",
        'JSON：{"passed": 布尔, "reason": 理由}',
        "不改写草稿、不代写答复、不因「差不多」放行",
        TAG_REVIEWER,
    ),
)

ROLE_NAMES: dict[str, str] = {spec.role: spec.name for spec in AGENTS}
ROLE_NAMES[ROLE_INITIATOR] = "发起者（用户）"


def role_name(role: str) -> str:
    """渲染与报错都用中文名，避免输出里一半英文键一半中文名。"""
    return ROLE_NAMES.get(role, role)


SYSTEM_PROMPTS: dict[str, str] = {
    ROLE_PLANNER: (
        f"{TAG_PLANNER}\n"
        "你是电商售后服务智能体群聊里的规划员。任务和工具证据在第一条消息里，"
        "证据由平台工具取出，不是模型生成的，不得改写或补充。"
        "只输出规划：分成哪几步、每步覆盖任务的哪一段、执行时要点名哪些编号与依据文件。"
        "不要写面向用户的答复，也不要替评审员下结论。"
    ),
    ROLE_EXECUTOR: (
        f"{TAG_EXECUTOR}\n"
        "你是电商售后服务智能体群聊里的答复执行员。依据工具证据与规划员的发言，"
        "写一段面向买家的答复：先给结论，再给依据（编号与文件名照抄证据里的原文），"
        "最后说明哪些结论有证据、哪些需要转人工。"
        "证据里没有的时限、金额、承诺一律不许写；查不到就明说资料不足。"
        "你写的是草稿，能不能交付由评审员决定。"
    ),
    ROLE_REVIEWER: (
        f"{TAG_REVIEWER}\n"
        "你是电商售后服务智能体群聊里的评审员。对照工具证据核对执行员的草稿："
        "是否回答了任务、是否只用了证据里的事实、是否说明了缺口、有没有编造时限或承诺。"
        "不得改写草稿，也不得代写答复。"
        '只输出 JSON：{"passed":布尔值,"reason":"理由"}。'
    ),
}


# ---------------------------------------------------------------- ③ 成本估算


def estimate_tokens(text: str) -> int:
    """按字符估算 token（中文约 1.6 字符/token）。

    刻意不引 tokenizer：课堂演示只需要一个**可核对、可复现**的成本口径，
    真值由服务商账单决定。估算规则写在这里，谁都能拿计算器复算。
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def validate_rounds(rounds: Any) -> int:
    """轮次守门：不合格就在让任何角色开口之前拦下（一次模型调用都不发）。"""
    if isinstance(rounds, bool) or not isinstance(rounds, int):
        raise ValueError(f"轮次必须是整数，实际是：{type(rounds).__name__}")
    if not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"轮次必须在 1 到 {MAX_ROUNDS} 之间，实际是：{rounds}")
    return rounds


def validate_budget(token_budget: Any) -> int:
    """成本守门：预算必须为正整数，否则等于没有上限。"""
    if isinstance(token_budget, bool) or not isinstance(token_budget, int):
        raise ValueError(f"token 预算必须是整数，实际是：{type(token_budget).__name__}")
    if token_budget <= 0:
        raise ValueError(f"token 预算必须大于 0，实际是：{token_budget}")
    return token_budget


# ---------------------------------------------------------------- ④ 证据（Python 调工具）


def extract_faq_ids(task: str) -> list[str]:
    """从任务里取 FAQ 编号（去重、保序）；没有编号就返回空列表。"""
    return list(dict.fromkeys(FAQ_ID_PATTERN.findall(task)))


def knowledge_query(task: str) -> str:
    """挑一句**短**检索词：整句任务会稀释相关性，命中反而更差。"""
    for word, query in KNOWLEDGE_QUERIES:
        if word in task:
            return query
    return " ".join(task.split())[:FALLBACK_QUERY_CHARS]


def collect_evidence(registry: Any, task: str) -> dict[str, Any]:
    """群聊开口之前，先用 Python 把事实取回来（模型不碰工具）。

    输入：单元15 的注册表 + 用户任务。
    输出：{"faq": [...], "knowledge": {...}, "tool_calls": n}。

    这么做的理由和 AutoGen 参考实现一致：**先由平台工具获取证据，再让模型发言**。
    群聊里的模型只负责措辞，不负责取数——它连工具都没有，就不可能编出"查过了"。
    """
    calls = 0
    faqs: list[dict[str, Any]] = []
    for question_id in extract_faq_ids(task):
        faqs.append(registry.call(TOOL_FAQ, {"question_id": question_id}))
        calls += 1
    knowledge = registry.call(TOOL_SEARCH, {"query": knowledge_query(task)})
    calls += 1
    return {
        "faq": faqs,
        "knowledge": knowledge,
        "tool_calls": calls,
        "note": "证据由 Python 调用单元15 的只读工具取出；群聊里没有工具，模型无法新增事实",
    }


def _evidence_facts(evidence: dict[str, Any]) -> list[str]:
    """把证据压成几行可读事实：执行员的离线草稿、评审的核对都用它。"""
    lines: list[str] = []
    for item in evidence.get("faq") or []:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "未找到":
            lines.append(f"{item.get('question_id', '')}：未找到（{item.get('storage', '')}）")
            continue
        lines.append(
            f"{item.get('question_id', '')}：{item.get('answer', '')}"
            f"（来源 {item.get('source_id', '')}）"
        )
    knowledge = evidence.get("knowledge") or {}
    hits = knowledge.get("hits") if isinstance(knowledge, dict) else None
    if isinstance(hits, list) and hits:
        first = hits[0]
        text = " ".join(str(first.get("text", "")).split())[:120]
        lines.append(f"规则片段 {len(hits)} 条，首条来源 {first.get('source', '')}：{text}")
    elif isinstance(hits, list):
        lines.append("规则检索：0 条命中")
    return lines


# ---------------------------------------------------------------- ⑤ 群聊的数据结构


@dataclass
class ChatMessage:
    """群聊里的一条消息：谁说的、第几位发言、说了什么、估算多少 token。"""

    seq: int
    sender: str
    content: str
    turn: int
    note: str = ""

    @property
    def sender_name(self) -> str:
        return role_name(self.sender)

    @property
    def chars(self) -> int:
        return len(self.content)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)

    def preview(self, limit: int = MESSAGE_PREVIEW) -> str:
        text = " ".join(self.content.split())
        return text if len(text) <= limit else text[:limit] + "…"


@dataclass
class ChatAgent:
    """等价于 AutoGen 的 ConversableAgent：有名字、有系统提示词、能接一轮话就说。

    它**只能发言**：没有工具、没有代码执行、不会主动找人说话。角色之间的先后由
    `GroupChatManager` 决定，这也是"轮次受控"能成立的前提。
    """

    role: str
    model: Any
    system_message: str = ""
    calls: int = 0

    def __post_init__(self) -> None:
        if not self.system_message:
            self.system_message = SYSTEM_PROMPTS[self.role]

    @property
    def name(self) -> str:
        return role_name(self.role)

    def generate_reply(
        self, chat: "GroupChat", task: str, evidence: dict[str, Any]
    ) -> str:
        """输入群聊现状，输出这一位的发言；一次发言 = 一次模型调用。"""
        response = self.model.invoke([
            SystemMessage(content=self.system_message),
            HumanMessage(content=_prompt_for(self, chat, task, evidence)),
        ])
        self.calls += 1
        text = str(getattr(response, "content", response)).strip()
        if not text:
            raise ChatProtocolError(f"{self.name}返回空内容")
        return text


def _prompt_for(
    agent: ChatAgent, chat: "GroupChat", task: str, evidence: dict[str, Any]
) -> str:
    """拼接这一位看到的全部信息：任务 → 证据 → 群聊记录 → 自己的职责。

    顺序固定：证据在群聊记录之前，模型读到的第一手材料永远是工具取回的事实。
    """
    lines = [
        f"用户任务：{task}",
        "",
        EVIDENCE_START,
        json.dumps(evidence, ensure_ascii=False),
        "",
        TRANSCRIPT_START,
    ]
    said = [m for m in chat.messages if m.sender != ROLE_INITIATOR]
    if said:
        for message in said:
            lines.append(f"[{message.turn}] {message.sender_name}：{message.content}")
    else:
        lines.append("（还没有人发言，你是第一位）")
    lines.extend([
        "",
        f"现在轮到你（{agent.name}）。只完成自己的职责，直接给出你的发言内容；"
        "不要复述别人的整段内容，也不要替别人发言。",
    ])
    return "\n".join(lines)


@dataclass
class GroupChat:
    """等价于 AutoGen 的 GroupChat：消息按顺序累积，按名单轮流选下一位发言人。

    `max_round` 是**消息条数**上限（**含发起者那条**）——AutoGen 里也是这个口径，
    写错就会少或少一轮。所以 3 个角色各发言一次要写 `max_round = 3 + 1`。
    """

    agents: list[ChatAgent]
    max_round: int
    token_budget: int = DEFAULT_TOKEN_BUDGET
    speaker_selection_method: str = "round_robin"
    messages: list[ChatMessage] = field(default_factory=list)

    # ---- 追加与选择 ----

    def append(self, sender: str, content: str, note: str = "") -> ChatMessage:
        """追加一条消息；发言序号（turn）由这里统一分配，别处不自己数。"""
        turn = 0 if sender == ROLE_INITIATOR else len(self.agent_messages) + 1
        message = ChatMessage(
            seq=len(self.messages) + 1, sender=sender, content=content, turn=turn, note=note
        )
        self.messages.append(message)
        return message

    def next_speaker(self) -> ChatAgent:
        """轮流发言：按名单顺序循环，已发言条数决定轮到谁。"""
        if self.speaker_selection_method != "round_robin":
            raise ValueError(f"不支持的选人方式：{self.speaker_selection_method}")
        if not self.agents:
            raise ValueError("群聊里没有任何角色")
        return self.agents[len(self.agent_messages) % len(self.agents)]

    # ---- 状态 ----

    @property
    def agent_messages(self) -> list[ChatMessage]:
        """角色发言（不含发起者那条）：轮次、成本、发言人数都按它算。"""
        return [message for message in self.messages if message.sender != ROLE_INITIATOR]

    @property
    def senders(self) -> list[str]:
        """实际发过言的角色（去重、按出现顺序）。"""
        seen: list[str] = []
        for message in self.agent_messages:
            if message.sender not in seen:
                seen.append(message.sender)
        return seen

    @property
    def tokens_used(self) -> int:
        return sum(message.tokens for message in self.messages)

    @property
    def full(self) -> bool:
        """消息条数触顶：再往下就是超轮次，不偷偷多给一轮。"""
        return len(self.messages) >= self.max_round

    @property
    def over_budget(self) -> bool:
        return self.tokens_used > self.token_budget

    def said_by(self, role: str) -> list[ChatMessage]:
        return [message for message in self.agent_messages if message.sender == role]


# ---------------------------------------------------------------- ⑥ 群聊管理器


@dataclass
class GroupChatManager:
    """等价于 AutoGen 的 GroupChatManager：只转发与推进轮次，**自己不调模型**。

    `model_calls` 恒为 0，这不是巧合而是断言点：把"管理器不产生成本"写成可核对的
    字段，模型调用次数才等于发言条数。
    """

    groupchat: GroupChat
    model_calls: int = 0
    # 每次复核的结论都留一份：多轮时"第1次不通过 → 第2次通过"要能对着看。
    verdicts: list[dict[str, Any]] = field(default_factory=list)

    def run(
        self, task: str, evidence: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """推进群聊直到出现终止条件；返回 (终止原因, 最后一次复核结论)。

        三个终止条件，优先级固定：复核通过 → 轮次用完 → 成本用完。
        检查永远在花钱之前：先看轮次、再看成本，最后才让下一位开口，所以不会出现
        "预算已经超了还多问一句"。

        **复核不通过不立刻收尾**：群聊继续走下一轮，执行员会看到复核意见再改一版，
        直到有人通过、或者轮次/成本用完。这正是"轮次受控"要压的东西——不靠模型自觉停，
        靠 Python 数着条数停。
        """
        verdict: dict[str, Any] = {}
        while True:
            if self.groupchat.full:
                # 轮次用完时有两种收尾：复核说过"不通过"就是复核不通过，
                # 复核根本没轮到就是"没复核完"——两件事不能混成一句。
                failed = bool(verdict) and not verdict.get("passed", False)
                return (STOP_REJECTED if failed else STOP_MAX_ROUND), verdict
            if self.groupchat.over_budget:
                return STOP_BUDGET, verdict
            speaker = self.groupchat.next_speaker()
            content = speaker.generate_reply(self.groupchat, task, evidence)
            self.groupchat.append(
                speaker.role, content, f"第{len(self.groupchat.agent_messages) + 1}位发言"
            )
            if speaker.role == ROLE_REVIEWER:
                verdict = parse_verdict(content)
                self.verdicts.append(dict(verdict))
                if verdict["passed"]:
                    # 通过了就收尾：多问一轮既不加分，也要多花钱。
                    return STOP_REVIEWED, verdict


def parse_verdict(content: str) -> dict[str, Any]:
    """解析复核结论；不合约定就判**不通过**，绝不"解析失败当通过"。"""
    raw = str(content).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {
            "passed": False,
            "reason": f"复核输出不是 JSON，按不通过处理：{' '.join(raw.split())[:100]}",
            "protocol_error": True,
        }
    if not isinstance(parsed, dict) or not isinstance(parsed.get("passed"), bool):
        return {
            "passed": False,
            "reason": f"复核输出缺少布尔字段 passed，按不通过处理：{' '.join(raw.split())[:100]}",
            "protocol_error": True,
        }
    return {"passed": parsed["passed"], "reason": str(parsed.get("reason", ""))}


# ---------------------------------------------------------------- ⑦ 一次协作


@dataclass
class Collaboration:
    """一次群聊协作的完整结果：消息流 + 终止原因 + 成本 + 交付结论。

    全部字段都是"跑完就能对账"的：消息条数、发言人数、估算 token、模型调用次数。
    """

    task: str
    evidence: dict[str, Any] = field(default_factory=dict)
    messages: list[ChatMessage] = field(default_factory=list)
    stop_reason: str = ""
    status: str = ""
    answer: str = ""
    review: dict[str, Any] = field(default_factory=dict)
    review_history: list[dict[str, Any]] = field(default_factory=list)
    rounds_requested: int = DEFAULT_ROUNDS
    max_round: int = DEFAULT_ROUNDS + 1
    token_budget: int = DEFAULT_TOKEN_BUDGET
    model_calls: int = 0
    manager_model_calls: int = 0
    elapsed_ms: int = 0
    model_name: str = ""
    error: str = ""

    # ---- 下面几个属性给渲染与验收用，避免各处重复过滤 ----

    @property
    def agent_messages(self) -> list[ChatMessage]:
        return [message for message in self.messages if message.sender != ROLE_INITIATOR]

    @property
    def participants(self) -> list[str]:
        seen: list[str] = []
        for message in self.agent_messages:
            if message.sender not in seen:
                seen.append(message.sender)
        return seen

    @property
    def tokens_used(self) -> int:
        return sum(message.tokens for message in self.messages)

    @property
    def delivered(self) -> bool:
        return self.stop_reason == STOP_REVIEWED and bool(self.answer)

    @property
    def reviewed(self) -> bool:
        """复核角色有没有发过言——与"通过没通过"是两件事。"""
        return ROLE_REVIEWER in self.participants

    @property
    def stop_note(self) -> str:
        return STOP_NOTES.get(self.stop_reason, "（未知终止原因）")


def run_collaboration(
    model: Any,
    registry: Any,
    task: str,
    rounds: int = DEFAULT_ROUNDS,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> Collaboration:
    """跑一次三角色群聊：守门 → 取证据 → 建群 → 轮流发言 → 按终止原因收尾。

    输入：模型、单元15 注册表、用户任务、轮次与成本上限。
    输出：Collaboration（消息流、终止原因、成本、交付结论）。

    守门在所有模型调用之前：任务不合格、轮次越界、预算非正一律当场报错。
    """
    clean_task = validate_goal(task)
    clean_rounds = validate_rounds(rounds)
    clean_budget = validate_budget(token_budget)
    started = time.perf_counter()

    collaboration = Collaboration(
        task=clean_task,
        rounds_requested=clean_rounds,
        # AutoGen 的口径：max_round 是消息条数，发起者那条也占一格。
        max_round=clean_rounds + 1,
        token_budget=clean_budget,
        model_name=_model_name(model),
    )
    collaboration.evidence = collect_evidence(registry, clean_task)

    agents = [ChatAgent(role=spec.role, model=model) for spec in AGENTS]
    chat = GroupChat(
        agents=agents, max_round=collaboration.max_round, token_budget=clean_budget
    )
    # 发起者那条消息把"任务 + 证据 + 发言顺序"一次交代清楚：它是群聊的第 1 条，
    # 也是 max_round 里被占掉的那一格。
    chat.append(
        ROLE_INITIATOR,
        "\n".join([
            f"用户任务：{clean_task}",
            f"{EVIDENCE_START}{json.dumps(collaboration.evidence, ensure_ascii=False)}",
            f"请按 {' → '.join(spec.name for spec in AGENTS)} 的顺序各发言一次，"
            f"最多 {clean_rounds} 轮、成本上限 {clean_budget} token（估算）。",
        ]),
        "发起者提交任务与工具证据（不调模型）",
    )

    manager = GroupChatManager(groupchat=chat)
    try:
        stop_reason, verdict = manager.run(clean_task, collaboration.evidence)
    except ChatProtocolError as exc:
        collaboration.error = str(exc)
        collaboration.stop_reason = STOP_REJECTED
        collaboration.status = STATUS_REJECTED
        collaboration.messages = list(chat.messages)
        collaboration.review_history = list(manager.verdicts)
        collaboration.model_calls = sum(agent.calls for agent in agents)
        collaboration.manager_model_calls = manager.model_calls
        collaboration.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return collaboration

    collaboration.messages = list(chat.messages)
    collaboration.stop_reason = stop_reason
    collaboration.status = STATUS_OF_STOP[stop_reason]
    collaboration.review = verdict
    collaboration.review_history = list(manager.verdicts)
    collaboration.model_calls = sum(agent.calls for agent in agents)
    collaboration.manager_model_calls = manager.model_calls

    # 复核通过才把执行员的**最后一版**草稿变成正式答复；其余情况答复一律为空。
    if stop_reason == STOP_REVIEWED:
        drafts = chat.said_by(ROLE_EXECUTOR)
        if not drafts:
            # 复核说通过却没有草稿，说明协作本身不成立：不交付，如实报错。
            collaboration.error = "复核通过但执行员没有发言，无法交付"
            collaboration.stop_reason = STOP_REJECTED
            collaboration.status = STATUS_REJECTED
        else:
            collaboration.answer = drafts[-1].content
    collaboration.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return collaboration


def _model_name(model: Any) -> str:
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(model).__name__


# ---------------------------------------------------------------- ⑧ 离线剧本模型


@dataclass
class ChatScript:
    """一份离线剧本：三个角色各自的发言队列，评审结论按调用顺序取。

    为什么用队列：多轮时规划员可能第二次发言，评审也要能演"先不通过"，
    所以每位角色各有一条队列，而不是一个固定字符串。
    """

    planner: list[str] = field(default_factory=list)
    executor: list[str] = field(default_factory=list)
    reviewer: list[dict[str, Any]] = field(default_factory=list)


class ScriptedChatModel(BaseChatModel):
    """离线剧本模型：占住 ChatDeepSeek 那一格，没密钥也能把群聊跑完。

    按 system 消息里的角色标记分派（标记用整段「【角色：xx】」，避免被群聊记录里的
    同名词误命中）。队列取完就用默认值：规划员说"无新增规划"，执行员**摘录证据原文**，
    评审员默认通过——离线模式不生成新事实，也不假装评审很严格。
    """

    model_name: str = OFFLINE_MODEL_NAME
    planner: list[str] = []
    executor: list[str] = []
    reviewer: list[dict[str, Any]] = []
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return OFFLINE_MODEL_NAME

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        object.__setattr__(self, "calls", int(self.calls) + 1)
        tag, payload = _split_prompt(messages)
        if tag == ROLE_PLANNER:
            content = self._take(self.planner, _default_plan(payload))
        elif tag == ROLE_EXECUTOR:
            content = self._take(self.executor, _scripted_answer(payload))
        elif tag == ROLE_REVIEWER:
            content = json.dumps(self._take_review(), ensure_ascii=False)
        else:
            raise ChatProtocolError(f"离线剧本模型认不出角色标记：{_joined_text(messages)[:80]}")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @staticmethod
    def _take(queue: list[str], fallback: str) -> str:
        return queue.pop(0) if queue else fallback

    def _take_review(self) -> dict[str, Any]:
        if self.reviewer:
            return dict(self.reviewer.pop(0))
        return {
            "passed": True,
            "reason": "（离线剧本）未安排评审结论，默认通过：发言只摘录证据原文",
        }


def _split_prompt(messages: Any) -> tuple[str, dict[str, Any]]:
    """从提示词里取出角色标记与证据 JSON。

    角色标记写在 system、数据写在 human，所以两段都要看：只取最后一条会认错角色。
    但**认角色只看 system**——群聊记录里可能出现「规划员：…」这样的字样，按整段
    文本匹配会在评审员那一轮误判成规划员。
    """
    text = _joined_text(messages)
    system_text = "\n".join(
        str(getattr(message, "content", ""))
        for message in _iter_messages(messages)
        if getattr(message, "type", "") == "system"
    )
    tag = ""
    for scope in (system_text, text):
        for role, marker in ROLE_TAGS.items():
            if marker in scope:
                tag = role
                break
        if tag:
            break
    payload: dict[str, Any] = {}
    if EVIDENCE_START in text:
        raw = text.split(EVIDENCE_START, 1)[1]
        raw = raw.split(TRANSCRIPT_START, 1)[0].strip()
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = {}
        payload = parsed if isinstance(parsed, dict) else {}
    return tag, payload


def _default_plan(evidence: dict[str, Any]) -> str:
    facts = _evidence_facts(evidence)
    return (
        "（离线剧本）规划：1) 已由平台工具取回事实，共 "
        f"{len(facts)} 项；2) 执行员按「结论 → 依据（编号与文件名）→ 待确认」三段作答；"
        "3) 评审员逐条核对依据是否出自证据。"
    )


def _scripted_answer(evidence: dict[str, Any]) -> str:
    """离线草稿：只摘录证据原文，不生成新事实。"""
    facts = _evidence_facts(evidence)
    lines = ["（离线脚本模式：以下只摘录平台工具返回的原文，最终以人工复核为准）"]
    if not facts:
        lines.append("本次没有取得任何工具证据，资料不足，建议转人工核实。")
        return "\n".join(lines)
    lines.append("结论：")
    lines.extend(f"- {fact}" for fact in facts)
    if any("未找到" in fact for fact in facts):
        lines.append("待确认：编号在本组资料库中没有记录，需人工核实后再答复。")
    return "\n".join(lines)


def _joined_text(messages: Any) -> str:
    parts: list[str] = []
    for message in _iter_messages(messages):
        content = getattr(message, "content", message)
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _iter_messages(messages: Any):
    if isinstance(messages, (list, tuple)):
        return list(messages)
    to_messages = getattr(messages, "to_messages", None)
    if callable(to_messages):
        return list(to_messages())
    return [messages]


def build_chat_model(script: ChatScript | None = None, provider: str | None = None):
    """按环境变量挑模型：`deepseek` 走真实 ChatDeepSeek，`mock` 走离线剧本。

    真实模式刻意不兜底：缺包或缺密钥当场报错，让课堂上看见原因，而不是悄悄换成
    离线答案（单元05 静默降级那个坑代价很大）。
    """
    chosen = (provider or os.getenv("LLM_PROVIDER", "mock")).strip().lower()
    if chosen == "deepseek":
        return build_plan_model(provider="deepseek")  # 复用单元16 的工厂：同一份密钥
    if chosen != "mock":
        raise ValueError("LLM_PROVIDER 只能是 mock 或 deepseek")
    script = script or ChatScript()
    return ScriptedChatModel(
        planner=list(script.planner),
        executor=list(script.executor),
        reviewer=[dict(item) for item in script.reviewer],
    )


# ---------------------------------------------------------------- ⑨ 渲染


def render_agents() -> str:
    """第1节要交的角色表：职责 / 输入 / 输出 / 不许做什么。"""
    lines = ["群聊角色（任务卡要求至少两个角色参与）", "角色\t职责\t输入\t输出\t不许做什么"]
    for spec in AGENTS:
        lines.append(
            f"{spec.name}\t{spec.duty}\t{spec.inputs}\t{spec.outputs}\t{spec.forbidden}"
        )
    lines.append(
        "发起者（用户）\t提交任务与工具证据、不发言\t用户任务\t群聊第 1 条消息"
        "\t不参与规划/答复/评审，不调模型"
    )
    return "\n".join(lines)


def render_flow() -> str:
    """群聊时序：谁先说、谁后说、靠什么停。"""
    return "\n".join([
        "三角色群聊（AutoGen 等价实现，本组电商售后）",
        "",
        "  发起者 user ── 任务 + 平台工具证据（Python 取数，模型不碰工具）",
        "      │         （这条占掉 max_round 的一格）",
        "      ▼",
        "  ┌──────────────── GroupChat（消息按顺序累积）────────────────┐",
        "  │ ① 规划员      看任务 + 证据          → 一段规划            │",
        "  │ ② 答复执行员  看任务 + 证据 + 规划    → 一版草稿（未交付）  │",
        "  │ ③ 评审员      看任务 + 证据 + 前两位  → {passed, reason}    │",
        "  └──────────────────────────────────────────────────────────┘",
        "      │  GroupChatManager：选下一位、判终止（自己不调模型）",
        "      ├ 复核通过       → 草稿成为正式答复，交付（不再多问一轮）",
        "      ├ 复核不通过     → **继续下一轮**：执行员看到意见再改一版",
        "      │                   └ 直到轮次/成本用完仍未通过 → 答复留空，转人工",
        "      ├ 轮次用完       → 复核还没轮到，未交付（不补一轮）",
        "      └ 成本超预算     → 下一位不再开口，未交付（不硬撑）",
        "",
        "终止条件三条：复核通过 / 轮次用完 / 成本用完；检查都发生在花钱之前。",
    ])


def render_evidence(evidence: dict[str, Any]) -> str:
    """工具证据：谁取的、取了几次、取到了什么。"""
    lines = [
        f"平台工具证据（工具调用 {evidence.get('tool_calls', 0)} 次，由 Python 执行）",
    ]
    for item in evidence.get("faq") or []:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "未找到":
            lines.append(f"- {item.get('question_id', '')}：未找到（{item.get('storage', '')}）")
        else:
            lines.append(
                f"- {item.get('question_id', '')}：{item.get('answer', '')}"
                f"｜来源 {item.get('source_id', '')}"
            )
    knowledge = evidence.get("knowledge") or {}
    hits = knowledge.get("hits") if isinstance(knowledge, dict) else None
    if isinstance(hits, list):
        lines.append(f"- 规则检索 {len(hits)} 条命中（查询：{knowledge.get('query', '')}）")
        for hit in hits[:2]:
            if isinstance(hit, dict):
                lines.append(
                    f"  · {hit.get('source', '')}：{' '.join(str(hit.get('text', '')).split())[:80]}"
                )
    return "\n".join(lines)


def render_transcript(collaboration: Collaboration, wrap: bool = True) -> str:
    """消息流全文：序号 / 发言序号 / 发送者 / 字符 / 估算 token / 正文。

    任务卡要求"消息内容可查看"，所以这里**打印正文**（按 76 字折行），不截断。
    唯一的例外是发起者那条：它的证据 JSON 与上文「平台工具证据」完全重复，
    这里只留一句摘要并注明原文仍在 `messages[0].content`，避免同一段 JSON 印两遍。
    """
    lines = ["消息流（全文，可逐条回放）", "序号\t第几位\t发送者\t字符\t估算token\t说明"]
    for message in collaboration.messages:
        turn = "—" if message.turn == 0 else str(message.turn)
        lines.append(
            f"{message.seq}\t{turn}\t{message.sender_name}\t{message.chars}\t"
            f"{message.tokens}\t{message.note}"
        )
    lines.append("")
    for message in collaboration.messages:
        head = f"#{message.seq} {message.sender_name}"
        if message.turn:
            head += f"（第{message.turn}位）"
        lines.append(head)
        body = _body_for_render(message) if wrap else " ".join(message.content.split())
        for paragraph in body.splitlines() or [""]:
            wrapped = textwrap.wrap(paragraph, TRANSCRIPT_WIDTH) or [""]
            lines.extend(f"    {line}" for line in wrapped)
        lines.append("")
    return "\n".join(lines).rstrip()


def _body_for_render(message: ChatMessage) -> str:
    """渲染用正文：发起者那条把证据 JSON 换成摘要（证据已在上文完整列出）。"""
    if message.sender != ROLE_INITIATOR or EVIDENCE_START not in message.content:
        return message.content
    kept: list[str] = []
    for line in message.content.splitlines():
        if line.startswith(EVIDENCE_START):
            kept.append(f"{EVIDENCE_START}（JSON 全文见上文「平台工具证据」段，原文保留在消息记录里）")
        else:
            kept.append(line)
    return "\n".join(kept)


def render_termination(collaboration: Collaboration) -> str:
    """终止条件与成本：轮次上限、实际发言、模型调用、估算 token、停止原因。"""
    return "\n".join([
        "终止条件与成本（可核对）",
        f"申请轮次：{collaboration.rounds_requested}；max_round（消息条数上限，含发起者）："
        f"{collaboration.max_round}；实际消息：{len(collaboration.messages)} 条",
        f"实际发言：{len(collaboration.agent_messages)} 次；参与角色："
        f"{len(collaboration.participants)} 个（{'、'.join(role_name(r) for r in collaboration.participants) or '无'}）",
        f"模型调用：{collaboration.model_calls} 次（= 发言条数）；管理器调用："
        f"{collaboration.manager_model_calls} 次（不调模型）",
        f"估算 token：{collaboration.tokens_used} / 预算 {collaboration.token_budget}"
        f"（{CHARS_PER_TOKEN} 字符≈1 token）",
        f"终止原因：{collaboration.stop_reason} —— {collaboration.stop_note}",
        f"状态：{collaboration.status}",
        f"交付的答复：{collaboration.answer or '（空：未交付）'}",
        render_verdicts(collaboration),
    ])


def render_verdicts(collaboration: Collaboration) -> str:
    """复核结论逐次列出：多轮时能看出"第一次不通过、第二次通过"。"""
    if not collaboration.review_history:
        return "复核结论：（未发生：复核角色没轮到）"
    rows = []
    for index, verdict in enumerate(collaboration.review_history, start=1):
        flag = "通过" if verdict.get("passed") else "不通过"
        rows.append(f"第{index}次 {flag}（{verdict.get('reason', '')}）")
    return "复核结论：" + "；".join(rows)


def render_result(collaboration: Collaboration) -> str:
    """一次协作的完整结果：证据 → 消息流 → 终止条件。"""
    return "\n".join([
        f"任务：{collaboration.task}",
        "",
        render_evidence(collaboration.evidence),
        "",
        render_transcript(collaboration),
        "",
        render_termination(collaboration),
        f"耗时：{collaboration.elapsed_ms} ms",
    ])


CONCEPTS: list[tuple[str, str, str]] = [
    ("群聊", "多个角色在同一段消息流里轮流发言", "GroupChat：messages 按顺序累积"),
    ("发言顺序", "谁来发言由管理器选，不靠模型自觉", "round_robin：发言条数决定轮到谁"),
    ("消息流", "后发言的人看得到前面所有人的原话", "render_transcript() 打印全文"),
    ("轮次上限", "消息条数硬上限（含发起者那条）", f"max_round = 轮次 + 1，最多 {MAX_ROUNDS} 轮"),
    ("成本上限", "估算 token 超预算就不再开口", "检查在花钱之前，超了状态即「未交付」"),
    ("复核确认", "执行员写草稿，评审员才有放行权", "passed=True 才把草稿变成答复"),
]


def render_concepts() -> str:
    lines = ["概念记录（第1节）", "概念\t含义\t在本项目里落在哪"]
    lines.extend(f"{name}\t{meaning}\t{here}" for name, meaning, here in CONCEPTS)
    return "\n".join(lines)


__all__ = [
    "AGENTS",
    "AgentSpec",
    "CHARS_PER_TOKEN",
    "ChatAgent",
    "ChatMessage",
    "ChatProtocolError",
    "ChatScript",
    "Collaboration",
    "DEFAULT_ROUNDS",
    "DEFAULT_TOKEN_BUDGET",
    "GroupChat",
    "GroupChatManager",
    "MAX_ROUNDS",
    "ROLE_EXECUTOR",
    "ROLE_INITIATOR",
    "ROLE_NAMES",
    "ROLE_ORDER",
    "ROLE_PLANNER",
    "ROLE_REVIEWER",
    "ScriptedChatModel",
    "STATUS_BUDGET",
    "STATUS_DELIVERED",
    "STATUS_MAX_ROUND",
    "STATUS_REJECTED",
    "STOP_BUDGET",
    "STOP_MAX_ROUND",
    "STOP_NOTES",
    "STOP_REJECTED",
    "STOP_REVIEWED",
    "build_chat_model",
    "collect_evidence",
    "estimate_tokens",
    "extract_faq_ids",
    "knowledge_query",
    "parse_verdict",
    "render_agents",
    "render_concepts",
    "render_evidence",
    "render_flow",
    "render_result",
    "render_termination",
    "render_verdicts",
    "render_transcript",
    "role_name",
    "run_collaboration",
    "validate_budget",
    "validate_rounds",
]
