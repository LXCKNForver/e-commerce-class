"""单元19：对话记忆——完整历史 / 摘要 / 向量长期记忆，三种模式 + 会话隔离。

承接单元17 的执行器（模型 + 工具 + `create_agent`）与单元15 的工具注册表，
本单元**不新增任何业务能力**，只回答一个问题：上一轮说过的话，这一轮怎么用。

顺序固定为 **读 → 答 → 写**：

    ① load(session, 本轮问题)  回忆：取出历史/摘要/长期记忆，顺便解析「刚才那项」
    ② run_agent(...)           作答：工具照查，事实一律重新取，不来自记忆
    ③ save(session, 问答)      落记忆：只有正常完成的轮次才写，失败轮次不污染记忆

三条容易被混淆的界线，本单元用代码固定住：

1. **对话记忆库 ≠ 业务知识库**。资料在 `.chroma_unit15/`（制度原文），
   对话写在 `.chroma_unit19_memory/`（用户说过的话）。长期记忆召回的内容只作语境，
   渲染时带「仅作语境」前缀，永远不当成证据来源。
2. **记忆只解决指代，不解决事实**。第二轮问「刚才那项，时限是多久」，记忆负责告诉
   模型「那项 = EC-FAQ-01」，时限仍然由工具现查——步骤表里第二轮必须有工具调用。
3. **会话隔离靠键前缀 + 库端过滤双保险**。完整历史与摘要的字典键带项目与会话前缀；
   长期记忆按 `source=session:<id>` 在 Chroma 端过滤，不靠"查回来再自己筛"。

    uv run --extra embedding --extra chroma --extra langchain python -m app.memory_lab
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.chroma_store import (
    PROJECT_ID,
    UNIT19_MEMORY_COLLECTION,
    UNIT19_MEMORY_COLLECTION_METADATA,
    ChromaVectorStore,
    resolve_unit19_memory_db_path,
)
from app.lc_chain import validate_question
from app.lc_agent import STOP_FINISHED

# 第2、4节演示用的默认会话；换一个 ID 就是另一个用户，两者互不串话。
DEFAULT_SESSION = "class-demo"
# 会话 ID 只允许这些字符：它既进字典键，也进长期记忆的过滤值，太随意会引来注入。
SESSION_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")

MODE_BUFFER = "buffer"
MODE_SUMMARY = "summary"
MODE_VECTOR = "vector"
MODE_HYBRID = "hybrid"
MODES = (MODE_BUFFER, MODE_SUMMARY, MODE_VECTOR, MODE_HYBRID)
DEFAULT_MODE = MODE_HYBRID
MODE_LABELS = {
    MODE_BUFFER: "完整历史（buffer）",
    MODE_SUMMARY: "摘要记忆（summary）",
    MODE_VECTOR: "向量长期记忆（vector）",
    MODE_HYBRID: "短期 + 长期（hybrid）",
}

# 长期记忆按语义召回几条。给多了会把无关轮次也塞进语境，3 条够讲清取舍。
DEFAULT_TOP_K = 3
# 存进长期记忆的答案只留这么多字：记忆只需记住"问了什么、落在哪个编号"，
# 全文留给知识库；存太长反而让无关文字稀释向量，还会把答案当事实反复引用。
MAX_MEMORY_ANSWER_CHARS = 160
# 记忆文本第一行固定这个格式，轮次就从这里取（库的过滤字段里带不回来，见 _as_items）。
MEMORY_TURN_PATTERN = re.compile(r"^第(\d+)轮")

# 本组业务对象只有这两类编号；指代消解找的就是它们。
OBJECT_PATTERN = re.compile(r"EC-FAQ-\d{2}|DOC-02-\d{2}")
# 跨轮指代的说法；命中任一个才认为下一句在"接着上一轮说"。
REFERENCE_WORDS = (
    "刚才", "刚刚", "上一轮", "上一问", "上一条", "上面", "那项", "那一条",
    "这项", "这一条", "同上面", "它", "这个编号", "那个编号",
)

SUMMARY_PREFIX = "对话历史摘要（仅作语境，事实以工具返回为准）："
LONG_TERM_PREFIX = "相关历史对话（仅作语境，事实以工具返回为准）："
RESOLVE_PREFIX = "上一轮关注的对象"
OFFLINE_SUMMARY_MODEL = "offline-summary（不调用模型，按固定格式记录）"
MEMORY_DOC_TYPE = "conversation_memory"

SUMMARY_PROMPT = (
    "请把下面的对话压缩成简短、准确的中文摘要。"
    "必须保留 FAQ 编号、资料编号和用户关注点；不得加入对话里没有的事实。\n"
    "已有摘要：{summary}\n新增对话：{new_lines}\n新摘要："
)


@dataclass(frozen=True)
class MemoryTurn:
    """落进记忆的一轮问答；tools 记的是这一轮真正调用过的工具。"""

    index: int
    question: str
    answer: str
    tools: tuple[str, ...] = ()

    def render(self) -> str:
        return f"用户：{self.question}\n助手：{self.answer}"


@dataclass(frozen=True)
class ReferenceResolution:
    """一次跨轮指代解析的结果：把「刚才那项」落到具体编号上。"""

    question: str
    keyword: str
    object_id: str
    from_turn: int
    source: str

    def render(self) -> str:
        return (
            f"{RESOLVE_PREFIX}：{self.object_id}（来自第 {self.from_turn} 轮的{self.source}，"
            f"由问句中的「{self.keyword}」触发）"
        )


@dataclass(frozen=True)
class MemorySnapshot:
    """① 回忆的结果：这些是本轮要注入给模型的全部内容，多一个字都没有。"""

    mode: str
    session_id: str
    history: tuple[dict[str, str], ...]
    context: str
    recalled: tuple[dict[str, Any], ...]
    resolution: ReferenceResolution | None


@dataclass(frozen=True)
class TurnRecord:
    """一轮完整对话的运行记录：既有答案，也有"记忆帮了什么忙"。"""

    index: int
    session_id: str
    mode: str
    question: str
    answer: str
    tools: tuple[str, ...]
    stop_reason: str
    snapshot: MemorySnapshot
    saved: bool
    save_note: str

    @property
    def finished(self) -> bool:
        return self.stop_reason == STOP_FINISHED and bool(self.answer)


@dataclass
class SummaryEngine:
    """摘要记忆的压缩器：有模型就调模型，没有就走离线格式（两边都保留编号）。

    `calls` 记的是压缩摘要额外调用模型的次数——这是摘要模式的成本证据。
    """

    model: Any = None
    calls: int = 0

    @property
    def engine_name(self) -> str:
        if self.model is None:
            return OFFLINE_SUMMARY_MODEL
        return type(self.model).__name__

    def update(self, summary: str, turn: MemoryTurn) -> str:
        """输入旧摘要与本轮问答；返回新摘要，并保证编号一个不丢。"""
        new_lines = turn.render()
        if self.model is None:
            fresh = _offline_summary(summary, turn)
        else:
            prompt = SUMMARY_PROMPT.format(summary=summary or "（无）", new_lines=new_lines)
            self.calls += 1
            fresh = _text_of(self.model.invoke(prompt)).strip()
        # 摘要最容易丢的就是编号，而编号恰恰是下一轮指代要落的地方：
        # 模型写漏了就由程序补回，并在摘要里写明是补的，不装作模型记得。
        needed = list(dict.fromkeys(OBJECT_PATTERN.findall(f"{summary}\n{new_lines}")))
        missing = [name for name in needed if name not in fresh]
        if missing:
            fresh = f"{fresh}（程序补回的关键编号：{'、'.join(missing)}）"
        return fresh


def _text_of(result: Any) -> str:
    """模型返回可能是 AIMessage 或纯字符串，统一取文字。"""
    content = getattr(result, "content", result)
    return str(content or "")


def _offline_summary(summary: str, turn: MemoryTurn) -> str:
    """离线模式不调模型，也**不把答案原文抄进摘要**。

    只记"这一轮问了什么、涉及哪个编号"：摘要的职责是让下一轮知道在聊哪个对象，
    不是保存事实——事实要查工具。这样离线模式下也不会出现"记忆编造答复"。
    """
    ids = OBJECT_PATTERN.findall(turn.question)
    focus = "、".join(dict.fromkeys(ids)) if ids else "（本轮没有出现编号）"
    line = f"第{turn.index}轮：用户问「{turn.question}」；涉及编号：{focus}。"
    return f"{summary} {line}".strip()


@dataclass
class VectorMemory:
    """长期记忆：一轮问答写一条向量，按语义召回；不同会话靠 source 在库端隔离。"""

    database_path: Path | None = None
    project_id: str = PROJECT_ID
    top_k: int = DEFAULT_TOP_K
    _store: Any = field(default=None, init=False, repr=False)

    def store(self) -> ChromaVectorStore:
        """延迟建库：只有真要写/读长期记忆时才碰 Chroma。"""
        if self._store is None:
            self._store = ChromaVectorStore(
                self.database_path or resolve_unit19_memory_db_path(),
                UNIT19_MEMORY_COLLECTION,
                UNIT19_MEMORY_COLLECTION_METADATA,
            )
        return self._store

    @staticmethod
    def source_of(session_id: str) -> str:
        """会话隔离的过滤值：写进每条记忆的 source，检索时原样带回做过滤。"""
        return f"session:{session_id}"

    def count(self, session_id: str) -> int:
        """该会话已有多少条长期记忆（不含别的会话，也不含知识库）。"""
        store = self.store()
        if store.collection.count() == 0:
            return 0
        got = store.collection.get(
            where={
                "$and": [
                    {"project_id": self.project_id},
                    {"source": self.source_of(session_id)},
                ]
            }
        )
        return len(got["ids"])

    def clear(self, session_id: str) -> None:
        """删掉这个会话的全部长期记忆；换一批演示前先清，免得旧轮次混进来。"""
        store = self.store()
        if store.collection.count() == 0:
            return
        store.collection.delete(where={"source": self.source_of(session_id)})

    def add(self, session_id: str, turn: MemoryTurn) -> None:
        """把一轮问答写成一条长期记忆；同一轮重复写入是更新，不会多出一条。"""
        from app.embedding import embed_many

        store = self.store()
        compact = " ".join(turn.answer.split())[:MAX_MEMORY_ANSWER_CHARS]
        store.collection.upsert(
            ids=[f"memory:{session_id}:{turn.index}"],
            documents=[f"第{turn.index}轮 用户：{turn.question}｜助手：{compact}"],
            # 向量必须和知识库同一个模型：换模型等于换向量空间，旧记忆全废。
            embeddings=embed_many([f"{turn.question}\n{turn.answer}"]),
            metadatas=[
                {
                    "project_id": self.project_id,
                    "doc_type": MEMORY_DOC_TYPE,
                    "source": self.source_of(session_id),
                    "session_id": session_id,
                    "turn": turn.index,
                }
            ],
        )

    def recall(self, session_id: str, question: str, top_k: int | None = None) -> list[dict]:
        """按当前问句召回本会话最相关的几轮；别的会话一条都拿不到。"""
        store = self.store()
        if store.collection.count() == 0:
            return []
        return store.query(
            question,
            project_id=self.project_id,
            top_k=top_k or self.top_k,
            source=self.source_of(session_id),
        )


class AgentMemoryManager:
    """按"项目 + 会话"隔离的多模式记忆；读、答、写三步由调用方按序调用。

    buffer 与 summary 只活在内存里（进程重启就消失），vector 落 Chroma
    （重启后仍可召回）——这个差异本身就是第3节要对照的东西，别抹平它。
    """

    def __init__(
        self,
        model: Any = None,
        database_path: Path | None = None,
        project_id: str = PROJECT_ID,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.project_id = project_id
        self.summary_engine = SummaryEngine(model=model)
        self.vector = VectorMemory(database_path, project_id, top_k)
        self._turns: dict[str, list[MemoryTurn]] = {}
        self._histories: dict[str, list[dict[str, str]]] = {}
        self._summaries: dict[str, str] = {}

    def _key(self, session_id: str) -> str:
        """校验会话 ID；项目 ID 也进键，防止同名会话在不同项目间串话。"""
        if not isinstance(session_id, str) or not SESSION_PATTERN.fullmatch(session_id):
            raise ValueError("session_id 只允许 1—64 位英文字母、数字、_ 或 -")
        return f"{self.project_id}:{session_id}"

    # ---------- ① 读 ----------

    def load(self, session_id: str, question: str, mode: str = DEFAULT_MODE) -> MemorySnapshot:
        """回忆：按模式取出历史/摘要/长期记忆，并解析本轮的跨轮指代。"""
        key = self._key(session_id)
        if mode not in MODES:
            raise ValueError(f"未知记忆模式：{mode}，可选 {MODES}")
        # 问题校验与作答入口用同一个函数（run_turn 里也会走一次），
        # 这样"什么样的输入根本进不了记忆"在两处是一致的。
        question = validate_question(question)

        history: tuple[dict[str, str], ...] = ()
        if mode in (MODE_BUFFER, MODE_HYBRID):
            history = tuple(self._histories.get(key, []))

        summary = self._summaries.get(key, "") if mode in (MODE_SUMMARY, MODE_HYBRID) else ""
        recalled: tuple[dict[str, Any], ...] = ()
        if mode in (MODE_VECTOR, MODE_HYBRID):
            recalled = tuple(self.vector.recall(session_id, question))

        # 候选来源按模式决定，但解析规则只有一份（下面那个函数）。
        candidates = _candidates(mode, self._turns.get(key, []), summary, recalled)
        resolution = resolve_reference(question, candidates)

        blocks: list[str] = []
        if summary:
            blocks.append(f"{SUMMARY_PREFIX}\n{summary}")
        if recalled:
            lines = [f"- 第{item['turn']}轮：{item['text']}" for item in _as_items(recalled)]
            blocks.append(f"{LONG_TERM_PREFIX}\n" + "\n".join(lines))
        if resolution is not None:
            blocks.append(
                f"{resolution.render()}。本轮若要给事实，仍须调用工具现查，不要用记忆里的数字。"
            )
        return MemorySnapshot(
            mode=mode,
            session_id=session_id,
            history=history,
            context="\n".join(blocks),
            recalled=recalled,
            resolution=resolution,
        )

    # ---------- ③ 写 ----------

    def save(
        self,
        session_id: str,
        question: str,
        answer: str,
        mode: str = DEFAULT_MODE,
        tools: tuple[str, ...] = (),
    ) -> MemoryTurn:
        """落记忆：只写正常完成的轮次，触顶或空答案的轮次不进记忆。"""
        key = self._key(session_id)
        if mode not in MODES:
            raise ValueError(f"未知记忆模式：{mode}，可选 {MODES}")
        turns = self._turns.setdefault(key, [])
        turn = MemoryTurn(index=len(turns) + 1, question=question, answer=answer, tools=tuple(tools))
        turns.append(turn)

        if mode in (MODE_BUFFER, MODE_HYBRID):
            history = self._histories.setdefault(key, [])
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
        if mode in (MODE_SUMMARY, MODE_HYBRID):
            self._summaries[key] = self.summary_engine.update(self._summaries.get(key, ""), turn)
        if mode in (MODE_VECTOR, MODE_HYBRID):
            self.vector.add(session_id, turn)
        return turn

    # ---------- 查看（验收与渲染用） ----------

    def turns(self, session_id: str) -> list[MemoryTurn]:
        return list(self._turns.get(self._key(session_id), []))

    def summary(self, session_id: str) -> str:
        return self._summaries.get(self._key(session_id), "")

    def long_term_count(self, session_id: str) -> int:
        return self.vector.count(session_id)

    def clear_session(self, session_id: str) -> None:
        """清空某个会话的全部记忆（含已落盘的长期记忆）。

        长期记忆是持久的，重跑一遍演示就会在同一会话下多出一轮——不清的话
        "第2轮回忆到 3 条"这种输出会随历史变，课堂上讲不清。要演示跨运行召回时
        别调它（证据脚本里是另起一个进程读，不靠清空）。
        """
        key = self._key(session_id)
        self._turns.pop(key, None)
        self._histories.pop(key, None)
        self._summaries.pop(key, None)
        self.vector.clear(session_id)


def _as_items(recalled: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    """补齐渲染要用的 turn 字段。

    `ChromaVectorStore.query` 返回的命中只有 id/text/source 等固定字段，metadata 里的
    turn 带不回来，所以轮次从文本开头那句「第N轮」正则取——写记忆时格式是固定的。
    取不到才退回用召回顺序（0 起），并显式标成 0，免得把顺序当轮次看。
    """
    items: list[dict[str, Any]] = []
    for position, item in enumerate(recalled, start=1):
        record = dict(item)
        text = str(record.get("text", "") or "")
        matched = MEMORY_TURN_PATTERN.match(text)
        record["turn"] = int(matched.group(1)) if matched else 0
        record["text"] = text
        items.append(record)
    return items


def recalled_lines(recalled) -> list[str]:
    """召回条目渲染成一行一条（记忆文本本身是单行的，预览截断只为排版）。"""
    lines: list[str] = []
    for item in _as_items(tuple(recalled)):
        # 文本开头那句「第N轮」已经取出来当 turn 了，预览里不再重复一遍。
        body = MEMORY_TURN_PATTERN.sub("", str(item["text"]), count=1).strip()
        lines.append(f"第{item['turn']}轮：{body[:70]}")
    return lines


def _candidates(
    mode: str,
    turns: list[MemoryTurn],
    summary: str,
    recalled: tuple[dict[str, Any], ...],
) -> list[tuple[int, str, str]]:
    """按模式给出「第几轮 → 文本 → 来源名」的候选；解析规则只有这一处。"""
    if mode in (MODE_BUFFER, MODE_HYBRID):
        return [(turn.index, f"{turn.question}\n{turn.answer}", "对话历史") for turn in turns]
    if mode == MODE_SUMMARY:
        items = [(turn.index, turn.question, "摘要") for turn in turns]
        if summary:
            items.append((len(turns), summary, "摘要"))
        return items
    if mode == MODE_VECTOR:
        return [
            (int(item.get("turn", position)), str(item.get("text", "")), "长期记忆")
            for position, item in enumerate(recalled, start=1)
        ]
    return []


def resolve_reference(
    question: str, candidates: list[tuple[int, str, str]]
) -> ReferenceResolution | None:
    """把「刚才那项」落到具体编号；问句里已经写明编号就不算指代。

    刻意用规则而不是"让模型自己体会"：解析结果要能打印、能断言、能复现，
    而模型内部的理解看不见。模型拿到的就是这里打印出来的这一句。
    """
    if not isinstance(question, str) or not question.strip():
        return None
    if OBJECT_PATTERN.search(question):
        # 用户自己写清了编号，不需要消解。
        return None
    keyword = next((word for word in REFERENCE_WORDS if word in question), "")
    if not keyword:
        return None
    for turn_index, text, source in reversed(candidates):
        body = str(text or "")
        # 先在这一轮的"问句"里找编号，找不到才看整段（含答复）。
        # 答复里常出现来源编号（DOC-02-xx），若一律取最后一个，指代就会落到资料编号上，
        # 而用户说的"刚才那项"指的是 FAQ 条目。
        found = OBJECT_PATTERN.findall(body.split("\n", 1)[0]) or OBJECT_PATTERN.findall(body)
        if found:
            return ReferenceResolution(
                question=question,
                keyword=keyword,
                object_id=found[0],
                from_turn=turn_index if turn_index else len(candidates),
                source=source,
            )
    return None


def run_turn(
    executor: Any,
    manager: AgentMemoryManager,
    session_id: str,
    question: str,
    mode: str = DEFAULT_MODE,
    max_steps: int | None = None,
) -> TurnRecord:
    """跑一轮带记忆的问答：先回忆，再作答，最后只在正常完成时落记忆。"""
    from app.lc_agent import run_agent

    clean = validate_question(question)
    # 轮次在"写"之前数：save 之后它就多了一轮，渲染会显示成下一轮。
    turn_index = len(manager.turns(session_id))
    snapshot = manager.load(session_id, clean, mode)

    messages: list[Any] = []
    if snapshot.context:
        # 摘要与长期记忆是"语境"，不是对话的一部分，所以走 system 而不是 user。
        messages.append({"role": "system", "content": snapshot.context})
    messages.extend(dict(item) for item in snapshot.history)

    run = run_agent(executor, clean, max_steps, history=messages)
    tools = tuple(step.tool for step in run.steps)
    finished = run.stop_reason == STOP_FINISHED and bool(run.answer)
    if finished:
        manager.save(session_id, clean, run.answer, mode, tools)
        note = "已写入记忆"
    else:
        note = f"未写入记忆（{run.stop_reason}）：失败轮次不能变成下一轮的事实"
    return TurnRecord(
        index=turn_index,
        session_id=session_id,
        mode=mode,
        question=clean,
        answer=run.answer,
        tools=tools,
        stop_reason=run.stop_reason,
        snapshot=snapshot,
        saved=finished,
        save_note=note,
    )


# ---------------- 渲染（第1、3、4节的证据） ----------------

def render_memory_flow() -> str:
    """第1节要交的本组记忆流程图：读、答、写三步与两个库的分工。"""
    lines = [
        "本组记忆流程（读 → 答 → 写，顺序固定）：",
        "步骤\t做什么\t落在哪\t注意",
        "① 读\t按会话取出历史/摘要/长期记忆，解析「刚才那项」\t"
        "内存（buffer/summary）＋ .chroma_unit19_memory（vector）\t只读，不改任何业务数据",
        "② 答\t执行器按本轮问题重新选工具、重新取事实\t"
        "单元17 执行器 → 单元15 三项只读工具\t事实只来自工具，不来自记忆",
        "③ 写\t把这一轮问答写进对应模式的记忆\t"
        "同上\t只写正常完成的轮次；触顶/空答案不写",
        "",
        "两个库严格分开（混在一起会把上一轮的话当成制度原文引用）：",
        "库\t装什么\t谁在读\t本单元是否写入",
        ".chroma_unit15/\t本组 3md+1pdf+1csv 资料原文\tsearch_knowledge\t否",
        ".sqlite_unit15/\t常见售后问题.csv 的 3 行标准答复\tlookup_faq\t否",
        ".chroma_unit19_memory/\t多轮对话（按 session 隔离）\t本单元的长期记忆\t是",
    ]
    return "\n".join(lines)


def render_modes() -> str:
    """第3节要交的三种模式对比：注入什么、重启后还在不在、代价是什么。"""
    lines = [
        "三种记忆模式对照：",
        "模式\t这一轮带进模型的内容\t进程重启后\t额外模型调用\t代价",
        f"{MODE_LABELS[MODE_BUFFER]}\t全部历史消息，一轮两条\t消失\t0\t"
        "轮次多了上下文变长，token 线性上涨",
        f"{MODE_LABELS[MODE_SUMMARY]}\t一句压缩摘要\t消失\t每轮 1 次\t"
        "省上下文，但细节可能丢（编号由程序补回兜底）",
        f"{MODE_LABELS[MODE_VECTOR]}\t按本轮问句召回的若干轮\t仍在（落盘）\t0\t"
        "依赖向量召回，问句与旧轮次不相似就召回不到",
        f"{MODE_LABELS[MODE_HYBRID]}\t完整历史 + 召回的长期记忆\t部分在\t每轮 0—1 次\t"
        "信息最全，注入也最长",
    ]
    return "\n".join(lines)


def render_snapshot(snapshot: MemorySnapshot) -> str:
    """① 回忆这一步到底带回了什么——打印出来才叫"可查看"。"""
    lines = [
        f"记忆模式：{MODE_LABELS.get(snapshot.mode, snapshot.mode)}；会话：{snapshot.session_id}",
        f"带回来的对话消息：{len(snapshot.history) / 2:.0f} 轮（{len(snapshot.history)} 条）",
    ]
    if snapshot.recalled:
        lines.append(f"长期记忆召回：{len(snapshot.recalled)} 条")
        for text in recalled_lines(snapshot.recalled):
            lines.append(f"  - {text}")
    else:
        lines.append("长期记忆召回：0 条")
    if snapshot.resolution is not None:
        lines.append(f"指代解析：{snapshot.resolution.render()}")
    else:
        lines.append("指代解析：本轮问句不含指代词，或已写明编号，无需消解")
    if snapshot.context:
        lines.append("注入的语境：")
        for block in snapshot.context.split("\n"):
            lines.append(f"  {block}")
    else:
        lines.append("注入的语境：（空）")
    return "\n".join(lines)


def render_turn(record: TurnRecord) -> str:
    """一轮问答的现场：问了什么、重新查了哪些工具、答了什么、写没写进记忆。"""
    tools = "、".join(record.tools) if record.tools else "（无）"
    lines = [
        f"第 {record.index + 1} 轮｜会话 {record.session_id}｜模式 {record.mode}",
        f"  问：{record.question}",
        f"  本轮调用的工具：{tools}",
        f"  答：{record.answer or '（空）'}",
        f"  停止原因：{record.stop_reason}；{record.save_note}",
    ]
    return "\n".join(lines)


def render_turns(records: list[TurnRecord]) -> str:
    return "\n".join(render_turn(record) for record in records)


__all__ = [
    "DEFAULT_MODE",
    "DEFAULT_SESSION",
    "DEFAULT_TOP_K",
    "MODE_BUFFER",
    "MODE_HYBRID",
    "MODE_SUMMARY",
    "MODE_VECTOR",
    "MODE_LABELS",
    "MODES",
    "OFFLINE_SUMMARY_MODEL",
    "AgentMemoryManager",
    "MemorySnapshot",
    "MemoryTurn",
    "ReferenceResolution",
    "SummaryEngine",
    "TurnRecord",
    "VectorMemory",
    "render_memory_flow",
    "render_modes",
    "render_snapshot",
    "render_turn",
    "render_turns",
    "recalled_lines",
    "resolve_reference",
    "run_turn",
]
