"""单元15：工具集的三项只读工具 + 白名单校验 + 注册表调度。

任务卡要求工具统一按 `name`、`description`、`parameters`、`execute` 四项定义。
这四项不是形式：前三项是**给模型看的接口说明**，第四项是**程序里唯一的执行入口**，
两者由同一个 `ToolSpec` 生成，所以"模型能选的"和"程序会跑的"永远一致——
改一处就同时生效，不会出现"模型以为有这个工具、程序里没有"的裂缝。

三项工具对应两类数据源（第1节课要区分的东西）：

    search_knowledge        非结构化：本组资料进 Chroma，靠向量相似度命中（单元09）
    lookup_faq              结构化  ：同一份 CSV 进 SQLite，靠编号精确命中（app/faq_db）
    compute_return_deadline 计算    ：签收日期 + 资料里写明的时限 → 截止日期（选做）

第三项是选做。本组 DOC-02-01 里写着"签收后7天内"，**有真实数值**，所以算得动；
但计算工具的前提是：天数必须来自资料原文，且由 Python 做日期加法——
模型既不能自己编天数，也不允许自己口算日期（它算错过也不会告诉你）。

    注册 → build_registry()
    调度 → ToolRegistry.call()：先过 authorize_tool()，再进真实业务函数
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from app.evaluation import normalize_source  # 来源口径只有一个标准：文件名
from app.faq_db import fetch_faq
from app.retriever import RetrievalService

# 三个工具名写死在常量里：白名单、提示词、轨迹都读它，别在别处再写字面量。
TOOL_SEARCH = "search_knowledge"
TOOL_FAQ = "lookup_faq"
TOOL_DEADLINE = "compute_return_deadline"
FINISH_TOOL = "finish"

# 允许模型调用的只有这三项；`finish` 是循环的停止动作，不进工具白名单。
ALLOWED_TOOLS: frozenset[str] = frozenset({TOOL_SEARCH, TOOL_FAQ, TOOL_DEADLINE})
# 每个工具只接受指定的字段，且格式要匹配：多一个字段、少一个字段、格式不对都拒绝。
ARGUMENT_RULES: dict[str, dict[str, str]] = {
    TOOL_SEARCH: {"query": r"(?s).{2,120}"},
    TOOL_FAQ: {"question_id": r"EC-FAQ-\d{2}"},
    TOOL_DEADLINE: {
        "sign_date": r"\d{4}-\d{2}-\d{2}",
        "window_days": r"[1-9]\d?",
        "basis_doc": r"DOC-02-\d{2}-[\w一-鿿-]+\.md",
    },
}

OBSERVE_CHARS = 900
TOP_K = 3


class ToolRejected(ValueError):
    """模型请求了未授权的工具，或提交了不符合接口约定的参数。"""


# ---------------------------------------------------------------- ① 校验


def authorize_tool(name: str, arguments: Any) -> dict[str, str]:
    """输入工具名与参数；返回经白名单、字段与格式校验后的干净参数。

    这一步是"模型与数据之间"的那道门：模型只能从预先开放的只读能力里挑，
    给不出路径、给不出 SQL、给不出任意代码。
    """
    if not isinstance(name, str) or name not in ALLOWED_TOOLS:
        raise ToolRejected(f"工具未授权：{name!r}；只有 {sorted(ALLOWED_TOOLS)} 可以调用")
    if not isinstance(arguments, dict):
        raise ToolRejected("arguments 必须是 JSON 对象")
    rules = ARGUMENT_RULES[name]
    if set(arguments) != set(rules):
        raise ToolRejected(
            f"{name} 只接受参数：{'、'.join(rules)}；实际给了：{'、'.join(map(str, arguments))}"
        )
    clean: dict[str, str] = {}
    for field, pattern in rules.items():
        value = arguments[field]
        # JSON 里没有"字符串"这一说，模型给数字 7 和给 "7" 是一回事：
        # 先按类型放行（bool 除外），再统一转成字符串去撞正则——
        # 宽的是类型，不是值：格式约束一步没松，多一个字都会被拒。
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ToolRejected(
                f"{name}.{field} 类型不正确：{type(value).__name__}；只接受字符串或整数"
            )
        text = str(value).strip()
        if not re.fullmatch(pattern, text):
            raise ToolRejected(f"{name}.{field} 格式不正确：{value!r}")
        clean[field] = text
    return clean


# ---------------------------------------------------------------- ② 工具


def search_knowledge(
    retriever: RetrievalService, project_id: str, query: str, top_k: int = TOP_K
) -> dict:
    """输入业务问题；返回本项目资料里的命中片段及来源（Chroma 端按 project_id 过滤）。"""
    hits = retriever.search(query, project_id, top_k=top_k)
    return {
        "storage": "Chroma（向量检索）",
        "query": query,
        "hits": [
            {"source": hit.source, "text": " ".join(hit.text.split())[:OBSERVE_CHARS]}
            for hit in hits
        ],
    }


def lookup_faq(db_path: Path, faq_file: Path, question_id: str) -> dict:
    """输入 FAQ 编号；用固定参数化 SQL 从只读库取出标准答复与来源编号。"""
    row = fetch_faq(db_path, question_id)
    if row is None:
        # 查不到就是查不到，不返回"看起来像答案"的东西。
        return {
            "storage": "SQLite（结构化查询）",
            "source": faq_file.name,
            "question_id": question_id,
            "status": "未找到",
        }
    return {
        "storage": "SQLite（结构化查询）",
        "source": faq_file.name,
        "question_id": row["question_id"],
        "question": row["question"],
        "answer": row["answer"],
        "source_id": row["source_id"],
    }


def compute_return_deadline(knowledge_dir: Path, basis_doc: str, sign_date: str, window_days: str) -> dict:
    """输入签收日期与退货时限；由 Python 算出退货评估截止日期。

    两个硬约束，缺一个就不算：
    1. `basis_doc` 必须是本组真实存在的资料文件（路径在注册时固定，只接受文件名）；
    2. 资料原文里必须真的写着"N天"——天数来自记录，不来自模型。
    """
    doc_path = Path(knowledge_dir) / basis_doc
    if not doc_path.is_file():
        raise ToolRejected(f"计算依据不在本组资料中：{basis_doc}")
    text = "".join(doc_path.read_text(encoding="utf-8").split())
    if f"{window_days}天" not in text:
        raise ToolRejected(
            f"{basis_doc} 里没有“{window_days}天”这个时限；计算参数必须来自已查到的记录"
        )
    try:
        start = date.fromisoformat(sign_date)
    except ValueError as exc:
        raise ToolRejected(f"签收日期不合法：{sign_date}") from exc
    deadline = start + timedelta(days=int(window_days))
    marker = text.find(f"{window_days}天")
    return {
        "storage": "Python 计算（日期加法）",
        "sign_date": start.isoformat(),
        "window_days": window_days,
        "deadline": deadline.isoformat(),
        "basis_doc": basis_doc,
        "basis_text": text[max(0, marker - 8): marker + len(window_days) + 1],
    }


# ---------------------------------------------------------------- ③ 注册


@dataclass(frozen=True)
class ToolSpec:
    """统一的工具契约：名称、用途、参数说明、执行入口。"""

    name: str
    description: str
    parameters: dict[str, str]
    execute: Callable[..., dict]


class ToolRegistry:
    """模型看得见的清单 = 程序肯执行的清单。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册：{spec.name}")
        if spec.name not in ALLOWED_TOOLS:
            raise ValueError(f"工具不在白名单内，不能注册：{spec.name}")
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._tools)

    def descriptions(self) -> str:
        """输出给模型看的接口说明：只讲用途与参数，**不暴露任何本地路径**。"""
        return "\n".join(
            f"- {tool.name}("
            + ", ".join(f"{name}: {meaning}" for name, meaning in tool.parameters.items())
            + f"): {tool.description}"
            for tool in self._tools.values()
        )

    def spec(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ToolRejected(f"工具未授权：{name!r}")
        return self._tools[name]

    def call(self, name: str, arguments: Any) -> dict:
        """先校验动作与参数，再调度到已注册的真实业务函数。"""
        clean = authorize_tool(name, arguments)
        return self._tools[name].execute(**clean)


def build_registry(
    retriever: RetrievalService,
    project_id: str,
    faq_file: Path,
    db_path: Path,
    knowledge_dir: Path,
    top_k: int = TOP_K,
) -> ToolRegistry:
    """输入本组检索服务、项目标识、原始 FAQ、SQLite 库与资料目录；返回工具集。"""
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            TOOL_SEARCH,
            "检索本组制度或流程资料，返回片段和文件来源",
            {"query": "制度或流程问题，2—120 字"},
            lambda query: search_knowledge(retriever, project_id, query, top_k),
        )
    )
    registry.register(
        ToolSpec(
            TOOL_FAQ,
            "按本组 FAQ 编号从只读 SQLite 查标准答复与来源编号",
            {"question_id": "本组 FAQ 编号，形如 EC-FAQ-01"},
            lambda question_id: lookup_faq(db_path, faq_file, question_id),
        )
    )
    registry.register(
        ToolSpec(
            TOOL_DEADLINE,
            "按签收日期与资料写明的退货时限，算出退货评估截止日期",
            {
                "sign_date": "签收日期，形如 2026-12-01",
                "window_days": "退货时限天数，必须与资料原文一致",
                "basis_doc": "时限出自哪份资料文件名",
            },
            lambda sign_date, window_days, basis_doc: compute_return_deadline(
                knowledge_dir, basis_doc, sign_date, window_days
            ),
        )
    )
    return registry


# ---------------------------------------------------------------- ④ 渲染


def render_tool_list(registry: ToolRegistry) -> str:
    """第5位成员要交的工具清单：接口四要素 + 真实数据来源 + 来源口径。"""
    lines = [
        "工具清单（模型看到的接口 = 程序可执行的清单）",
        "工具\t参数\t数据来源\t返回来源",
    ]
    sources = {
        TOOL_SEARCH: ("Chroma 向量库", "命中的资料文件名"),
        TOOL_FAQ: ("SQLite 只读库", "常见售后问题.csv（及行内来源编号）"),
        TOOL_DEADLINE: ("资料原文 + 日期加法", "时限出自的资料文件名"),
    }
    for name in registry.names():
        spec = registry.spec(name)
        origin, back = sources.get(name, ("—", "—"))
        lines.append(
            f"{spec.name}\t"
            + ", ".join(f"{key}" for key in spec.parameters)
            + f"\t{origin}\t{back}"
        )
    lines.append("")
    lines.append("接口四要素逐项对应：")
    for name in registry.names():
        spec = registry.spec(name)
        lines.append(
            f"  {spec.name}：name={spec.name}；description={spec.description}；"
            f"parameters={list(spec.parameters)}；"
            f"execute={getattr(spec.execute, '__name__', 'lambda')}"
        )
    lines.append(
        "模型只拿到 name/description/parameters；execute 是 Python 里的函数，"
        "路径与 SQL 都写在注册时，模型碰不到。"
    )
    return "\n".join(lines)


def render_rejections(registry: ToolRegistry) -> str:
    """拒绝路径的兜底演示：直接喂坏输入给注册表（不经模型，结果可复现）。"""
    cases = [
        ("未知工具", "delete_collection", {"name": "faq"}),
        ("未知工具", "drop_all", {}),
        ("参数缺字段", TOOL_FAQ, {}),
        ("参数格式错", TOOL_FAQ, {"question_id": "123"}),
        ("参数格式错", TOOL_FAQ, {"question_id": "EC-FAQ-01; DROP TABLE faq"}),
        ("参数多给了", TOOL_SEARCH, {"query": "退货时限", "project_id": "other"}),
        ("日期不合法", TOOL_DEADLINE,
         {"sign_date": "2026-13-40", "window_days": "7", "basis_doc": "DOC-02-01-普通商品退货规则.md"}),
        ("天数不在资料里", TOOL_DEADLINE,
         {"sign_date": "2026-12-01", "window_days": "30", "basis_doc": "DOC-02-01-普通商品退货规则.md"}),
    ]
    lines = ["拒绝路径（直接调用注册表，模型没参与）：", "输入\t结果"]
    for label, name, arguments in cases:
        try:
            registry.call(name, arguments)
            outcome = "竟然通过了（异常！）"
        except (ToolRejected, FileNotFoundError) as error:
            outcome = f"已拒绝：{error}"
        shown = ", ".join(f"{key}={value}" for key, value in arguments.items()) or "（空参数）"
        lines.append(f"{label}｜{name}({shown})\t{outcome}")
    return "\n".join(lines)


__all__ = [
    "ALLOWED_TOOLS",
    "ARGUMENT_RULES",
    "FINISH_TOOL",
    "TOOL_DEADLINE",
    "TOOL_FAQ",
    "TOOL_SEARCH",
    "TOP_K",
    "ToolRejected",
    "ToolRegistry",
    "ToolSpec",
    "authorize_tool",
    "build_registry",
    "compute_return_deadline",
    "lookup_faq",
    "render_rejections",
    "render_tool_list",
    "search_knowledge",
]
