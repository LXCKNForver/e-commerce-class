"""单元24：把"资料 + 工具 + 模型 + 台账 + 会话库"装成一份运行上下文。

为什么要有这一层：网页、命令行、测试、验收脚本都要"能跑一次问答"，但它们不该各自
写一遍"建向量库 → 同步 FAQ → 注册工具 → 选模型"。装配只在这里发生一次，别处只接收
一个 `PlatformContext`——这样从命令行切到网页，业务逻辑一行都不用改。

四份状态分开放，谁也别污染谁：

    .chroma_unit24/        本组资料（可重建，重建不影响别的单元的演示库）
    .sqlite_unit24/faq.db  本组 FAQ（由 CSV 同步而来，只读 + 参数化查询）
    .runtime/platform.db   会话、反馈、转人工、审计（运行产物，不进仓库）
    模拟数据/单元24平台数据/ 工单台账与坐席分工（只读，改了要重跑）

**模型可以是离线的**：`build_llm()` 在 `LLM_PROVIDER=mock` 时返回规则模型，网页照样
能点通全流程，只是答复来自规则而不是 DeepSeek。验收证据里会写明这一轮用的是哪个。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.chroma_store import (
    UNIT24_COLLECTION,
    UNIT24_COLLECTION_METADATA,
    ChromaVectorStore,
    resolve_unit24_db_path,
)
from app.faq_db import count_faq, prepare_faq_db
from app.llm import build_llm
from app.loaders import load_materials
from app.parse_lab import PROJECTS
from app.platform_data import PlatformData
from app.platform_store import SessionStore
from app.retriever import RetrievalService
from app.toolkit import ToolRegistry, build_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
PLATFORM_DATA_DIR = PROJECT_ROOT / "模拟数据" / "单元24平台数据"
# 本项目固定用本组（电商售后）资料；unit12 起每个单元都这么取目录。
PROJECT_KEY = "ecom"
KNOWLEDGE_DIR = PROJECTS[PROJECT_KEY].folder
FAQ_FILE = KNOWLEDGE_DIR / "常见售后问题.csv"
PROJECT_ID = "e-commerce-after-sales"

DEFAULT_FAQ_DB = PROJECT_ROOT / ".sqlite_unit24" / "faq.db"
DEFAULT_SESSION_DB = PROJECT_ROOT / ".runtime" / "platform.db"


@dataclass
class PlatformContext:
    """一次装配的全部依赖；业务层只认这个对象。"""

    model: Any
    registry: ToolRegistry
    retriever: RetrievalService
    store: ChromaVectorStore
    data: PlatformData
    sessions: SessionStore
    knowledge_dir: Path
    faq_file: Path
    knowledge_count: int
    faq_rows: int
    project_id: str = PROJECT_ID

    @property
    def model_name(self) -> str:
        return type(self.model).__name__


def _ingest_knowledge(store: ChromaVectorStore, knowledge_dir: Path) -> int:
    """把本组资料整篇写进向量库；重复运行靠 document_id 覆盖，不追加。"""
    records, failures = load_materials(knowledge_dir, PROJECT_ID)
    if not records:
        raise RuntimeError(f"知识库资料为空：{knowledge_dir}；解析失败：{failures}")
    store.upsert_documents(records)
    return store.count(PROJECT_ID)


def prepare_platform(
    *,
    knowledge_db: Path | None = None,
    faq_db: Path | None = None,
    session_db: Path | None = None,
    data_dir: Path | None = None,
    knowledge_dir: Path | None = None,
    model: Any | None = None,
) -> PlatformContext:
    """装配一次；参数只是为了测试能指向临时目录，正常运行全部用默认值。"""
    knowledge_dir = Path(knowledge_dir or KNOWLEDGE_DIR)
    faq_file = knowledge_dir / FAQ_FILE.name
    data = PlatformData.load(Path(data_dir or PLATFORM_DATA_DIR))

    store = ChromaVectorStore(
        Path(knowledge_db) if knowledge_db else resolve_unit24_db_path(),
        UNIT24_COLLECTION,
        UNIT24_COLLECTION_METADATA,
    )
    count = _ingest_knowledge(store, knowledge_dir)

    faq_path = prepare_faq_db(faq_file, Path(faq_db) if faq_db else DEFAULT_FAQ_DB)
    faq_rows = count_faq(faq_path)

    registry = build_registry(
        RetrievalService(store), PROJECT_ID, faq_file, faq_path, knowledge_dir
    )
    sessions = SessionStore(Path(session_db) if session_db else DEFAULT_SESSION_DB)
    return PlatformContext(
        model=model if model is not None else build_llm(),
        registry=registry,
        retriever=RetrievalService(store),
        store=store,
        data=data,
        sessions=sessions,
        knowledge_dir=knowledge_dir,
        faq_file=faq_file,
        knowledge_count=count,
        faq_rows=faq_rows,
    )


def refresh_knowledge(context: PlatformContext) -> int:
    """管理员点"重建索引"时调用：重新解析资料目录并覆盖写入。"""
    context.knowledge_count = _ingest_knowledge(context.store, context.knowledge_dir)
    return context.knowledge_count


def _show_path(path: Path) -> str:
    """相对项目根显示；临时目录（验收脚本、测试）只显示末两级并标注"临时"。

    直接 `relative_to` 会在临时库上抛 ValueError——验收脚本故意用临时会话库，
    这里不能假定"所有路径都在仓库里"。
    """
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return f"{path.parent.name}/{path.name}（临时）"


def render_context(context: PlatformContext) -> str:
    """启动时打印的一份"这套东西装了什么"，出问题先看它。"""
    return "\n".join([
        "平台装配（单元24）",
        f"  模型：{context.model_name}（LLM_PROVIDER={_provider_hint()}）",
        f"  资料：{context.knowledge_count} 条（{_show_path(context.knowledge_dir)}）",
        f"  FAQ：{context.faq_rows} 行（由 常见售后问题.csv 同步）",
        f"  工单：{len(context.data.tickets)} 张（{PLATFORM_DATA_DIR.name}）",
        f"  工具：{'、'.join(context.registry.names())}",
        f"  会话库：{_show_path(context.sessions.path)}",
    ])


def _provider_hint() -> str:
    import os

    return os.getenv("LLM_PROVIDER", "mock")
