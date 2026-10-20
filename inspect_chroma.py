"""单元07 查看：只读观察已创建的 Chroma 集合，不操作业务记录。

    uv run --extra chroma python inspect_chroma.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from setup_chroma import CHROMA_PATH, COLLECTION_NAME  # noqa: E402


def inspect_collection(database_path: Path = CHROMA_PATH) -> dict:
    """通过 Chroma 接口读取集合信息；不直接查 SQLite 内部表，也不写数据。"""
    import chromadb

    database_file = database_path / "chroma.sqlite3"
    if not database_file.is_file():
        raise FileNotFoundError(f"请先运行 setup_chroma.py：{database_file}")

    client = chromadb.PersistentClient(path=str(database_path))
    # get_collection 只取已有集合，名字写错会直接报错，不会悄悄新建一个。
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=None)
    return {
        "chroma_version": chromadb.__version__,
        "database_path": str(database_path),
        "name": collection.name,
        "collection_id": collection.id,
        "metadata": collection.metadata,
        "count": collection.count(),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    info = inspect_collection()
    print(f"Chroma 版本：{info['chroma_version']}")
    print(f"数据库目录：{info['database_path']}")
    print(f"集合名称：{info['name']}")
    print(f"集合 ID：{info['collection_id']}")
    print(f"集合元数据：{info['metadata']}")
    print(f"当前记录数：{info['count']}")
    print("本单元只验证空集合；第八单元再写入和查询文档。")


if __name__ == "__main__":
    main()
