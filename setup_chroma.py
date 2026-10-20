"""单元07 建库：用真实 Chroma 创建首个向量集合（本组电商售后知识库）。

本单元只完成环境与集合验证：建空集合、记录 ID 与元数据。
第八单元再讲记录写入、查询与更新。

    uv run --extra chroma python setup_chroma.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 运行后生成的数据库文件属于本单元，不进 Git，也不移动课程已有数据目录。
CHROMA_PATH = PROJECT_ROOT / ".chroma_unit7"
COLLECTION_NAME = "after_sale_knowledge_base"
# 与 app/embedding.py 保持一致；集合元数据里写清楚，便于以后核对。
COLLECTION_METADATA = {
    "purpose": "ecommerce_after_sale_knowledge_base",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "dimension": 512,
    "content_source": "模拟数据/knowledge/ + 模拟数据/sample_questions.json",
    "write_policy": "read_only_sample_data",
}


def create_first_collection(database_path: Path = CHROMA_PATH):
    """连接本地持久化库，重复运行时复用同名集合。

    PersistentClient 把数据落在指定目录。集合暂时为空，不传默认嵌入函数，
    因此不会触发任何模型下载；第八单元写入时再显式接单元06 的中文 Embedding。
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(database_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata=COLLECTION_METADATA,
    )
    return client, collection


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import chromadb

    _, collection = create_first_collection()
    print(f"Chroma 版本：{chromadb.__version__}")
    print(f"持久化目录：{CHROMA_PATH}")
    print(f"首个集合：{collection.name}")
    print(f"集合 ID：{collection.id}")
    print(f"集合元数据：{collection.metadata}")
    print(f"当前记录数：{collection.count()}")
    print("重复运行应复用同一个集合 ID；本单元尚不写入售后知识文档。")


if __name__ == "__main__":
    main()
