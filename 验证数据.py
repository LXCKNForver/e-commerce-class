from pathlib import Path
import json

root = Path(__file__).parent / "模拟数据"
records = json.loads((root / "records.json").read_text(encoding="utf-8"))
questions = json.loads((root / "sample_questions.json").read_text(encoding="utf-8"))
docs = sorted((root / "knowledge").glob("*.md"))
assert records["fictional"] is True
assert len(docs) == records["knowledge_count"], (len(docs), records["knowledge_count"])
assert len(questions) >= 6
for key, rows in records["data"].items():
    if not isinstance(rows, list):
        continue
    ids = [row[next(k for k in row if k.endswith("_id"))] for row in rows]
    assert len(ids) == len(set(ids)), key
    print(f"{key}: {len(rows)}条，主键唯一")
print(f"知识文档: {len(docs)}份；样例问题: {len(questions)}条；检查通过")
