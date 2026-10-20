"""单元06 课堂实验：本组真实样例文本 -> 向量 -> 相似度排序。

配合任务卡四节课的节奏：
  第2节 load_sample_texts()  拿到带编号和来源的真实文本清单
  第3节 run_lab()            用同一模型批量向量化，自拟问题排序取前三
  第4节 boundary_report()    核对数量、维度、空文本，misjudge_check() 说清高分误判风险

本模块只读 `模拟数据/sample_questions.json`，不碰订单库、不读知识库、不做问答。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.data_store import DATA_DIR
from app.embedding import MODEL_NAME, cosine_similarity, embed_many, resolve_model_dir

SAMPLE_FILE = DATA_DIR / "sample_questions.json"
MIN_TEXTS = 5

# 自拟问题：本组电商售后场景里真实会出现的一句话。
DEMO_QUERY = "我买的商品不满意，想退货，需要满足什么条件？"
# 高分误判演示：一句普通的政策咨询，最高分却命中"直接退款"这条越权拒答样例。
# 人工判断它应走的路由是 rag（查政策文档后答复），而不是 reject（拒答转人工）。
RISK_QUERY = "退货运费谁承担？"
RISK_EXPECTED_ROUTE = "rag"

DISCLAIMER = "相似度只表示向量方向接近，不代表答案正确，也不等于业务判断。"


@dataclass(frozen=True)
class SampleText:
    """一条样例文本：编号、正文、来源，以及它在课程数据里的路由标签。"""

    source_id: str
    text: str
    route: str
    source: str


def load_sample_texts(path: Path | None = None) -> list[SampleText]:
    """读取本组真实样例问题，要求至少 MIN_TEXTS 条且每条有编号和正文。"""
    target = path or SAMPLE_FILE
    if not target.is_file():
        raise FileNotFoundError(f"缺少样例文本数据：{target}")

    # utf-8-sig 兼容文件开头可能带 BOM 的情况。
    records = json.loads(target.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError("样例问题文件必须是列表")

    items: list[SampleText] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"第 {index} 条记录格式不正确")
        # 课程文件里编号字段叫 question_id，兼容 id 写法。
        source_id = str(record.get("question_id") or record.get("id") or "").strip()
        text = str(record.get("question") or record.get("text") or "").strip()
        if not source_id or not text:
            raise ValueError(f"第 {index} 条记录缺少编号或正文")
        items.append(
            SampleText(
                source_id=source_id,
                text=text,
                route=str(record.get("route", "")),
                source=f"模拟数据/sample_questions.json#{source_id}",
            )
        )

    if len(items) < MIN_TEXTS:
        raise ValueError(f"任务卡要求至少 {MIN_TEXTS} 条真实样例文本，当前只有 {len(items)} 条")
    return items


def run_lab(query: str, texts: list[SampleText] | None = None, top_n: int = 3) -> dict:
    """把自拟问题与所有样例文本一起向量化，按余弦相似度降序返回完整排序。"""
    items = texts or load_sample_texts()
    if not query or not query.strip():
        raise ValueError("自拟问题不能为空")

    # 查询句和候选句放进同一批，确保它们落在同一个向量空间里。
    vectors = embed_many([query, *[item.text for item in items]])
    query_vector, candidate_vectors = vectors[0], vectors[1:]

    scored = [
        (item, cosine_similarity(query_vector, vector))
        for item, vector in zip(items, candidate_vectors)
    ]
    ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)

    return {
        "model": MODEL_NAME,
        "model_dir": str(resolve_model_dir()),
        "query": query,
        "text_count": len(items),
        "vector_count": len(vectors),
        "dim": len(query_vector),
        "one_to_one": len(candidate_vectors) == len(items),
        "ranked": [
            {
                "source_id": item.source_id,
                "text": item.text,
                "route": item.route,
                "source": item.source,
                "score": round(score, 4),
            }
            for item, score in ranked
        ],
        "top": [
            {
                "source_id": item.source_id,
                "text": item.text,
                "route": item.route,
                "score": round(score, 4),
            }
            for item, score in ranked[:top_n]
        ],
    }


def render_report(report: dict) -> str:
    """把实验报告渲染成可直接贴进课堂证据的纯文本。"""
    lines = [
        f"模型：{report['model']}（本地离线加载：{report['model_dir']}）",
        f"自拟问题：{report['query']}",
        f"样例文本：{report['text_count']} 条；生成向量：{report['vector_count']} 条；"
        f"向量维度：{report['dim']}",
        f"数量核对：文本与向量一一对应 = {report['one_to_one']}",
        "",
        "相似度排名（降序）：",
    ]
    for rank, row in enumerate(report["ranked"], start=1):
        lines.append(
            f"{rank}. {row['source_id']}  相似度={row['score']:.4f}  "
            f"路由={row['route'] or '-'}  文本={row['text']}"
        )
    lines.append("")
    lines.append(f"前三条：{', '.join(row['source_id'] for row in report['top'])}")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def boundary_report() -> dict:
    """第4节边界核对：空文本、空列表、维度不一致、零向量、同一文本自比。"""
    checks: list[dict] = []

    def record(name: str, action) -> None:
        try:
            value = action()
            checks.append({"name": name, "ok": True, "detail": str(value)})
        except (ValueError, RuntimeError) as exc:
            checks.append({"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"})

    record("空列表输入", lambda: embed_many([]))
    record("含空白文本的一批", lambda: embed_many(["正常问题", "   "]))
    record("维度不一致求相似度", lambda: cosine_similarity([1.0], [1.0, 0.0]))
    record("零向量求相似度", lambda: cosine_similarity([0.0, 0.0], [1.0, 0.0]))
    record("同一向量自比", lambda: round(cosine_similarity([0.6, 0.8], [0.6, 0.8]), 6))

    return {"checks": checks}


def render_boundary(report: dict) -> str:
    lines = ["边界场景核对（全部为实际运行捕获的结果）："]
    for check in report["checks"]:
        flag = "通过" if check["ok"] else f"已拦截 -> {check['detail']}"
        lines.append(f"  - {check['name']}：{flag}")
    return "\n".join(lines)


def misjudge_check(query: str = RISK_QUERY, expected_route: str = RISK_EXPECTED_ROUTE) -> dict:
    """高分误判风险演示：看最高分那条的业务路由与预期是否一致。"""
    report = run_lab(query)
    top = report["ranked"][0]
    return {
        "query": query,
        "expected_route": expected_route,
        "source_id": top["source_id"],
        "text": top["text"],
        "top_route": top["route"],
        "score": top["score"],
        "matched": top["route"] == expected_route,
        "runner_up": report["ranked"][1] if len(report["ranked"]) > 1 else None,
    }


def render_misjudge(case: dict) -> str:
    lines = [
        f"高分误判风险演示：自拟问题「{case['query']}」",
        f"  最高分命中：{case['source_id']} 相似度={case['score']} 文本={case['text']}",
        f"  该条在课程数据里的路由：{case['top_route'] or '-'}；"
        f"人工判断这条问题应走的路由：{case['expected_route'] or '-'}",
    ]
    if case["matched"]:
        lines.append("  结论：最高分与人工判断一致，本次没有出现误判。")
    else:
        lines.append(
            f"  结论：最高分那条的处理方式是「{case['top_route'] or '-'}」，"
            f"而这句话真正需要的是「{case['expected_route'] or '-'}」——"
            "照搬最高分就会答错。分数只说明两句话像，不说明该怎么处理。"
        )
    return "\n".join(lines)
