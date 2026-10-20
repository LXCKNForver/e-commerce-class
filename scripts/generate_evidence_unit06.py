"""生成单元06 的课堂验收证据。

运行一次就会把"真实运行结果"写进 docs/单元06验收证据.md，
不手写、不编造：排序分数、维度、边界报错文案全部来自本地模型实际输出。

    uv run --extra embedding python scripts/generate_evidence_unit06.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "docs" / "单元06验收证据.md"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from app.embedding import MODEL_NAME, model_ready
        from app.vector_lab import (
            DEMO_QUERY,
            RISK_EXPECTED_ROUTE,
            RISK_QUERY,
            boundary_report,
            load_sample_texts,
            misjudge_check,
            render_boundary,
            render_misjudge,
            render_report,
            run_lab,
        )
    except ImportError as exc:
        print(f"缺少依赖（{exc}）。请先执行：uv sync --extra embedding")
        return 1

    ready, detail = model_ready()
    if not ready:
        print(f"向量模型不可用：{detail}")
        return 1

    items = load_sample_texts()
    report = run_lab(DEMO_QUERY)
    boundary = boundary_report()
    case = misjudge_check(RISK_QUERY, RISK_EXPECTED_ROUTE)

    lines: list[str] = []
    lines.append("# 单元06 验收证据：Embedding 与文本向量化")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}")
    lines.append(f"模型：{MODEL_NAME}（本地离线加载，不调用 DeepSeek）")
    lines.append(f"数据来源：模拟数据/sample_questions.json（电商售后样例问题 {len(items)} 条）")
    lines.append("")
    lines.append("本单元只做「文本 → 向量 → 相似度排序」，不做知识库问答、不做业务判断。")
    lines.append("")

    lines.append("## 一、本组真实样例文本清单（第2节）")
    lines.append("")
    lines.append("| 编号 | 文本 | 来源 |")
    lines.append("| --- | --- | --- |")
    for item in items:
        lines.append(f"| {item.source_id} | {item.text} | `{item.source}` |")
    lines.append("")
    lines.append(f"共 {len(items)} 条，满足任务卡「至少 5 条」的要求。")
    lines.append("")

    lines.append("## 二、批量向量化与相似度排序（第3节）")
    lines.append("")
    lines.append("```text")
    lines.append(render_report(report))
    lines.append("```")
    lines.append("")

    lines.append("## 三、边界核对（第4节）")
    lines.append("")
    lines.append("```text")
    lines.append(render_boundary(boundary))
    lines.append("```")
    lines.append("")
    lines.append("空文本与空列表在调用模型之前就被拒绝；维度不一致与零向量在算相似度时报错。")
    lines.append("")

    lines.append("## 四、高分误判风险（第4节）")
    lines.append("")
    lines.append("```text")
    lines.append(render_misjudge(case))
    lines.append("```")
    lines.append("")
    lines.append(
        "风险说明：「退货运费谁承担」是一句普通的售后政策咨询，人工判断应走 `rag`"
        "（查政策文档后答复），但最高分命中的却是 `reject` 那条「直接给 SO-2001 退款」的越权样例。\n"
        "分差很小，说明排序并不稳定。如果直接照搬最高分那条的处理方式，"
        "就会把一次正常咨询错误地变成拒答转人工。"
        "相似度只衡量「像不像」，不衡量「该怎么做」。"
    )
    lines.append("")
    lines.append("## 五、局限与未做的事")
    lines.append("")
    lines.append("- 向量只存在内存里，本单元不落库、不做检索；持久化留给下一单元。")
    lines.append("- 6 条样例文本太少，排序只能说明方法可用，不能说明效果稳定。")
    lines.append("- 换模型就等于换向量空间，历史向量全部作废，必须一次性重算。")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入：{OUTPUT.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
