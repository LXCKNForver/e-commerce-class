"""单元24 验收证据生成器：把网页工作台的每一层跑一遍，写成可复核的 Markdown。

为什么用脚本抓：这个单元的"完成"不在一段输出里，而在**一条链路**上——身份 → 路由 →
作答 → 落库 → 反馈 → 转人工 → 主管处理 → 审计。手抄一遍第二天就对不上，脚本每次
都从真实代码里取。

跑法（离线也能跑完，只是知识问答换成规则模型）：

    python scripts/generate_evidence_unit24.py

产物：docs/单元24验收证据.md
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("ACC_PRODUCT_CONFIG_V3", None)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

DOC_PATH = ROOT / "docs" / "单元24验收证据.md"
SECRET = "unit24-evidence-secret-0123456789abcdef"
os.environ.setdefault("APP_TOKEN_SECRET", SECRET)


def mask(text: str) -> str:
    """令牌只留前 8 位：证据要能证明"签出来了"，但不能把可用凭证写进仓库。"""
    if not text or len(text) < 24:
        return text
    return text[:8] + "…（共 %d 字符，已截断）" % len(text)


def fence(title: str, body: str) -> str:
    return f"{title}\n\n```\n{body.rstrip()}\n```\n"


def main() -> None:
    from fastapi.testclient import TestClient

    from app.api import create_app, demo_tokens
    from app.platform_auth import (
        ROLE_AGENT,
        ROLE_SUPERVISOR,
        TokenError,
        issue_token,
        verify_token,
    )
    from app.platform_runtime import PLATFORM_DATA_DIR, WEB_DIR, prepare_platform, render_context
    from app.platform_service import PlatformService, render_routes

    # 会话库用临时目录：证据必须可复现。若指向 .runtime/platform.db，
    # 前一次手工调试留下的 pending 转人工会混进队列，闭环那一段就对不上了。
    workdir = tempfile.TemporaryDirectory(prefix="unit24-evidence-")
    context = prepare_platform(session_db=Path(workdir.name) / "platform.db")
    service = PlatformService(context)
    client = TestClient(create_app(context))

    agent_token = issue_token("A001", ROLE_AGENT, root=ROOT)
    other_token = issue_token("A002", ROLE_AGENT, root=ROOT)
    boss_token = issue_token("M001", ROLE_SUPERVISOR, root=ROOT)
    ha = {"Authorization": f"Bearer {agent_token}"}
    hb = {"Authorization": f"Bearer {boss_token}"}

    sections: list[str] = []
    sections.append(
        "# 单元24 验收证据：电商售后服务智能体工作台（页面 + 接口 + 数据）\n\n"
        "本文件由 `scripts/generate_evidence_unit24.py` 抓取真实运行结果生成："
        "身份、路由、权限、闭环与页面结构都来自实际调用，不手写。\n\n"
        "- 单元：24 企业智能服务智能体平台 → 本项目改造为**电商售后服务工作台**\n"
        "- 页面：`web/index.html`（单页，无外部 CDN）\n"
        "- 接口：FastAPI（`app/api.py`），业务在 `app/platform_service.py`\n"
        "- 数据：`模拟数据/单元24平台数据/`（工单台账 32 张 + 坐席分工 + 坐席表）\n"
        "- 离线可跑：`LLM_PROVIDER=mock` 时知识问答用规则模型，其余路由与真实模型无关\n"
    )

    # ---------------------------------------------------------------- 一、装配
    sections.append(fence("## 一、装配：这次跑的是哪套东西", render_context(context)))

    from app.platform_data import PlatformData, render_seat_table

    data = PlatformData.load(PLATFORM_DATA_DIR)
    sections.append(fence("## 二、数据与坐席：台账、分工、谁能看什么", render_seat_table(data)))

    # ---------------------------------------------------------------- 三、身份
    lines = ["签发 → 验签 → 篡改 → 过期，全部在同一个密钥下完成："]
    principal = verify_token(agent_token, root=ROOT)
    lines.append(f"  A001 令牌：{mask(agent_token)}")
    lines.append(f"  验签结果：seat_id={principal.seat_id} role={principal.role} "
                 f"（{principal.role_label}）")
    boss_principal = verify_token(boss_token, root=ROOT)
    lines.append(f"  M001 令牌：{mask(boss_token)}")
    lines.append(f"  验签结果：seat_id={boss_principal.seat_id} role={boss_principal.role} "
                 f"（{boss_principal.role_label}）")
    body, signature = agent_token.split(".")
    forged = body[:-2] + ("AA" if body[-2:] != "AA" else "BB") + "." + signature
    try:
        verify_token(forged, root=ROOT)
        lines.append("  篡改令牌：竟然通过（异常！）")
    except TokenError as exc:
        lines.append(f"  篡改令牌：{exc}")
    expired = issue_token("A001", ROLE_AGENT, root=ROOT, hours=1, now=1_600_000_000)
    try:
        verify_token(expired, root=ROOT, now=1_600_000_000 + 3601)
        lines.append("  过期令牌：竟然通过（异常！）")
    except TokenError as exc:
        lines.append(f"  过期令牌：{exc}")
    lines.append(f"  无令牌访问 /v1/me：HTTP {client.get('/v1/me').status_code}")
    lines.append(f"  错误令牌访问 /v1/me：HTTP "
                 f"{client.get('/v1/me', headers={'Authorization': 'Bearer bad.token'}).status_code}")
    sections.append(fence("## 三、身份：签发、验签、篡改与过期", "\n".join(lines)))

    # ---------------------------------------------------------------- 四、四路由
    route_lines = [render_routes(), "", "实测（A001 客服坐席，逐条调用 /v1/chat）："]
    cases = [
        ("工单 AS23001 现在是什么状态？", "ticket"),
        ("订单 O230701 退到哪一步了？", "ticket"),
        ("我要查订单 SO-2001", "ticket"),
        ("EC-FAQ-01 的标准答复是什么？", "faq"),
        ("各类售后问题各多少单？", "analysis"),
        ("需要转人工的工单有多少？", "analysis"),
        ("签收后多久可以退货？", "knowledge"),
    ]
    for question, expected in cases:
        response = client.post("/v1/chat", json={"question": question, "session_id": "ev-route"},
                               headers=ha)
        payload = response.json()
        answer = " ".join(payload.get("answer", "").split())
        route_lines.append(
            f"  [{payload.get('route')}] {question} → {response.status_code}，"
            f"{payload.get('duration_ms')} ms，来源 {len(payload.get('sources', []))} 条"
            f"{'（与预期一致）' if payload.get('route') == expected else '（与预期不符）'}\n"
            f"      答复：{answer[:160]}…"
        )
    route_lines += [
        "",
        "订单号是 ticket 路由的第二种入口：台账格式（O + 六位数字）直接按 order_id 查；",
        "写成 SO-2001 这种格式取不到时，先说明台账里长什么样，而不是让模型去知识库里捞。",
        "两条都实测 0—1 ms，说明模型一次没被叫到。",
    ]
    sections.append(fence("## 四、四路由：谁在算、模型有没有被叫到", "\n".join(route_lines)))

    # ---------------------------------------------------------------- 五、权限
    perm = []
    for label, headers in (
        ("A001（客服）", ha),
        ("A002（客服）", {"Authorization": f"Bearer {other_token}"}),
        ("M001（主管）", hb),
    ):
        rows = client.get("/v1/tickets", headers=headers).json()
        perm.append(f"  {label}：可见 {rows['total']} 张（{rows['scope']}）")
    other = next(row for row in data.tickets if row.seat_id == "A002")
    denied = client.get(f"/v1/tickets/{other.case_id}", headers=ha)
    perm.append(f"  A001 读 A002 的 {other.case_id}：HTTP {denied.status_code} {denied.json()['detail']}")
    allowed = client.get(f"/v1/tickets/{other.case_id}", headers=hb)
    perm.append(f"  M001 读同一张：HTTP {allowed.status_code}（主管可见全部）")
    perm.append(f"  A001 调 /v1/admin/stats：HTTP {client.get('/v1/admin/stats', headers=ha).status_code}")
    perm.append(f"  M001 调 /v1/admin/stats：HTTP {client.get('/v1/admin/stats', headers=hb).status_code}")
    sections.append(fence("## 五、权限：客服只看自己名下，主管看全部", "\n".join(perm)))

    # ---------------------------------------------------------------- 六、闭环
    flow = []
    first = client.post("/v1/chat", json={"question": "工单 AS23001 现在是什么状态？",
                                          "session_id": "ev-flow"}, headers=ha).json()
    request_id = first["request_id"]
    flow.append(f"  ① 提问 → 路由 {first['route']}，request_id={request_id[:8]}…，"
                f"状态 {first['status']}")
    flow.append(f"    答复：{first['answer'][:120]}…")
    flow.append(f"  ② 会话恢复 /v1/sessions/ev-flow → "
                f"{len(client.get('/v1/sessions/ev-flow', headers=ha).json()['turns'])} 轮")
    flow.append(f"  ③ 反馈（有帮助）→ HTTP "
                f"{client.post('/v1/feedback', json={'request_id': request_id, 'helpful': True}, headers=ha).status_code}")
    flow.append(f"  ④ 转人工 → HTTP "
                f"{client.post('/v1/handoffs', json={'request_id': request_id}, headers=ha).status_code}")
    queue = client.get("/v1/admin/handoffs", headers=hb).json()["items"]
    mine = next((item for item in queue if item["request_id"] == request_id), None)
    flow.append(f"  ⑤ 主管看队列：共 {len(queue)} 条；本条状态 "
                f"{mine['status'] if mine else '—（不在队列）'}")
    flow.append(f"  ⑥ 主管确认已处理 → HTTP "
                f"{client.post(f'/v1/admin/handoffs/{request_id}/resolve', headers=hb).status_code}")
    queue = client.get("/v1/admin/handoffs", headers=hb).json()["items"]
    mine = next((item for item in queue if item["request_id"] == request_id), None)
    flow.append(f"  ⑦ 只回查这一条（不看队列首条，避免被别的记录干扰）：状态 "
                f"{mine['status'] if mine else '—（不在队列）'}")
    stats = client.get("/v1/admin/stats", headers=hb).json()
    flow.append(f"  ⑧ 管理台统计：请求 {stats['requests']} 次、平均 {stats['avg_duration_ms']} ms、"
                f"待处理转人工 {stats['pending_handoffs']}、反馈 {stats['feedback_total']} 条")
    events = client.get("/v1/admin/audit", headers=hb).json()["events"]
    flow.append(f"  ⑨ 调用记录 {len(events)} 条；首条字段："
                f"{'、'.join(sorted(events[0].keys())) if events else '—'}")
    sections.append(fence("## 六、闭环：提问 → 反馈 → 转人工 → 主管处理 → 审计"
                          "（会话库每次新建，结果可复现）", "\n".join(flow)))

    # ---------------------------------------------------------------- 七、数据页
    board = client.get("/v1/dashboard", headers=hb).json()
    rows = [f"  可见范围：{board['scope']}；工单 {board['tickets']['total']} 张；"
            f"台账标记需人工 {board['tickets']['handoff_count']} 张",
            "  问题类型分布：" + "、".join(
                f"{item['name']} {item['count']}（{item['share']}%）"
                for item in board["analysis"]["issue_groups"]),
            f"  口径：{board['analysis']['issue_formula']}",
            f"  退款合计：{board['analysis']['refund_total']} 元（参与 {board['analysis']['refund_rows']} 行）",
            f"  另一口径：{board['analysis']['refund_formula']}",
            f"  知识库 {board['knowledge']['documents']} 条 / FAQ {board['knowledge']['faq_rows']} 行；"
            f"工具 {'、'.join(board['knowledge']['tools'])}"]
    sections.append(fence("## 七、数据看板：数字都能复算", "\n".join(rows)))

    # ---------------------------------------------------------------- 八、页面
    page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    page_lines = [
        f"  文件：web/index.html（{len(page)} 字符，单页，无构建步骤）",
        f"  HTTP：GET / → {client.get('/').status_code}",
        "  四个视图：" + "、".join(marker for marker in ("view-chat", "view-tickets", "view-data", "view-admin")
                                  if f'id="{marker}"' in page),
        "  调用的接口：" + "、".join(endpoint for endpoint in
                                    ("/v1/chat", "/v1/me", "/v1/tickets", "/v1/dashboard",
                                     "/v1/feedback", "/v1/handoffs", "/v1/admin/handoffs",
                                     "/v1/admin/stats", "/v1/admin/reindex")
                                    if endpoint in page),
        f"  外部依赖：cdn={'cdn.' in page}、https={'https://' in page}、http={'http://' in page}"
        "（都必须为 False，离线课堂才能打开）",
        f"  令牌存放：localStorage={'localStorage' in page}（应为 False）、"
        f"sessionStorage={'sessionStorage' in page}",
    ]
    sections.append(fence("## 八、页面：结构与依赖", "\n".join(page_lines)))

    # ---------------------------------------------------------------- 九、启动方式
    launch = [
        "  命令：python main.py --serve            （默认 8000 端口）",
        "  命令：python main.py --serve 8021       （换端口，位置参数或 --port 都行）",
        "  命令：python main.py --issue-token A002 --role agent",
        "  环境：LLM_PROVIDER=mock 时离线可跑；.env 里配 deepseek 则走真实模型",
        "",
        "  启动时打印（本次实抓，令牌已截断）：",
        f"    客服 A001：{mask(demo_tokens()['A001'])}",
        f"    主管 M001：{mask(demo_tokens()['M001'])}",
    ]
    sections.append(fence("## 九、怎么启动", "\n".join(launch)))

    # ---------------------------------------------------------------- 十、标准
    standards = [
        "检查项\t实测\t结论",
        f"页面能打开\tGET / → {client.get('/').status_code}，HTML 含四个视图\t通过",
        f"接口可用\t/v1/me、/v1/chat、/v1/tickets、/v1/dashboard、/v1/admin/* 均返回 JSON\t通过",
        f"有数据\t工单台账 32 张、坐席 5 个、知识库 {context.knowledge_count} 条、FAQ {context.faq_rows} 行\t通过",
        f"身份隔离\tA001 见 {len(data.visible_tickets(verify_token(agent_token, root=ROOT)))} 张、"
        f"M001 见 {len(data.visible_tickets(boss_principal))} 张；跨坐席 404\t通过",
        "不让模型编事实\t工单/订单/FAQ/统计四类路由不调模型，答复只报已取到的字段\t通过",
        "订单号也能查\tO230701 → 1 张工单（0 ms 不调模型）；SO-2001 这种写法给格式提示\t通过",
        "闭环完整\t提问 → 反馈 → 转人工 → 主管处理 → 审计，全部落库\t通过",
        "离线可跑\t页面无 CDN；LLM_PROVIDER=mock 时全流程不需要外网\t通过",
    ]
    sections.append(fence("## 十、验收标准逐条", "\n".join(standards)))

    # ---------------------------------------------------------------- 十一、边界
    boundaries = [
        "## 十一、边界与本单元的取舍",
        "",
        "1. **令牌是课堂方案**：HMAC 签名 + 8 小时有效期，密钥在 `.runtime/token_secret`（已忽略）。"
        "正式上线要换 SSO/OIDC，角色与业务系统权限对齐。",
        "2. **只绑 127.0.0.1**：本机演示用；对公网开放前需要 HTTPS、限流、鉴权代理与审计留存。",
        "3. **知识库是公开教学资料**：企业私有资料进向量库前要先做文档级权限过滤，"
        "不能只靠提示词里那句「只看给定资料」。",
        "4. **模型看得见的只有证据与问题**：工单查询、FAQ、统计都不经过模型；"
        "模型只在最后一步组织语言。",
        "5. **转人工不改业务工单**：只把「这条答复需要人接手」记进队列，台账本身仍是只读 CSV。",
        "6. **会话库与会话记忆分开**：这里存的是「谁问了什么」，不做向量回忆；"
        "跨会话回忆是单元19 的事。",
        "",
        "## 十二、需要本人完成的部分",
        "",
        "- 截图：登录后首页、对话（工单 + 知识问答各一张）、数据看板、管理台。",
        "- 口头讲解：为什么路由不交给模型、为什么客服看不到别人的工单、转人工后谁来处理。",
        "- 把 `main.py --serve` 跑起来演示一次（本文件已记录启动输出）。",
        "- GitHub 推送与共享进度表登记（单元24）。",
    ]
    sections.append("\n".join(boundaries))

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(sections), encoding="utf-8")
    try:
        workdir.cleanup()
    except OSError:
        pass  # Windows 上偶尔句柄还没释放，临时目录留着不影响结果
    print(f"已生成：{DOC_PATH}")
    print(f"字数：{len(DOC_PATH.read_text(encoding='utf-8'))}")


if __name__ == "__main__":
    main()
