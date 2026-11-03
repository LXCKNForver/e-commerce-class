"""单元24：FastAPI 接口层——把业务能力变成网页能调的 HTTP 接口。

分层只用三句话说明白：

    接口层（本文件）   只做三件事：验身份、收参数、把业务异常翻译成状态码
    业务层（platform_service）  决定"谁来答、怎么答"
    数据层（platform_data / platform_store）  只认已经验过的身份，做读写

这样分层的好处很实在：换前端不用改业务，换数据库不用改接口，业务逻辑也能在没有
FastAPI 的环境里单测（`tests/test_unit24.py` 里大部分用例都不启服务）。

接口清单（页面用的就是这些）：
    GET  /                      工作台页面
    GET  /health                存活检查（不加载数据、不调模型）
    GET  /v1/meta               角色表、路由表、知识库规模（页面"关于"用）
    GET  /v1/me                 确认令牌身份
    POST /v1/chat               主流程：路由 → 作答 → 落库
    GET  /v1/sessions/{id}      当前会话最近几轮（跨坐席读不到）
    POST /v1/feedback           有帮助 / 没帮助
    POST /v1/handoffs           把本人这条答复转人工
    GET  /v1/tickets            工单列表（按身份收窄）
    GET  /v1/tickets/{case_id}  单张工单
    GET  /v1/dashboard          数据页（工单口径 + 分析口径）
    GET  /v1/admin/*            管理台：转人工队列、反馈、调用记录、统计、重建索引
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.platform_auth import (
    PermissionDenied,
    Principal,
    ROLE_AGENT,
    ROLE_SUPERVISOR,
    TokenError,
    can_view_all,
    issue_token,
    parse_bearer,
    render_roles,
    require_admin,
    verify_token,
)
from app.platform_data import TICKET_PATTERN
from app.platform_errors import BadRequest, NotFound
from app.platform_runtime import PROJECT_ROOT, WEB_DIR, PlatformContext, prepare_platform, refresh_knowledge
from app.platform_service import PlatformService, render_routes, validate_session_id

REQUEST_ID_PATTERN = r"^[0-9a-f]{32}$"
SESSION_ID_PATTERN = r"^[\w-]{1,80}$"
DEFAULT_PORT = 8000


class ChatRequest(BaseModel):
    """页面只提交问题与会话号；身份来自已验签的 Bearer 令牌。"""

    question: str = Field(min_length=1, max_length=500)
    session_id: str = Field(min_length=1, max_length=80, pattern=SESSION_ID_PATTERN)


class FeedbackRequest(BaseModel):
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)
    helpful: bool


class HandoffRequest(BaseModel):
    request_id: str = Field(pattern=REQUEST_ID_PATTERN)


class DemoLoginRequest(BaseModel):
    seat_id: str


@lru_cache(maxsize=1)
def get_platform() -> PlatformContext:
    """首次请求才装配（建向量库、同步 FAQ、选模型），`/health` 不会触发。"""
    return prepare_platform()


def get_service() -> PlatformService:
    return PlatformService(get_platform())


def current_principal(authorization: Annotated[str | None, Header()] = None) -> Principal:
    """从 Authorization 头取令牌并验签；失败一律 401，不告诉前端"哪一步错了"。"""
    token = parse_bearer(authorization)
    return verify_token(token, root=PROJECT_ROOT)


PrincipalDep = Annotated[Principal, Depends(current_principal)]
ServiceDep = Annotated[PlatformService, Depends(get_service)]


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, TokenError):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, PermissionDenied):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, NotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, BadRequest):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="服务内部错误")


def create_app(context: PlatformContext | None = None) -> FastAPI:
    """建应用；传入 context 时用它（测试与验收脚本用临时目录装配）。"""
    app = FastAPI(
        title="电商售后服务智能体工作台",
        version="1.0.0",
        description="单元24：一个页面 + 一组接口，把工单、知识、统计、转人工连成闭环。",
    )
    if context is not None:
        get_platform.cache_clear()
        # 一次装配、一个 service：工单表与台账的缓存不会每个请求重建一遍。
        service = PlatformService(context)
        app.dependency_overrides[get_platform] = lambda: context
        app.dependency_overrides[get_service] = lambda: service

    @app.exception_handler(TokenError)
    def _token_error(_request, exc: TokenError) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": str(exc)})

    @app.exception_handler(PermissionDenied)
    def _permission_error(_request, exc: PermissionDenied) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(NotFound)
    def _not_found(_request, exc: NotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(BadRequest)
    def _bad_request(_request, exc: BadRequest) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    # ------------------------------------------------------------ 页面与健康

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=500, detail="缺少 web/index.html")
        return FileResponse(page)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "ecommerce-after-sales-workbench"}

    @app.get("/v1/meta")
    def meta() -> dict[str, Any]:
        """页面初始化用：不含任何业务数据，也不触发装配。"""
        return {
            "roles": render_roles(),
            "routes": render_routes(),
            "demo_seats": ["A001", "A002", "M001"],
        }

    # ------------------------------------------------------------ 身份

    @app.post("/v1/demo-login", include_in_schema=False)
    def demo_login(payload: DemoLoginRequest, request: Request) -> dict[str, str]:
        if os.getenv("APP_DEMO_LOGIN", "1") == "0" or not request.client or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
            raise HTTPException(status_code=404, detail="未找到页面")
        roles = {"A001": ROLE_AGENT, "A002": ROLE_AGENT, "M001": ROLE_SUPERVISOR}
        role = roles.get(payload.seat_id)
        if role is None:
            raise HTTPException(status_code=422, detail="请选择可用的演示身份")
        return {"token": issue_token(payload.seat_id, role, root=PROJECT_ROOT)}

    @app.get("/v1/me")
    def me(principal: PrincipalDep, service: ServiceDep) -> dict[str, Any]:
        seat = service.context.data.seat_of(principal)
        return {
            "seat_id": principal.seat_id,
            "name": seat.name if seat else principal.seat_id,
            "team": seat.team if seat else "",
            "role": principal.role,
            "role_label": principal.role_label,
            "can_view_all": can_view_all(principal),
            "knowledge": service.context.knowledge_count,
            "model": service.context.model_name,
        }

    # ------------------------------------------------------------ 对话

    @app.post("/v1/chat")
    def chat(request: ChatRequest, principal: PrincipalDep, service: ServiceDep) -> dict[str, Any]:
        try:
            result = service.answer(principal, request.question, request.session_id)
        except HTTPException:
            raise
        except (TokenError, PermissionDenied, NotFound, BadRequest) as exc:
            raise _http_error(exc) from exc
        except (RuntimeError, ConnectionError, TimeoutError) as exc:
            # 内部故障不是 400：不把密钥、路径、堆栈回给浏览器。
            raise HTTPException(status_code=503, detail="知识服务暂不可用，请稍后重试") from exc
        return {
            "request_id": result.request_id,
            "route": result.route,
            "route_label": result.route_label,
            "answer": result.answer,
            "sources": result.sources,
            "status": result.status,
            "duration_ms": result.duration_ms,
            "meta": result.meta,
        }

    @app.get("/v1/sessions/{session_id}")
    def session_history(
        session_id: str, principal: PrincipalDep, service: ServiceDep
    ) -> dict[str, Any]:
        try:
            clean = validate_session_id(session_id)
        except BadRequest as exc:
            raise _http_error(exc) from exc
        turns = service.context.sessions.history(principal, clean, limit=20)
        return {"session_id": clean, "turns": turns}

    @app.post("/v1/feedback")
    def feedback(request: FeedbackRequest, principal: PrincipalDep, service: ServiceDep) -> dict[str, str]:
        try:
            service.context.sessions.save_feedback(principal, request.request_id, request.helpful)
        except HTTPException:
            raise
        except (PermissionDenied, NotFound, BadRequest) as exc:
            raise _http_error(exc) from exc
        return {"status": "saved"}

    @app.post("/v1/handoffs")
    def handoff(request: HandoffRequest, principal: PrincipalDep, service: ServiceDep) -> dict[str, str]:
        try:
            service.context.sessions.request_handoff(principal, request.request_id)
        except HTTPException:
            raise
        except (PermissionDenied, NotFound, BadRequest) as exc:
            raise _http_error(exc) from exc
        return {"status": "pending"}

    # ------------------------------------------------------------ 工单与数据页

    @app.get("/v1/tickets")
    def tickets(
        principal: PrincipalDep,
        service: ServiceDep,
        status: str | None = Query(default=None, max_length=20),
        issue: str | None = Query(default=None, max_length=20),
        keyword: str | None = Query(default=None, max_length=40),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, Any]:
        rows = service.context.data.list_tickets(
            principal, status=status, issue=issue, keyword=keyword, limit=limit
        )
        return {
            "scope": "全部坐席" if can_view_all(principal) else f"{principal.seat_id} 名下",
            "total": len(rows),
            "items": [row.as_dict() for row in rows],
        }

    @app.get("/v1/tickets/{case_id}")
    def ticket_detail(case_id: str, principal: PrincipalDep, service: ServiceDep) -> dict[str, Any]:
        if not TICKET_PATTERN.fullmatch(case_id.strip().upper()):
            raise HTTPException(status_code=422, detail="工单号格式应为 AS 加五位数字")
        try:
            ticket = service.context.data.read_ticket(principal, case_id)
        except (NotFound, BadRequest) as exc:
            raise _http_error(exc) from exc
        return {"ticket": ticket.as_dict()}

    @app.get("/v1/dashboard")
    def dashboard(principal: PrincipalDep, service: ServiceDep) -> dict[str, Any]:
        return service.dashboard(principal)

    # ------------------------------------------------------------ 管理台

    @app.get("/v1/admin/handoffs")
    def admin_handoffs(principal: PrincipalDep, service: ServiceDep) -> dict[str, Any]:
        try:
            require_admin(principal)
            return {"items": service.context.sessions.list_handoffs(principal)}
        except PermissionDenied as exc:
            raise _http_error(exc) from exc

    @app.post("/v1/admin/handoffs/{request_id}/resolve")
    def resolve_handoff(
        request_id: str, principal: PrincipalDep, service: ServiceDep
    ) -> dict[str, str]:
        if not re.fullmatch(REQUEST_ID_PATTERN, request_id):
            raise HTTPException(status_code=422, detail="请求编号格式错误")
        try:
            service.context.sessions.resolve_handoff(principal, request_id)
        except (PermissionDenied, NotFound) as exc:
            raise _http_error(exc) from exc
        return {"status": "resolved"}

    @app.get("/v1/admin/feedback")
    def admin_feedback(principal: PrincipalDep, service: ServiceDep) -> dict[str, Any]:
        try:
            require_admin(principal)
            return {"items": service.context.sessions.list_feedback(principal)}
        except PermissionDenied as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/admin/audit")
    def admin_audit(principal: PrincipalDep, service: ServiceDep) -> dict[str, Any]:
        try:
            require_admin(principal)
            return {"events": service.context.sessions.recent_audit(principal)}
        except PermissionDenied as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/admin/stats")
    def admin_stats(principal: PrincipalDep, service: ServiceDep) -> dict[str, Any]:
        try:
            require_admin(principal)
            return service.context.sessions.stats(principal)
        except PermissionDenied as exc:
            raise _http_error(exc) from exc

    @app.post("/v1/admin/reindex")
    def reindex(principal: PrincipalDep, service: ServiceDep) -> dict[str, int]:
        try:
            require_admin(principal)
        except PermissionDenied as exc:
            raise _http_error(exc) from exc
        return {"records": refresh_knowledge(service.context)}

    return app


app = create_app()


def demo_tokens() -> dict[str, str]:
    """启动脚本打印两个演示令牌：A001（客服）与 M001（主管）。"""
    return {
        "A001": issue_token("A001", ROLE_AGENT, root=PROJECT_ROOT),
        "M001": issue_token("M001", "supervisor", root=PROJECT_ROOT),
    }


def _port_busy(host: str, port: int) -> bool:
    """端口能不能绑上；绑不上就是有人（或上一次没退干净的服务）在听。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def _pick_port(host: str, port: int) -> int | None:
    """优先用要求的端口；被占就往后挪，挪不到再放弃。

    课堂现场最常见的失败不是代码错，而是**上一个工作台还开着**：
    uvicorn 报 `[Errno 10048]` 然后 exit 3，看起来像启动失败。
    这里自己挑一个能用的端口，并说清"原来的端口被谁占了"。
    """
    for candidate in (port, port + 1, port + 2, port + 21):
        if not _port_busy(host, candidate):
            if candidate != port:
                print(f"提示：{port} 端口已被占用（可能已经有一个工作台在跑），改用 {candidate}。")
                print(f"      想继续用 {port} 的话，先把那个服务关掉（关掉它的终端，或结束占用进程）。\n")
            return candidate
    return None


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
    """启动服务：先装配一次并打印"装了什么 + 两个演示令牌"，再起 uvicorn。

    只绑本机，课堂验收默认 8000 端口；`APP_PORT` 可覆盖（端口被占时用）。
    """
    import os

    import uvicorn

    from app.platform_runtime import prepare_platform, render_context

    port = int(os.getenv("APP_PORT", port))
    port = _pick_port(host, port)
    if port is None:
        print("8000—8002 与 8021 都被占用，请用 --port 指定一个空闲端口（例如 --port 8030）。")
        return
    context = prepare_platform()
    tokens = demo_tokens()
    print(render_context(context))
    print(f"\n工作台地址：http://{host}:{port}")
    print(f"客服 A001 令牌：{tokens['A001']}")
    print(f"主管 M001 令牌：{tokens['M001']}")
    print("把令牌粘贴到页面「身份接入」里即可；令牌 8 小时后过期，重启会重新打印。\n")
    uvicorn.run(create_app(context), host=host, port=port, reload=False, log_level="info")


def page_path() -> Path:
    return WEB_DIR / "index.html"
