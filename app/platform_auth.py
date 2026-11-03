"""单元24：网页工作台的身份出入口——HMAC 签名令牌 + 角色判定。

这一段是纯标准库实现，不 import FastAPI：先能单测，才谈得上挂到网页上。

三件事必须在这里做完，不能留给"业务代码里顺手判断"：

1. **签名**：令牌载荷里写着坐席号与角色，但角色是权限的来源，**不能由浏览器自报**。
   所以载荷用服务端密钥做 HMAC-SHA256，前端改一个字符就验不过。
2. **有效期**：课堂令牌 8 小时过期；过期不是"再给一次机会"，是直接 401。
3. **角色**：`agent`（客服坐席）只能看见自己名下的工单；`supervisor`（主管）与
   `admin`（管理员）能看全部。判定写在 `can_view_all()` 一处，业务代码只调它。

课堂令牌是教学方案：正式接企业门户时应换成 SSO/OIDC，并把角色、租户、知识文档
ACL 与业务系统权限对齐——这段话写进 README，也写在这里，免得被当成生产实现。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

TENANT = "ecommerce-demo"

ROLE_AGENT = "agent"
ROLE_SUPERVISOR = "supervisor"
ROLE_ADMIN = "admin"
ROLES = (ROLE_AGENT, ROLE_SUPERVISOR, ROLE_ADMIN)
ROLE_LABELS = {ROLE_AGENT: "客服坐席", ROLE_SUPERVISOR: "售后主管", ROLE_ADMIN: "系统管理员"}
# 管理台（转人工队列、反馈、调用记录、重建索引）只有这两个角色能进。
ADMIN_ROLES = frozenset({ROLE_SUPERVISOR, ROLE_ADMIN})

SEAT_PATTERN = re.compile(r"^[AM]\d{3}$")
SECRET_ENV = "APP_TOKEN_SECRET"
SECRET_FILE = Path(".runtime") / "token_secret"
MIN_SECRET_CHARS = 32
DEFAULT_HOURS = 8


class TokenError(ValueError):
    """令牌缺失、签名不对、已过期或角色非法。API 层统一映射成 401。"""


class PermissionDenied(PermissionError):
    """身份合法但没有这个权限。API 层映射成 403。"""


@dataclass(frozen=True)
class Principal:
    """一次请求的"我是谁"：坐席号、角色、租户。角色只能来自已验签的令牌。"""

    seat_id: str
    role: str
    tenant: str = TENANT

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)


# ---------------------------------------------------------------- 密钥


def _secret_path(root: Path) -> Path:
    return root / SECRET_FILE


def load_secret(root: Path, secret: str | None = None) -> bytes:
    """取签名密钥：显式参数 > 环境变量 > `.runtime/token_secret`（首次自动生成）。

    不往 `.env` 里追加内容：`.env` 是随课程材料分发的模板，被程序改过就说不清了；
    `.runtime/` 本来就是运行产物目录。
    """
    if secret:
        if len(secret) < MIN_SECRET_CHARS:
            raise TokenError(f"签名密钥至少需要 {MIN_SECRET_CHARS} 个字符")
        return secret.encode("utf-8")
    from_env = os.getenv(SECRET_ENV, "")
    if from_env:
        if len(from_env) < MIN_SECRET_CHARS:
            raise TokenError(f"环境变量 {SECRET_ENV} 至少需要 {MIN_SECRET_CHARS} 个字符")
        return from_env.encode("utf-8")
    path = _secret_path(root)
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if len(value) >= MIN_SECRET_CHARS:
            return value.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(48)
    path.write_text(value, encoding="utf-8")
    return value.encode("utf-8")


# ---------------------------------------------------------------- 签发与校验


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue_token(
    seat_id: str,
    role: str = ROLE_AGENT,
    *,
    root: Path | None = None,
    secret: str | None = None,
    hours: int = DEFAULT_HOURS,
    now: int | None = None,
) -> str:
    """签发课堂令牌。只有命令行/启动脚本能调，API 不暴露签发入口。"""
    if not SEAT_PATTERN.match(seat_id or ""):
        raise TokenError("坐席号格式应为 A001 / M001（首字母 A 或 M，后接三位数字）")
    if role not in ROLES:
        raise TokenError(f"角色必须是 {'、'.join(ROLES)}")
    if hours <= 0:
        raise TokenError("有效期必须是正数小时")
    issued = int(now if now is not None else time.time())
    payload = {
        "sub": seat_id,
        "role": role,
        "tenant": TENANT,
        "iat": issued,
        "exp": issued + hours * 3600,
    }
    body = _b64(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    key = load_secret(root or Path.cwd(), secret)
    signature = _b64(hmac.new(key, body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{signature}"


def verify_token(
    token: str,
    *,
    root: Path | None = None,
    secret: str | None = None,
    now: int | None = None,
) -> Principal:
    """先验签名与有效期，再信任载荷里的坐席号与角色。任何一步不过都抛 TokenError。"""
    if not isinstance(token, str) or not token.strip():
        raise TokenError("请提供 Bearer 身份凭证")
    token = token.strip()
    try:
        body, signature = token.split(".", 1)
        key = load_secret(root or Path.cwd(), secret)
        expected = _b64(hmac.new(key, body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise TokenError("身份凭证无效或已过期")
        payload = json.loads(_unb64(body))
        if not isinstance(payload, dict):
            raise TokenError("身份凭证格式错误")
        expires = int(payload["exp"])
        issued = int(payload.get("iat", 0))
        current = int(now if now is not None else time.time())
        if expires <= current:
            raise TokenError("身份凭证已过期，请重新签发")
        if issued > current + 300:
            raise TokenError("身份凭证的签发时间不合法")
        if payload.get("tenant") != TENANT:
            raise TokenError("身份凭证不属于本项目")
        role = str(payload.get("role"))
        if role not in ROLES:
            raise TokenError("身份角色不合法")
        seat_id = str(payload.get("sub"))
        if not SEAT_PATTERN.match(seat_id):
            raise TokenError("坐席号不合法")
        return Principal(seat_id=seat_id, role=role, tenant=TENANT)
    except TokenError:
        raise
    except (ValueError, KeyError, TypeError, binascii.Error) as exc:
        raise TokenError("身份凭证无效或已过期") from exc


def parse_bearer(header: str | None) -> str:
    """从 `Authorization: Bearer xxx` 里取出令牌本体。"""
    if not header:
        raise TokenError("请提供 Bearer 身份凭证")
    parts = header.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise TokenError("Authorization 头应形如：Bearer <令牌>")
    return parts[1].strip()


# ---------------------------------------------------------------- 角色判定


def can_view_all(principal: Principal) -> bool:
    """主管与管理员能看全部工单；客服只能看自己名下的。"""
    return principal.role in ADMIN_ROLES


def require_admin(principal: Principal) -> None:
    """管理台入口的统一守门。"""
    if principal.role not in ADMIN_ROLES:
        raise PermissionDenied(f"该操作需要主管或管理员权限，当前角色：{principal.role_label}")


def render_roles() -> str:
    """第1节课要讲的角色表；渲染与文档都读它。"""
    lines = ["角色\t坐席号示例\t能看什么\t能做什么"]
    lines.append("客服坐席\tA001—A004\t只有自己名下的售后工单\t提问、查工单、反馈、转人工")
    lines.append("售后主管\tM001\t全部工单 + 管理台\t处理转人工队列、看反馈与调用记录、重建索引")
    lines.append("系统管理员\t（可签发）\t同上\t同上（含重建索引）")
    return "\n".join(lines)
