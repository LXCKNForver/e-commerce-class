"""单元24：业务层自己的错误类型。

为什么不直接在业务代码里 `raise HTTPException`：会话存储与工单读取都要能在没有
FastAPI 的环境里单测。业务层只说"没找到""没权限"，接口层再决定回 404 还是 403。
"""

from __future__ import annotations


class NotFound(LookupError):
    """查不到，或者没有权限看——两者对外都回 404，不泄露"这条记录存在"。"""


class BadRequest(ValueError):
    """参数本身不合法（编号格式、问题长度、会话编号等）。"""
