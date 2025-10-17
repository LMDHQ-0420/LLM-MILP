from pydantic import BaseModel
from typing import Any, Optional


class ErrorResponse(BaseModel):
    """统一API错误响应模型"""
    status_code: int
    detail: str
    error_type: str
    path: Optional[str] = None
    timestamp: str
    additional_info: Optional[dict[str, Any]] = None
