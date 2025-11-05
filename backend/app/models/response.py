from typing import Generic, TypeVar, Optional, Any, List, Dict
from pydantic import BaseModel

T = TypeVar('T')


class APIResponse(BaseModel, Generic[T]):
    """API响应模型"""
    success: bool
    data: Optional[T] = None
    message: str = ""
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """错误响应模型"""
    success: bool = False
    error: str
    message: str = ""
    details: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应模型"""
    success: bool = True
    data: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int