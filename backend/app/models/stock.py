from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class StockData(BaseModel):
    """股票数据模型"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    code: str = Field(..., description="股票代码")


class PredictionRequest(BaseModel):
    """预测请求模型"""
    code: str = Field(..., description="股票代码")
    prediction_days: int = Field(default=5, ge=1, le=30, description="预测天数")
    start_date: Optional[str] = Field(None, description="开始日期 (YYYY-MM-DD)")


class PredictionResult(BaseModel):
    """预测结果模型"""
    code: str
    prediction_days: int
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)


class PredictionPoint(BaseModel):
    """预测点模型"""
    date: str
    predicted_close: float
    confidence: float
    trend: str  # "up", "down", "stable"


class PredictionResponse(BaseModel):
    """预测响应模型"""
    code: str
    prediction_days: int
    predictions: List[PredictionPoint]
    model_info: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)


class ModelInfo(BaseModel):
    """模型信息模型"""
    name: str
    version: str
    status: str  # "loaded", "loading", "error", "offline"
    description: Optional[str] = None
    last_updated: Optional[datetime] = None


class StockInfo(BaseModel):
    """股票信息模型"""
    code: str
    name: str
    industry: Optional[str] = None
    market: Optional[str] = None
    list_date: Optional[str] = None