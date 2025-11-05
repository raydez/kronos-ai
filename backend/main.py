"""
Kronos A股预测分析系统 - FastAPI主应用
"""
import asyncio
import logging
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models.stock import PredictionRequest, StockData
from app.models.response import APIResponse, ErrorResponse
from app.services.prediction_service import prediction_service
from app.services.stock_service import stock_service
from app.services.model_manager import model_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        # 控制台处理器
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    logger.info("Starting Kronos A股预测分析系统...")
    
    # 检查模型状态
    model_info = model_manager.get_model_info()
    logger.info(f"Model status: {model_info}")
    
    yield
    
    # 关闭时执行
    logger.info("Shutting down Kronos A股预测分析系统...")


# 创建FastAPI应用
app = FastAPI(
    title="Kronos A股预测分析系统",
    description="基于Kronos金融大模型的A股日线预测分析系统",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Kronos A股预测分析系统",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    model_info = model_manager.get_model_info()
    return {
        "status": "healthy",
        "model": model_info,
        "timestamp": asyncio.get_event_loop().time()
    }


@app.post("/predict", response_model=APIResponse[dict])
async def predict_stock(request: PredictionRequest):
    """预测股票走势"""
    try:
        # 验证股票代码
        if not await stock_service.validate_stock_code(request.code):
            raise HTTPException(status_code=400, detail="无效的股票代码")
        
        # 进行预测
        result = await prediction_service.predict_stock(request.code, request.prediction_days, request.start_date)
        
        if not result:
            raise HTTPException(status_code=500, detail="预测失败")
        
        return APIResponse(
            success=True,
            data=result.dict(),
            message="预测成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"预测过程中发生错误: {str(e)}")


@app.get("/stock/{code}")
async def get_stock_info(code: str):
    """获取股票信息"""
    try:
        stock_info = await stock_service.get_stock_info(code)
        if not stock_info:
            raise HTTPException(status_code=404, detail="股票信息未找到")
        
        return APIResponse(
            success=True,
            data=stock_info,
            message="获取股票信息成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get stock info error: {e}")
        raise HTTPException(status_code=500, detail=f"获取股票信息失败: {str(e)}")


@app.get("/stock/{code}/history")
async def get_stock_history(code: str, days: int = 30):
    """获取股票历史数据"""
    try:
        if days < 1 or days > 365:
            raise HTTPException(status_code=400, detail="天数必须在1-365之间")
        
        stock_data = await stock_service.get_stock_data(code, days)
        if not stock_data:
            raise HTTPException(status_code=404, detail="股票历史数据未找到")
        
        return APIResponse(
            success=True,
            data={
                "code": code,
                "history": stock_data
            },
            message="获取股票历史数据成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get stock history error: {e}")
        raise HTTPException(status_code=500, detail=f"获取股票历史数据失败: {str(e)}")


@app.get("/stock/{code}/actual")
async def get_stock_actual_data(code: str, start_date: str, end_date: str):
    """获取指定时间范围内的实际股票数据（用于对比预测结果）"""
    try:
        # 验证股票代码
        if not await stock_service.validate_stock_code(code):
            raise HTTPException(status_code=400, detail="无效的股票代码")
        
        # 获取实际数据
        actual_data = await stock_service.get_stock_data_for_dates(code, start_date, end_date)
        
        return APIResponse(
            success=True,
            data={
                "code": code,
                "start_date": start_date,
                "end_date": end_date,
                "actual": actual_data
            },
            message="获取实际股票数据成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get actual stock data error: {e}")
        raise HTTPException(status_code=500, detail=f"获取实际股票数据失败: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """获取模型信息"""
    try:
        model_info = model_manager.get_model_info()
        
        return APIResponse(
            success=True,
            data=model_info,
            message="获取模型信息成功"
        )
        
    except Exception as e:
        logger.error(f"Get model info error: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


@app.post("/model/reload")
async def reload_model():
    """重新加载模型"""
    try:
        logger.info("收到重新加载模型请求")
        
        # 调用模型管理器的重新加载方法
        result = model_manager.reload_model()
        
        # 获取更新后的模型信息
        model_info = model_manager.get_model_info()
        
        if result.get("success", False):
            return APIResponse(
                success=True,
                data={
                    "model_info": model_info,
                    "message": result.get("message", "模型重新加载成功"),
                    "download_info": result.get("download_info")
                },
                message="模型重新加载成功"
            )
        else:
            return APIResponse(
                success=False,
                data={
                    "model_info": model_info,
                    "error": result.get("error", "未知错误")
                },
                message=result.get("message", "模型重新加载失败")
            )
        
    except Exception as e:
        logger.error(f"Reload model error: {e}")
        raise HTTPException(status_code=500, detail=f"重新加载模型失败: {str(e)}")


@app.get("/model/available")
async def get_available_models():
    """获取可用的模型列表"""
    try:
        logger.info("获取可用模型列表")
        
        # 获取可用模型列表
        models = model_manager.get_available_models()
        
        return APIResponse(
            success=True,
            data={
                "models": models,
                "current_model": model_manager.get_current_model()
            },
            message="获取可用模型列表成功"
        )
        
    except Exception as e:
        logger.error(f"Get available models error: {e}")
        raise HTTPException(status_code=500, detail=f"获取可用模型列表失败: {str(e)}")


@app.post("/model/switch")
async def switch_model(request: dict):
    """切换模型"""
    try:
        model_name = request.get("model_name")
        if not model_name:
            raise HTTPException(status_code=400, detail="缺少模型名称参数")
        
        logger.info(f"收到切换模型请求: {model_name}")
        
        # 调用模型管理器的切换方法
        result = model_manager.switch_model(model_name)
        
        # 获取更新后的模型信息
        model_info = model_manager.get_model_info()
        
        if result.get("success", False):
            return APIResponse(
                success=True,
                data={
                    "model_info": model_info,
                    "message": result.get("message", "模型切换成功"),
                    "download_info": result.get("download_info"),
                    "unloaded_model": result.get("unloaded_model")
                },
                message="模型切换成功"
            )
        else:
            return APIResponse(
                success=False,
                data={
                    "model_info": model_info,
                    "error": result.get("error", "未知错误")
                },
                message=result.get("message", "模型切换失败")
            )
        
    except Exception as e:
        logger.error(f"Switch model error: {e}")
        raise HTTPException(status_code=500, detail=f"切换模型失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )