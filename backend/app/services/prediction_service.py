"""
预测服务 - 负责使用Kronos模型进行股票预测
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from .model_manager import model_manager
from .stock_service import stock_service
from ..models.stock import PredictionResponse, PredictionPoint, ModelInfo

logger = logging.getLogger(__name__)


class PredictionService:
    """预测服务"""
    
    def __init__(self):
        self.prediction_cache = {}
        self.cache_ttl = 3600  # 缓存1小时
    
    async def predict_stock(self, code: str, prediction_days: int = 5) -> Optional[PredictionResponse]:
        """预测股票走势"""
        try:
            # 检查缓存
            cache_key = f"{code}_{prediction_days}"
            if cache_key in self.prediction_cache:
                cached_data, timestamp = self.prediction_cache[cache_key]
                if datetime.now().timestamp() - timestamp < self.cache_ttl:
                    logger.info(f"使用缓存预测结果为 {code}")
                    return cached_data
            
            # 获取股票信息
            stock_info = await stock_service.get_stock_info(code)
            if not stock_info:
                raise ValueError(f"无法获取股票 {code} 的信息")
            
            # 获取历史数据
            stock_data = await stock_service.get_stock_data(code, days=60)
            if not stock_data or len(stock_data) < 30:
                raise ValueError(f"股票 {code} 的历史数据不足")
            
            # 使用Kronos模型进行预测
            predictions = await self._predict_with_kronos(stock_data, prediction_days)
            
            # 构建响应
            model_info = ModelInfo(**model_manager.get_model_info())
            
            response = PredictionResponse(
                code=code,
                name=stock_info["name"],
                predictions=predictions,
                model_info=model_info
            )
            
            # 缓存结果
            self.prediction_cache[cache_key] = (response, datetime.now().timestamp())
            
            return response
            
        except Exception as e:
            logger.error(f"预测股票 {code} 失败: {e}")
            raise
    
    async def _predict_with_kronos(self, stock_data: List[Dict[str, Any]], prediction_days: int) -> List[PredictionPoint]:
        """使用Kronos模型进行预测"""
        try:
            # 获取模型使用权
            async with model_manager.get_model() as model:
                # 进行预测
                predictions = model.predict_stock(stock_data, prediction_days)
                
                if predictions:
                    # 转换为PredictionPoint格式
                    prediction_points = []
                    for pred in predictions:
                        # 确保日期格式正确
                        date_value = pred['date']
                        if hasattr(date_value, 'strftime'):
                            date_str = date_value.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date_value).split('T')[0]  # 去掉时间部分
                        
                        prediction_points.append(PredictionPoint(
                            date=date_str,
                            open=pred['open'],
                            high=pred['high'],
                            low=pred['low'],
                            close=pred['close'],
                            confidence=pred['confidence']
                        ))
                    return prediction_points
                else:
                    # 如果模型预测失败，返回None
                    logger.error(f"Kronos模型预测股票 {code} 返回空结果")
                    return None
                
        except Exception as e:
            logger.error(f"使用Kronos模型预测股票 {code} 失败: {e}")
            # 如果模型预测失败，返回None
            return None
    
    def _prepare_input_data(self, stock_data: List[Dict[str, Any]]) -> np.ndarray:
        """准备模型输入数据"""
        # 转换为DataFrame
        df = pd.DataFrame(stock_data)
        
        # 提取特征
        features = []
        for i in range(len(df)):
            row = df.iloc[i]
            # 基本特征
            feature = [
                row['open'],
                row['high'], 
                row['low'],
                row['close'],
                row['volume'] if 'volume' in row else 0
            ]
            
            # 技术指标
            if i >= 5:
                # 5日移动平均
                ma5 = df.iloc[i-5:i]['close'].mean()
                feature.append(ma5)
                
                # 价格变化率
                price_change = (row['close'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close']
                feature.append(price_change)
            else:
                feature.extend([row['close'], 0.0])
            
            features.append(feature)
        
        return np.array(features)
    

    
    async def batch_predict(self, codes: List[str], prediction_days: int = 5) -> List[Dict[str, Any]]:
        """批量预测"""
        tasks = []
        for code in codes:
            task = self.predict_stock(code, prediction_days)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                batch_results.append({
                    "code": codes[i],
                    "success": False,
                    "error": str(result)
                })
            else:
                batch_results.append({
                    "code": codes[i],
                    "success": True,
                    "data": result
                })
        
        return batch_results
    
    def clear_cache(self):
        """清空预测缓存"""
        self.prediction_cache.clear()
        logger.info("预测缓存已清空")


# 全局预测服务实例
prediction_service = PredictionService()