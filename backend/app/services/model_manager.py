"""
Kronos模型管理器 - 单例模式
"""
import asyncio
import logging
from typing import Optional
from pathlib import Path
import threading
from contextlib import asynccontextmanager

from .kronos_integration import kronos_integration

logger = logging.getLogger(__name__)


class ModelManager:
    """Kronos模型单例管理器"""
    
    _instance = None
    _lock = threading.Lock()
    _model_lock = asyncio.Semaphore(3)  # 限制并发模型使用数量
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._load_model()
    
    def _load_model(self):
        """加载Kronos模型"""
        try:
            logger.info("加载Kronos模型...")
            
            # 尝试加载真实的Kronos模型
            if kronos_integration.is_available():
                success = kronos_integration.load_model("kronos-small")
                if success:
                    logger.info("Kronos模型加载成功")
                    return
                else:
                    logger.warning("Kronos模型加载失败")
            else:
                logger.warning("Kronos模型不可用")
            
        except Exception as e:
            logger.error(f"加载Kronos模型失败: {e}")
    
    @asynccontextmanager
    async def get_model(self):
        """获取模型使用权（异步上下文管理器）"""
        async with self._model_lock:
            if not kronos_integration.model_loaded:
                # 如果真实模型未加载，返回错误信息
                logger.error("Kronos模型不可用或未加载")
                raise RuntimeError("Kronos模型不可用或未加载")
            yield kronos_integration
    
    def get_model_info(self):
        """获取模型信息"""
        return kronos_integration.get_model_info()
    
    def reload_model(self):
        """重新加载模型"""
        logger.info("重新加载Kronos模型...")
        
        try:
            # 先卸载当前模型
            kronos_integration.unload_model()
            
            # 获取当前模型名称，如果未设置则使用默认值
            current_model = getattr(kronos_integration, 'current_model', 'kronos-small')
            
            # 重新加载模型，并获取详细结果
            result = kronos_integration.load_model_with_details(current_model)
            
            return result
            
        except Exception as e:
            logger.error(f"重新加载模型时出错: {e}")
            return {
                "success": False,
                "message": f"重新加载模型失败: {str(e)}",
                "error": str(e)
            }
    
    def get_available_models(self):
        """获取可用的模型列表"""
        return kronos_integration.get_available_models()
    
    def get_current_model(self):
        """获取当前加载的模型名称"""
        return getattr(kronos_integration, 'current_model', None)
    
    def switch_model(self, model_name: str):
        """切换到指定模型"""
        logger.info(f"切换模型到: {model_name}")
        
        try:
            # 获取当前模型名称
            current_model = self.get_current_model()
            
            # 如果要切换的模型就是当前模型，直接返回
            if current_model == model_name and kronos_integration.model_loaded:
                return {
                    "success": True,
                    "message": f"模型 {model_name} 已经加载",
                    "model_name": model_name
                }
            
            # 卸载当前模型（如果有的话）
            unloaded_model = None
            if kronos_integration.model_loaded:
                unloaded_model = current_model
                logger.info(f"卸载当前模型: {current_model}")
                kronos_integration.unload_model()
            
            # 加载新模型
            result = kronos_integration.load_model_with_details(model_name)
            
            # 添加卸载的模型信息
            if result.get("success") and unloaded_model:
                result["unloaded_model"] = unloaded_model
                result["message"] = f"已从 {unloaded_model} 切换到 {model_name}"
            
            return result
            
        except Exception as e:
            logger.error(f"切换模型时出错: {e}")
            return {
                "success": False,
                "message": f"切换模型失败: {str(e)}",
                "error": str(e)
            }


# 全局模型管理器实例
model_manager = ModelManager()