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
            logger.info("Loading Kronos model...")
            
            # 尝试加载真实的Kronos模型
            if kronos_integration.is_available():
                success = kronos_integration.load_model("kronos-small")
                if success:
                    logger.info("Kronos model loaded successfully")
                    return
                else:
                    logger.warning("Failed to load Kronos model, will use fallback")
            else:
                logger.warning("Kronos not available, will use simulated predictions")
            
        except Exception as e:
            logger.error(f"Failed to load Kronos model: {e}")
    
    @asynccontextmanager
    async def get_model(self):
        """获取模型使用权（异步上下文管理器）"""
        async with self._model_lock:
            if not kronos_integration.model_loaded:
                # 如果真实模型未加载，仍然允许使用模拟预测
                logger.warning("Using simulated predictions - real model not loaded")
            yield kronos_integration
    
    def get_model_info(self):
        """获取模型信息"""
        return kronos_integration.get_model_info()
    
    def reload_model(self):
        """重新加载模型"""
        logger.info("Reloading Kronos model...")
        kronos_integration.unload_model()
        self._load_model()
    
    def get_available_models(self):
        """获取可用模型列表"""
        return kronos_integration.get_available_models()
    
    def switch_model(self, model_name: str) -> bool:
        """切换模型"""
        try:
            logger.info(f"Switching to model: {model_name}")
            kronos_integration.unload_model()
            success = kronos_integration.load_model(model_name)
            if success:
                logger.info(f"Successfully switched to {model_name}")
            else:
                logger.error(f"Failed to switch to {model_name}")
            return success
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False


# 全局模型管理器实例
model_manager = ModelManager()