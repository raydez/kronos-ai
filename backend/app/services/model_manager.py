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
        kronos_integration.unload_model()
        self._load_model()
    
    def get_available_models(self):
        """获取可用模型列表"""
        return kronos_integration.get_available_models()
    
    def switch_model(self, model_name: str) -> bool:
        """切换模型"""
        try:
            logger.info(f"切换到模型: {model_name}")
            kronos_integration.unload_model()
            success = kronos_integration.load_model(model_name)
            if success:
                logger.info(f"切换到模型 {model_name} 成功")
            else:
                logger.error(f"切换到模型 {model_name} 失败")
            return success
        except Exception as e:
            logger.error(f"切换模型时出错: {e}")
            return False


# 全局模型管理器实例
model_manager = ModelManager()