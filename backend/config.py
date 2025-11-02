"""
配置文件
"""
import os

class Config:
    # 数据源配置
    DATA_SOURCE = "baostock"
    BACKUP_SOURCE = "tushare"
    
    # 模型配置
    KRONOS_MODEL = "kronos-small"
    MODEL_PATH = "./data/models"
    MODEL_DEVICE = "cpu"  # 或 "cuda:0"
    MAX_CONTEXT = 512
    
    # 并发配置
    MAX_CONCURRENT_REQUESTS = 10
    REQUEST_TIMEOUT = 30  # 秒
    
    # 数据库配置
    DATABASE_URL = "sqlite:///./data/database/stock_predictor.db"
    
    # 缓存配置
    CACHE_DIR = "./data/cache"
    CACHE_EXPIRE = 3600  # 1小时
    REDIS_URL = "redis://localhost:6379"  # 可选
    
    # 日志配置
    LOG_DIR = "./data/logs"
    
    # 预测配置
    DEFAULT_PREDICTION_DAYS = 5
    MAX_PREDICTION_DAYS = 10
    
    # 性能配置
    BATCH_SIZE = 1  # 单个请求处理
    ENABLE_MODEL_WARMUP = True
    
    # API配置
    API_PREFIX = "/api/v1"
    HOST = "0.0.0.0"
    PORT = 8000
    
    # 跨域配置
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]

class DevelopmentConfig(Config):
    DEBUG = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    DEBUG = False
    LOG_LEVEL = "INFO"

# 根据环境变量选择配置
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}