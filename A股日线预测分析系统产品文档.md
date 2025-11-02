# A股日线预测分析系统产品文档 (MVP版本)

## 1. 产品概述

### 1.1 产品定位
基于Kronos金融基础模型构建的个人化A股日线行情预测系统，为个人投资者提供简单易用的股价预测工具。

### 1.2 核心价值
- **简单预测**：输入股票代码，获取日线行情，进行价格预测
- **直观展示**：可视化展示历史数据和预测结果
- **快速响应**：轻量级架构，快速生成预测结果

### 1.3 目标用户
- 个人投资者
- 股票爱好者

## 2. 技术架构

### 2.1 核心技术栈
- **前端**：React + TypeScript + Ant Design + ECharts
- **后端**：Python + FastAPI + asyncio
- **基础模型**：Kronos金融基础模型（优先使用kronos-small）
- **数据处理**：Pandas、NumPy
- **数据源**：baostock（主要）、tushare/akshare（备用）
- **数据库**：SQLite（轻量级）
- **缓存**：Python内置缓存 + Redis（可选）
- **并发处理**：异步请求处理 + 模型单例模式

### 2.2 系统架构图
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面      │    │   后端API       │    │   数据层        │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • 股票代码输入  │    │ • 行情数据获取  │    │ • SQLite数据库  │
│ • 预测结果展示  │    │ • Kronos预测    │    │ • 缓存机制      │
│ • 图表可视化    │    │ • 数据处理      │    │ • 文件存储      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   外部数据源    │
                    ├─────────────────┤
                    │ • baostock      │
                    │ • tushare       │
                    │ • akshare       │
                    └─────────────────┘
```

## 3. 核心功能

### 3.1 股票代码输入
- **输入框**：用户输入6位A股代码（如：000001、600000）
- **格式验证**：自动验证股票代码格式
- **股票信息**：显示股票名称和基本信息

### 3.2 日线行情获取
- **历史数据**：获取最近90个交易日的日线数据
- **数据字段**：日期、开盘价、最高价、最低价、收盘价、成交量
- **数据缓存**：缓存当日数据，减少重复请求

### 3.3 价格预测
- **预测模型**：使用Kronos-small模型进行预测
- **预测周期**：预测未来5-10个交易日
- **预测输出**：开盘价、最高价、最低价、收盘价预测

### 3.4 结果展示
- **历史K线图**：展示历史90天日线数据
- **预测K线图**：展示预测的未来走势
- **数据表格**：显示具体的预测数值
- **置信度**：显示预测的置信度评分

## 4. 数据流程

### 4.1 用户操作流程
```
用户输入股票代码 → 前端验证格式 → 后端获取历史数据 → 
Kronos模型预测 → 结果可视化 → 展示给用户
```

### 4.2 数据处理流程
1. **数据获取**：从baostock获取日线行情数据
2. **数据验证**：检查数据完整性和格式
3. **数据预处理**：转换为Kronos模型所需格式
4. **模型预测**：调用Kronos模型进行预测
5. **结果处理**：格式化预测结果
6. **缓存存储**：缓存数据和结果

### 4.3 数据格式
```python
# 输入数据格式（Kronos模型）
{
    "timestamps": ["2024-01-01", "2024-01-02", ...],
    "open": [10.5, 10.8, ...],
    "high": [10.9, 11.0, ...],
    "low": [10.4, 10.7, ...],
    "close": [10.8, 10.9, ...],
    "volume": [1000000, 1200000, ...]
}

# 预测结果格式
{
    "predictions": [
        {
            "date": "2024-01-03",
            "open": 11.0,
            "high": 11.2,
            "low": 10.9,
            "close": 11.1,
            "confidence": 0.85
        }
    ]
}
```

## 5. API设计

### 5.1 核心API接口

#### 5.1.1 股票信息查询
```python
GET /api/stock/{code}
Response:
{
    "success": true,
    "data": {
        "code": "000001",
        "name": "平安银行",
        "market": "深交所主板"
    }
}
```

#### 5.1.2 历史数据获取
```python
GET /api/stock/{code}/history?days=90
Response:
{
    "success": true,
    "data": {
        "code": "000001",
        "name": "平安银行",
        "history": [
            {
                "date": "2024-01-01",
                "open": 10.5,
                "high": 10.9,
                "low": 10.4,
                "close": 10.8,
                "volume": 1000000
            }
        ]
    }
}
```

#### 5.1.3 价格预测
```python
POST /api/predict
{
    "code": "000001",
    "prediction_days": 5
}

Response:
{
    "success": true,
    "data": {
        "code": "000001",
        "name": "平安银行",
        "predictions": [
            {
                "date": "2024-01-03",
                "open": 11.0,
                "high": 11.2,
                "low": 10.9,
                "close": 11.1,
                "confidence": 0.85
            }
        ],
        "model_info": {
            "model": "kronos-small",
            "accuracy": 0.78,
            "model_loaded": true,
            "processing_time": "1.2s"
        }
    }
}
```

#### 5.1.4 模型状态查询
```python
GET /api/model/status
Response:
{
    "success": true,
    "data": {
        "model_loaded": true,
        "model_name": "kronos-small",
        "device": "cpu",
        "memory_usage": "2.1GB",
        "active_requests": 2,
        "total_requests": 156
    }
}
```

### 5.2 错误处理
```python
# 股票代码不存在
{
    "success": false,
    "error": "Stock code not found",
    "code": 404
}

# 数据获取失败
{
    "success": false,
    "error": "Failed to fetch stock data",
    "code": 500
}

# 预测失败
{
    "success": false,
    "error": "Prediction service unavailable",
    "code": 503
}
```

## 6. 前端界面设计

### 6.1 主界面布局
```
┌─────────────────────────────────────────────────────┐
│                  股票预测系统                        │
├─────────────────────────────────────────────────────┤
│  股票代码: [000001] [查询]    平安银行 (深交所主板)    │
├─────────────────────────────────────────────────────┤
│                                                     │
│              K线图表区域                            │
│    ┌─────────────────────────────────────────┐      │
│    │        历史数据(90天) + 预测数据(5天)    │      │
│    │                                         │      │
│    │                                         │      │
│    └─────────────────────────────────────────┘      │
│                                                     │
├─────────────────────────────────────────────────────┤
│              预测结果表格                            │
│  日期      开盘价   最高价   最低价   收盘价   置信度  │
│  2024-01-03  11.00   11.20   10.90   11.10   85%   │
│  2024-01-04  11.10   11.30   11.00   11.25   82%   │
└─────────────────────────────────────────────────────┘
```

### 6.2 交互设计
- **输入框**：支持6位数字输入，自动格式化
- **查询按钮**：点击后显示加载状态
- **图表交互**：支持缩放、悬停显示详细信息
- **数据刷新**：提供刷新按钮更新最新数据

## 7. 部署方案

### 7.1 目录结构说明

#### 为什么data目录放在backend下？

1. **数据归属关系**：data目录主要由后端服务使用，包含数据库、缓存、模型文件等
2. **服务边界清晰**：前端通过API访问数据，不直接操作文件系统
3. **部署便利性**：后端部署时数据目录随应用一起部署，避免跨目录访问
4. **安全性**：数据文件不暴露在项目根目录，减少意外访问风险
5. **容器化友好**：Docker部署时更容易管理数据卷
6. **相对路径简化**：在backend目录下执行时，路径更简洁

#### 部署时的目录结构
实际部署时，前后端是独立部署的：

**后端部署结构（在backend目录下）：**
```
backend/
├── app/
├── data/                    # 相对路径：./data/
│   ├── cache/
│   ├── database/
│   ├── models/
│   └── logs/
├── requirements.txt
├── config.py
└── main.py                  # 启动：uvicorn app.main:app
```

**前端部署结构（在frontend目录下）：**
```
frontend/
├── src/
├── public/
├── package.json
└── build/                   # 构建产物
```

#### 目录权限管理
```bash
# 在backend目录下创建必要的目录并设置权限
cd backend
mkdir -p data/{cache,database,models,logs}
chmod 755 data
chmod 644 data/database
chmod 644 data/cache
chmod 644 data/logs
```

### 7.2 本地部署
```bash
# 后端启动
cd backend
# 创建数据目录
mkdir -p data/{cache,database,models,logs}
# 安装依赖
pip install -r requirements.txt
# 启动服务
uvicorn main:app --reload --port 8000 --workers 1

# 前端启动（新终端）
cd frontend
# 安装依赖
npm install
# 启动开发服务器
npm start
```

### 7.2 模型管理策略
```python
# 模型单例管理
class ModelManager:
    _instance = None
    _model = None
    _tokenizer = None
    _predictor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def load_model(self):
        if self._predictor is None:
            # 只在第一次调用时加载模型
            from model import Kronos, KronosTokenizer, KronosPredictor
            
            self._tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            self._model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
            self._predictor = KronosPredictor(
                self._model, 
                self._tokenizer, 
                device="cpu",  # 或 "cuda:0"
                max_context=512
            )
            print("Kronos模型加载完成")
    
    def get_predictor(self):
        return self._predictor

# FastAPI应用启动时初始化
@app.on_event("startup")
async def startup_event():
    model_manager = ModelManager()
    await model_manager.load_model()
```

### 7.2 目录结构
```
/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI主应用
│   │   ├── models/          # 数据模型
│   │   ├── services/        # 业务逻辑
│   │   │   ├── model_manager.py    # 模型管理器
│   │   │   ├── stock_service.py     # 股票数据服务
│   │   │   └── prediction_service.py # 预测服务
│   │   └── utils/           # 工具函数
│   ├── data/                # 数据目录
│   │   ├── cache/           # 数据缓存
│   │   ├── database/        # SQLite数据库
│   │   ├── models/          # 下载的模型文件
│   │   └── logs/            # 日志文件
│   ├── requirements.txt
│   └── config.py            # 配置文件
├── frontend/
│   ├── src/
│   │   ├── components/      # React组件
│   │   ├── pages/           # 页面组件
│   │   ├── services/        # API调用
│   │   └── utils/           # 工具函数
│   ├── package.json
│   └── public/
├── doc/                     # 文档目录
│   ├── A股日线预测分析系统产品文档.md
│   └── 架构文档.md
├── scripts/                 # 部署脚本
│   ├── setup.sh
│   └── deploy.sh
└── README.md
```

### 7.3 配置文件
```python
# config.py
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
```

### 7.4 并发处理实现
```python
# services/prediction_service.py
import asyncio
from app.models.model_manager import model_manager

class PredictionService:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_REQUESTS)
        self.request_queue = asyncio.Queue()
    
    async def predict_stock(self, code: str, days: int):
        async with self.semaphore:  # 限制并发请求数
            try:
                # 获取共享的模型实例
                predictor = model_manager.get_predictor()
                if predictor is None:
                    raise ValueError("模型未加载")
                
                # 异步处理预测
                result = await self._async_predict(predictor, code, days)
                return result
                
            except Exception as e:
                raise Exception(f"预测失败: {str(e)}")
    
    async def _async_predict(self, predictor, code: str, days: int):
        # 在线程池中执行CPU密集型预测任务
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._sync_predict, 
            predictor, 
            code, 
            days
        )
    
    def _sync_predict(self, predictor, code: str, days: int):
        # 同步预测逻辑
        stock_data = self._get_stock_data(code)
        prediction = predictor.predict(
            df=stock_data['data'],
            x_timestamp=stock_data['timestamps'],
            y_timestamp=stock_data['future_timestamps'],
            pred_len=days,
            T=1.0,
            top_p=0.9,
            sample_count=1
        )
        return prediction
```

## 8. 开发计划

### 8.1 第一阶段（2周）
- [ ] 搭建基础项目结构
- [ ] 集成baostock数据源
- [ ] 实现基础API接口
- [ ] 创建前端基础界面

### 8.2 第二阶段（2周）
- [ ] 集成Kronos模型（单例模式）
- [ ] 实现异步预测服务
- [ ] 添加并发控制和请求队列
- [ ] 完善前端图表展示
- [ ] 添加数据缓存机制

### 8.3 第三阶段（1周）
- [ ] 优化用户体验
- [ ] 添加错误处理
- [ ] 性能优化
- [ ] 测试和部署

## 9. 风险评估

### 9.1 技术风险
- **数据源稳定性**：baostock可能存在访问限制
- **模型准确性**：Kronos模型预测精度可能受市场影响
- **性能问题**：模型推理可能较慢
- **并发问题**：多用户同时访问可能导致资源竞争
- **内存管理**：模型加载占用大量内存

### 9.2 应对策略
- **多数据源**：准备tushare、akshare作为备用
- **模型优化**：使用kronos-small平衡精度和速度
- **缓存机制**：减少重复计算和数据请求
- **并发控制**：使用信号量限制同时请求数量
- **模型单例**：确保模型只加载一次，共享使用
- **异步处理**：使用asyncio处理并发请求
- **内存监控**：监控模型内存使用，必要时重启服务
- **请求队列**：高负载时使用队列管理请求

## 10. 成功指标

### 10.1 功能指标
- 支持95%以上的A股代码查询
- 预测响应时间 < 3秒
- 系统可用性 > 99%
- 支持并发用户数 > 10
- 模型内存占用 < 3GB
- 请求成功率 > 95%

### 10.2 用户体验指标
- 界面简洁直观，新用户30秒内完成首次预测
- 预测结果可视化清晰易懂
- 移动端适配良好

## 11. 后续扩展

### 11.1 功能扩展
- 支持更多预测周期（周线、月线）
- 添加技术指标分析
- 支持多股票对比
- 添加预测结果导出功能

### 11.2 技术扩展
- 支持更多数据源
- 模型自动更新机制
- 用户个性化设置
- 预测结果历史记录
- 分布式模型部署
- 负载均衡和水平扩展
- 实时预测流处理
- 模型版本管理

## 12. 并发架构设计

### 12.1 模型单例模式
```python
# 确保整个应用生命周期中模型只加载一次
class ModelManager:
    _instance = None
    _predictor = None
    _model_lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_predictor(self):
        async with self._model_lock:
            if self._predictor is None:
                await self._load_model()
            return self._predictor
```

### 12.2 请求处理流程
```
用户请求 → FastAPI接收 → 信号量控制 → 异步队列 → 
模型单例 → 线程池执行 → 返回结果
```

### 12.3 性能优化策略
- **预热机制**：应用启动时预热模型，避免首次请求延迟
- **连接池**：复用数据库连接，减少连接开销
- **结果缓存**：相同股票和参数的预测结果缓存5分钟
- **批量处理**：支持批量股票预测，提高吞吐量
- **内存管理**：定期清理缓存，监控内存使用

### 12.4 监控指标
- 活跃请求数
- 模型内存使用
- 平均响应时间
- 请求成功率
- 并发用户数

---

*本文档版本：MVP v1.0*  
*最后更新：2024年11月*  
*文档维护：Kronos A股预测系统团队*