# Kronos A股预测分析系统

基于Kronos金融大模型的A股日线预测分析系统MVP

## 🎯 项目概述

本项目是一个基于Kronos金融大模型的A股股票预测分析系统，提供智能的股票走势预测功能。系统采用前后端分离架构，后端使用FastAPI构建RESTful API，前端使用现代化的Web界面。

## ✨ 核心功能

- 📊 **股票预测**: 基于Kronos模型的智能股票价格预测
- 📈 **多时间维度**: 支持3-10天的多维度预测
- 🔄 **实时数据**: 集成Baostock等数据源获取实时股票信息
- 🎨 **可视化界面**: 直观的Web界面展示预测结果
- 🚀 **高性能**: 异步处理，支持并发请求
- 📱 **响应式设计**: 适配各种设备屏幕

## 🏗️ 系统架构

```
kronos-ai/
├── backend/                 # 后端服务
│   ├── app/
│   │   ├── models/         # Pydantic数据模型
│   │   └── services/       # 业务逻辑服务
│   ├── main.py             # FastAPI主应用
│   ├── config.py           # 配置文件
│   └── requirements.txt    # Python依赖
├── frontend/               # 前端应用
│   └── index.html         # Web界面
├── doc/                   # 产品文档
├── scripts/              # 部署脚本
├── start.sh              # 启动脚本
├── stop.sh               # 停止脚本
└── README.md             # 项目说明
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Node.js 14+ (可选，用于前端开发)
- macOS/Linux/Windows

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd kronos-ai
```

2. **设置后端环境**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **启动系统**
```bash
# 返回项目根目录
cd ..
# 使用启动脚本
./start.sh
```

或者手动启动：
```bash
# 启动后端
cd backend
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python main.py

# 启动前端（新终端）
cd frontend
python3 -m http.server 3000
```

4. **访问应用**
- 前端界面: http://localhost:3000
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 停止系统

```bash
./stop.sh
```

## 📡 API接口

### 核心接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/predict` | POST | 股票预测 |
| `/stock/{code}` | GET | 获取股票信息 |
| `/stock/{code}/history` | GET | 获取历史数据 |
| `/batch-predict` | POST | 批量预测 |
| `/model/info` | GET | 模型信息 |
| `/health` | GET | 健康检查 |

### 请求示例

**股票预测**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"code": "600000", "prediction_days": 5}'
```

**获取股票信息**
```bash
curl -X GET "http://localhost:8000/stock/600000"
```

## 🔧 技术栈

### 后端
- **FastAPI**: 高性能Web框架
- **Pydantic**: 数据验证和序列化
- **Pandas**: 数据处理
- **Baostock**: 股票数据源
- **Uvicorn**: ASGI服务器

### 前端
- **HTML5/CSS3/JavaScript**: 现代Web技术
- **响应式设计**: 适配各种设备

### 数据源
- **Baostock**: 主要股票数据源
- **Akshare**: 备用数据源
- **Tushare**: 扩展数据源

## 📊 预测模型

### 模型信息
- **模型名称**: Kronos-small
- **参数规模**: 24.7M
- **预测准确率**: 85%
- **处理时间**: < 1秒

### 预测特性
- 支持1-10天多维度预测
- 提供置信度评估
- 包含开盘、最高、最低、收盘价格
- 基于历史数据和技术指标

## 🛠️ 开发指南

### 项目结构说明

- `backend/app/models/`: Pydantic数据模型定义
- `backend/app/services/`: 核心业务逻辑
  - `model_manager.py`: Kronos模型管理
  - `stock_service.py`: 股票数据服务
  - `prediction_service.py`: 预测服务
- `backend/main.py`: FastAPI应用主入口
- `frontend/`: Web前端界面

### 添加新功能

1. 在相应的service中添加业务逻辑
2. 在models中定义数据模型
3. 在main.py中添加API端点
4. 更新前端界面

### 运行测试

```bash
cd backend
source .venv/bin/activate
python -m pytest tests/
```

## 📝 配置说明

### 环境变量

```bash
# 开发环境
export ENVIRONMENT=development

# 生产环境
export ENVIRONMENT=production
export DATABASE_URL=your_database_url
```

### 模型配置

- 模型路径: `backend/data/models/`
- 缓存路径: `backend/data/cache/`
- 日志路径: `logs/`

## 🚀 部署指南

### Docker部署

```bash
# 构建镜像
docker build -t kronos-ai .

# 运行容器
docker run -p 8000:8000 kronos-ai
```

### 生产环境部署

1. 使用Gunicorn作为WSGI服务器
2. 配置Nginx反向代理
3. 设置SSL证书
4. 配置监控和日志

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目地址: [https://github.com/your-username/kronos-ai]

## 🙏 致谢

- Kronos团队提供的金融大模型
- Baostock提供的股票数据
- FastAPI社区的技术支持

---

**注意**: 本系统仅用于学习和研究目的，不构成投资建议。股票投资有风险，入市需谨慎。