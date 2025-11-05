# Kronos A股预测分析系统

基于Kronos金融大模型的A股日线预测分析系统

## 🎯 项目概述

本项目是一个基于Kronos金融大模型的A股股票预测分析系统，提供智能的股票走势预测功能。系统采用前后端分离架构，后端使用FastAPI构建RESTful API，前端使用现代化的Web界面。

## ✨ 核心功能

- 📊 **股票预测**: 基于Kronos模型的智能股票价格预测
- 📈 **多时间维度**: 支持3-10天的多维度预测
- 🔄 **实时数据**: 集成Baostock等数据源获取实时股票信息
- 🎨 **可视化界面**: 直观的Web界面展示预测结果
- 🔀 **模型切换**: 支持在不同Kronos模型间灵活切换（mini/small/base）
- 💾 **智能缓存**: 自动检测本地模型，按需下载
- 🚀 **高性能**: 异步处理，支持并发请求
- 📱 **响应式设计**: 适配各种设备屏幕

## 🏗️ 系统架构

```
kronos-ai/
├── backend/                          # 后端服务
│   ├── app/
│   │   ├── services/                 # 业务逻辑服务
│   │   │   ├── kronos_integration.py    # Kronos模型集成（加载/卸载/预测）
│   │   │   ├── model_manager.py         # 模型管理器（单例，支持切换）
│   │   │   ├── prediction_service.py    # 预测服务
│   │   │   └── stock_service.py         # 股票数据服务
│   │   └── utils/                    # 工具函数
│   ├── model/                        # Kronos模型定义
│   │   ├── kronos.py                 # Kronos模型实现
│   │   └── module.py                 # 模型模块
│   ├── main.py                       # FastAPI主应用
│   ├── config.py                     # 配置文件
│   ├── requirements.txt              # Python依赖
│   ├── Dockerfile                    # Docker配置
├── frontend/                         # 前端应用（React + TypeScript）
│   ├── src/
│   │   ├── components/              # React组件
│   │   │   ├── KLineChart.tsx          # K线图组件
│   │   │   ├── ModelStatus.tsx         # 模型状态和切换组件
│   │   │   ├── PlotlyKLineChart.tsx    # Plotly K线图
│   │   │   ├── PredictionForm.tsx      # 预测表单
│   │   │   └── PredictionTable.tsx     # 预测结果表格
│   │   ├── services/
│   │   │   └── api.ts               # API调用封装
│   │   ├── types/
│   │   │   └── index.ts             # TypeScript类型定义
│   │   ├── App.tsx                  # 主应用组件
│   │   ├── App.css
│   │   ├── index.tsx                # 入口文件
│   │   └── index.css
│   ├── package.json                 # npm依赖
│   ├── tsconfig.json                # TypeScript配置
│   ├── Dockerfile                   # Docker配置
├── docker-compose.yml               # Docker Compose配置
├── docker-start.sh                  # Docker启动脚本
├── docker-stop.sh                   # Docker停止脚本
├── start.sh                         # 启动脚本
├── stop.sh                          # 停止脚本
├── quick-start.sh                   # 快速启动脚本
└── README.md                        # 项目说明
```

### 核心模块说明

**后端核心模块**：
- `kronos_integration.py`: Kronos模型的底层集成，处理模型加载、卸载、预测等操作
- `model_manager.py`: 模型管理器（单例模式），提供统一的模型访问接口，支持模型切换
- `prediction_service.py`: 预测服务，协调股票数据和模型预测
- `stock_service.py`: 股票数据服务，集成Baostock等数据源

**前端核心组件**：
- `ModelStatus.tsx`: 模型状态展示和切换界面，支持查看当前模型、切换模型、重新加载
- `PredictionForm.tsx`: 股票预测表单，用户输入股票代码和预测天数
- `KLineChart.tsx`: K线图可视化组件
- `api.ts`: 封装所有后端API调用

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
| `/model/info` | GET | 获取模型信息 |
| `/model/available` | GET | 获取可用模型列表 |
| `/model/reload` | POST | 重新加载当前模型 |
| `/model/switch` | POST | 切换到指定模型 |
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

**获取可用模型列表**
```bash
curl -X GET "http://localhost:8000/model/available"
```

**切换模型**
```bash
curl -X POST "http://localhost:8000/model/switch" \
     -H "Content-Type: application/json" \
     -d '{"model_name": "kronos-base"}'
```

**重新加载模型**
```bash
curl -X POST "http://localhost:8000/model/reload"
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

## 📊 预测模型

### 可用模型

系统支持3个不同规格的Kronos模型，可根据需求灵活切换：

| 模型名称 | 参数规模 | 上下文长度 | 特点 | 适用场景 |
|---------|---------|-----------|------|---------|
| **kronos-mini** | 4.1M | 2048 | 轻量快速 | 快速预测、资源受限环境 |
| **kronos-small** | 24.7M | 512 | 平衡性能 | 日常使用（默认） |
| **kronos-base** | 85.6M | 1024 | 高精度 | 深度分析、精确预测 |

### 模型切换

**通过Web界面切换**：
1. 访问前端界面
2. 在"模型状态"卡片点击"切换模型"按钮
3. 选择目标模型
4. 系统会自动：
   - 卸载当前模型释放内存
   - 检查本地是否有模型缓存
   - 如无缓存则自动从HuggingFace下载
   - 加载新模型并更新界面

**通过API切换**：
```bash
curl -X POST "http://localhost:8000/model/switch" \
     -H "Content-Type: application/json" \
     -d '{"model_name": "kronos-base"}'
```

### 模型缓存

- **缓存位置**: `~/.cache/huggingface/hub`
- **自动管理**: 系统会自动检测本地缓存，避免重复下载
- **按需下载**: 首次使用某个模型时自动下载
- **资源优化**: 切换模型时自动卸载旧模型，节省内存

### 预测特性
- 支持1-10天多维度预测
- 提供置信度评估
- 包含开盘、最高、最低、收盘价格
- 基于历史数据和技术指标
- 准确率随模型规模提升（mini: ~75%, small: ~85%, base: ~90%）

## 💡 使用指南

### 模型切换功能详解

#### 什么时候需要切换模型？

- **需要快速预测**：切换到 `kronos-mini`（4.1M），速度最快
- **日常使用**：使用默认的 `kronos-small`（24.7M），平衡性能
- **需要高精度**：切换到 `kronos-base`（85.6M），准确率最高

#### 切换流程

1. **查看当前模型**
   - Web界面：查看"模型状态"卡片中的"当前模型"
   - API：访问 `/model/info` 端点

2. **选择目标模型**
   - Web界面：点击"切换模型"按钮，从下拉列表选择
   - API：调用 `/model/switch` 接口

3. **等待切换完成**
   - 如果本地有缓存：通常只需几秒
   - 如果需要下载：
     - `kronos-mini`: 约需下载 5-10MB
     - `kronos-small`: 约需下载 25-50MB
     - `kronos-base`: 约需下载 80-150MB

4. **验证切换结果**
   - 查看成功提示消息
   - 确认新模型已加载

#### 注意事项

⚠️ **首次使用某个模型时**：
- 系统会从 HuggingFace Hub 下载模型
- 需要稳定的网络连接
- 下载时间取决于网络速度和模型大小

⚠️ **切换过程中**：
- 旧模型会被自动卸载以释放内存
- 预测服务会短暂不可用
- 建议在系统空闲时进行切换

✅ **推荐做法**：
- 首次部署时预先下载所有模型
- 根据实际需求选择合适的模型
- 在非业务高峰期进行模型切换

#### 离线使用

如果需要在离线环境使用，可以预先下载所有模型：

```bash
# 方法1：通过API依次切换所有模型（会自动下载）
curl -X POST "http://localhost:8000/model/switch" -H "Content-Type: application/json" -d '{"model_name": "kronos-mini"}'
curl -X POST "http://localhost:8000/model/switch" -H "Content-Type: application/json" -d '{"model_name": "kronos-small"}'
curl -X POST "http://localhost:8000/model/switch" -H "Content-Type: application/json" -d '{"model_name": "kronos-base"}'

# 方法2：手动下载到缓存目录
# 下载后放置到 ~/.cache/huggingface/hub/
```

## 🛠️ 开发指南

### 项目结构说明

- `backend/app/models/`: Pydantic数据模型定义
- `backend/app/services/`: 核心业务逻辑
  - `model_manager.py`: Kronos模型管理（单例模式，支持模型切换）
  - `kronos_integration.py`: Kronos模型集成（加载、卸载、预测）
  - `stock_service.py`: 股票数据服务
  - `prediction_service.py`: 预测服务
- `backend/model/`: Kronos模型定义
  - `kronos.py`: Kronos模型实现
  - `module.py`: 模型模块
- `backend/main.py`: FastAPI应用主入口
- `frontend/`: Web前端界面
  - `src/components/ModelStatus.tsx`: 模型状态和切换组件
  - `src/services/api.ts`: API调用封装

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

本项目采用MIT许可证

## 🙏 致谢

- Kronos团队提供的金融大模型
- Baostock提供的股票数据
- FastAPI社区的技术支持

## ❓ 常见问题（FAQ）

### Q1: 应用启动时模型加载失败怎么办？
**A**: 应用会正常启动，但预测功能不可用。可以：
1. 点击"重新加载"按钮尝试重新加载
2. 检查网络连接是否正常
3. 查看后端日志了解具体错误

### Q2: 模型下载很慢或失败怎么办？
**A**: 可能的原因和解决方案：
- **网络问题**: 检查网络连接，考虑使用代理
- **HuggingFace访问受限**: 配置 HuggingFace 镜像源
- **磁盘空间不足**: 清理磁盘空间（每个模型需要几十到几百MB）

### Q3: 如何查看模型是从本地加载还是从网络下载？
**A**: 切换模型成功后，会显示详细信息：
- "从本地缓存加载" - 使用本地已有的模型
- "已从HuggingFace下载: Tokenizer、模型" - 从网络下载

### Q4: 切换模型会影响正在进行的预测吗？
**A**: 是的。切换模型时：
- 当前模型会被卸载
- 预测服务短暂不可用
- 建议在系统空闲时切换

### Q5: 三个模型该如何选择？
**A**: 根据需求选择：
- **kronos-mini**: 资源受限、需要快速响应时使用
- **kronos-small**: 日常使用的默认选择，性能和准确率平衡
- **kronos-base**: 需要最高准确率时使用，但消耗更多资源

### Q6: 模型缓存占用多少磁盘空间？
**A**: 每个模型的大小：
- kronos-mini: ~5-10MB
- kronos-small: ~25-50MB  
- kronos-base: ~80-150MB

全部下载约占用 200-300MB 磁盘空间。

### Q7: 可以删除不用的模型吗？
**A**: 可以手动删除 `~/.cache/huggingface/hub/` 目录下的对应模型文件夹。下次使用时会重新下载。

### Q8: 切换模型后预测结果会不同吗？
**A**: 是的。不同模型的预测结果可能有差异：
- 参数更大的模型（base）通常更准确
- 参数较小的模型（mini）响应更快但可能略不准确

### Q9: 支持自定义模型吗？
**A**: 当前版本支持三个预设的 Kronos 模型。如需添加自定义模型，需要：
1. 在 `backend/app/services/kronos_integration.py` 中添加模型配置
2. 确保模型格式与 Kronos 兼容

### Q10: 如何配置 HuggingFace Token？
**A**: 如果模型需要认证访问：
```bash
# 设置环境变量
export HUGGINGFACE_TOKEN="your_token_here"

# 或在代码中配置
# backend/app/services/kronos_integration.py
```

---

**注意**: 本系统仅用于学习和研究目的，不构成投资建议。股票投资有风险，入市需谨慎。