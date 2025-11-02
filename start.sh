#!/bin/bash

# Kronos A股预测分析系统启动脚本

echo "🚀 启动 Kronos A股预测分析系统..."

# 检查是否在正确的目录
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    exit 1
fi

# 进入后端目录并启动服务
echo "📦 启动后端服务..."
cd backend

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行 setup.sh"
    exit 1
fi

# 激活虚拟环境
source .venv/bin/activate

# 启动FastAPI服务器（后台运行）
echo "🔄 启动 FastAPI 服务器..."
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload > ../logs/backend.log 2>&1 &
BACKEND_PID=$!

# 等待后端启动
echo "⏳ 等待后端服务启动..."
sleep 5

# 检查后端是否启动成功
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ 后端服务启动成功 (PID: $BACKEND_PID)"
else
    echo "❌ 后端服务启动失败"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# 返回根目录
cd ..

# 创建日志目录
mkdir -p logs

# 启动前端服务（简单的HTTP服务器）
echo "🌐 启动前端服务..."
cd frontend

# Python HTTP服务器
python3 -m http.server 3000 > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!

# 等待前端启动
sleep 2

echo "✅ 前端服务启动成功 (PID: $FRONTEND_PID)"

# 返回根目录
cd ..

# 保存PID到文件
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

echo ""
echo "🎉 Kronos A股预测分析系统启动完成！"
echo ""
echo "📱 前端地址: http://localhost:3000"
echo "🔧 后端API: http://localhost:8000"
echo "📖 API文档: http://localhost:8000/docs"
echo ""
echo "🛑 停止服务请运行: ./stop.sh"
echo ""

# 可选：自动打开浏览器
if command -v open > /dev/null; then
    echo "🌍 正在打开浏览器..."
    sleep 2
    open http://localhost:3000
fi