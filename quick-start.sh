#!/bin/bash

# 快速启动脚本 - 简化版本

echo "🚀 快速启动 Kronos A股预测分析系统..."

# 检查虚拟环境
if [ ! -d "backend/.venv" ]; then
    echo "❌ 虚拟环境不存在，请先运行 setup"
    exit 1
fi

# 启动后端
echo "🔄 启动后端服务..."
cd backend
source .venv/bin/activate
python main.py &
BACKEND_PID=$!

# 等待后端启动
sleep 3

# 启动前端
echo "🌐 启动前端服务..."
cd ../frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!

# 保存PID
echo $BACKEND_PID > ../.backend.pid
echo $FRONTEND_PID > ../.frontend.pid

echo ""
echo "✅ 系统启动完成！"
echo "📱 前端: http://localhost:3000"
echo "🔧 后端: http://localhost:8000"
echo ""
echo "🛑 停止: ./stop.sh"