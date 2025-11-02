#!/bin/bash

# Kronos A股预测分析系统停止脚本

echo "🛑 停止 Kronos A股预测分析系统..."

# 停止后端服务
if [ -f ".backend.pid" ]; then
    BACKEND_PID=$(cat .backend.pid)
    if ps -p $BACKEND_PID > /dev/null; then
        echo "🔄 停止后端服务 (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        echo "✅ 后端服务已停止"
    else
        echo "⚠️  后端服务进程不存在"
    fi
    rm .backend.pid
else
    echo "⚠️  未找到后端服务PID文件"
fi

# 停止前端服务
if [ -f ".frontend.pid" ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null; then
        echo "🔄 停止前端服务 (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        echo "✅ 前端服务已停止"
    else
        echo "⚠️  前端服务进程不存在"
    fi
    rm .frontend.pid
else
    echo "⚠️  未找到前端服务PID文件"
fi

# 清理可能残留的进程
echo "🧹 清理残留进程..."
pkill -f "uvicorn main:app" 2>/dev/null
pkill -f "python -m http.server 3000" 2>/dev/null

echo ""
echo "✅ Kronos A股预测分析系统已完全停止"
echo ""