#!/bin/bash

# Kronos A股预测分析系统 Docker 停止脚本

echo "🛑 停止 Kronos A股预测分析系统..."

# 停止并删除容器
if command -v docker-compose &> /dev/null; then
    docker-compose down
else
    docker compose down
fi

echo "✅ 服务已停止"