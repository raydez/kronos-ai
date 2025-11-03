#!/bin/bash

# Kronos A股预测分析系统 Docker 启动脚本

echo "🚀 启动 Kronos A股预测分析系统..."

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

# 创建必要的数据目录
echo "📁 创建数据目录..."
mkdir -p backend/data/{models,cache,logs,database}

# 构建并启动服务
echo "🔨 构建并启动服务..."
if command -v docker-compose &> /dev/null; then
    docker-compose up --build -d
else
    docker compose up --build -d
fi

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo "🔍 检查服务状态..."
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    docker compose ps
fi

echo ""
echo "✅ 服务启动完成！"
echo "🌐 前端访问地址: http://localhost:3000"
echo "🔗 后端API地址: http://localhost:8000"
echo "📖 API文档地址: http://localhost:8000/docs"
echo ""
echo "📋 常用命令:"
echo "  查看日志: docker-compose logs -f"
echo "  停止服务: docker-compose down"
echo "  重启服务: docker-compose restart"
echo "  查看状态: docker-compose ps"