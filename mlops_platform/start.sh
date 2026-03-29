#!/bin/bash

# MLOps Platform Startup Script

echo "🚀 Starting MLOps Platform..."

# Создаем необходимые директории
mkdir -p /workspace/mlops_platform/{models,hdfs_storage,pipelines}

# Проверяем наличие Docker
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "✅ Docker found, starting with docker-compose..."
    cd /workspace/mlops_platform
    docker-compose up --build
else
    echo "⚠️  Docker not found, starting in local mode..."
    
    # Установка зависимостей если нужно
    if ! python -c "import flask" 2>/dev/null; then
        echo "📦 Installing Python dependencies..."
        pip install Flask flask-cors PyJWT numpy scikit-learn joblib requests pandas
    fi
    
    # Запуск backend
    echo "🔧 Starting backend on http://localhost:5000..."
    cd /workspace/mlops_platform/backend
    python app.py &
    BACKEND_PID=$!
    
    echo ""
    echo "✅ Backend started!"
    echo ""
    echo "📱 Open frontend/index.html in your browser or run:"
    echo "   python -m http.server 8080 -d /workspace/mlops_platform/frontend"
    echo ""
    echo "🔐 Login: admin / 999999"
    echo ""
    echo "Press Ctrl+C to stop"
    
    wait $BACKEND_PID
fi
