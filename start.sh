#!/bin/bash

# MLOps Platform - Start Script
# This script starts the MLOps platform with backend and frontend services

echo "🚀 Starting MLOps Platform..."

# Create necessary directories
mkdir -p /workspace/models /workspace/hdfs_storage /workspace/pipelines

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "🐳 Using Docker deployment..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose up --build -d
    elif docker compose version &> /dev/null; then
        docker compose up --build -d
    else
        echo "❌ Neither docker-compose nor docker compose found"
        exit 1
    fi
    
    echo "✅ Services started!"
    echo "📊 Frontend: http://localhost:8080"
    echo "🔧 Backend API: http://localhost:5000"
    echo "🔑 Login: admin / 999999"
else
    echo "🐍 Using Python deployment (no Docker)..."
    
    # Install dependencies
    echo "Installing Python dependencies..."
    pip install -q Flask flask-cors PyJWT numpy scikit-learn joblib
    
    # Start backend in background
    echo "Starting backend server..."
    cd /workspace/backend
    python app.py &
    BACKEND_PID=$!
    
    # Wait for backend to start
    sleep 3
    
    echo "✅ Backend started on http://localhost:5000"
    echo "📁 To access the frontend, serve /workspace/frontend/index.html with a web server"
    echo "   or open it directly in a browser"
    echo ""
    echo "🔑 Login credentials: admin / 999999"
    echo ""
    echo "Press Ctrl+C to stop the server"
    
    # Keep running
    wait $BACKEND_PID
fi
