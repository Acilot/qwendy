#!/bin/bash

# MLOps Platform - Quick Start Script
# This script helps you get started with the MLOps platform

set -e

echo "🚀 MLOps Platform - Quick Start"
echo "================================"
echo ""

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "✅ Docker and Docker Compose found!"
    echo ""
    echo "Starting all services with Docker Compose..."
    docker-compose up --build
    
elif command -v docker &> /dev/null && command -v docker compose &> /dev/null; then
    echo "✅ Docker and Docker Compose (v2) found!"
    echo ""
    echo "Starting all services with Docker Compose..."
    docker compose up --build
    
else
    echo "⚠️  Docker not found. Running in local mode..."
    echo ""
    
    # Install dependencies
    echo "📦 Installing Python dependencies..."
    pip install -q scikit-learn numpy joblib flask requests
    
    # Create models directory
    mkdir -p models
    
    # Train initial model
    echo "🤖 Training initial model..."
    cd ml_pipeline && python pipeline.py
    cd ..
    
    # Copy models to web_interface
    cp -r ml_pipeline/models/* web_interface/models/ 2>/dev/null || true
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "To start the web interface, run:"
    echo "  cd web_interface && python app.py"
    echo ""
    echo "Then open http://localhost:8080 in your browser"
fi
