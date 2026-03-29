# MLOps Platform

Modern MLOps platform with ML pipeline management, model training, HDFS storage integration, and a beautiful web interface.

## Features

- 🔐 **Authentication**: Secure login system (admin/999999)
- 🚀 **Pipeline Management**: Create and run ML pipelines with different algorithms
- 📊 **Real-time Monitoring**: Track pipeline status and progress
- 🤖 **Multiple Models**: Random Forest, Gradient Boosting, Logistic Regression, Neural Networks
- 💾 **HDFS Storage**: Simulated Hadoop storage for model artifacts
- 🎯 **Model Inference**: Test models directly from the UI
- 📈 **Metrics Dashboard**: Accuracy, Precision, Recall, F1-Score visualization
- 🐳 **Docker Ready**: Full containerization support

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Or with docker compose v2
docker compose up --build
```

Access the platform at:
- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:5000

### Option 2: Python (No Docker)

```bash
# Install dependencies
pip install Flask flask-cors PyJWT numpy scikit-learn joblib

# Start the backend
cd backend && python app.py

# Open frontend/index.html in a browser or serve with nginx
```

### Option 3: Start Script

```bash
./start.sh
```

## Login Credentials

- **Username**: admin
- **Password**: 999999

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend API   │
│   (Vue.js +     │     │   (Flask)       │
│    TailwindCSS) │◀────│                 │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ Models   │ │  HDFS    │ │Pipelines │
              │ Storage  │ │ Storage  │ │ Config   │
              └──────────┘ └──────────┘ └──────────┘
```

## API Endpoints

### Authentication
- `POST /api/login` - User login

### Pipelines
- `GET /api/pipelines` - List all pipelines
- `POST /api/pipelines` - Create new pipeline
- `POST /api/pipelines/<id>/run` - Run pipeline
- `GET /api/pipelines/<id>` - Get pipeline details
- `DELETE /api/pipelines/<id>` - Delete pipeline (admin)

### Models
- `GET /api/models` - List trained models
- `POST /api/models/<id>/predict` - Make prediction

### HDFS
- `GET /api/hdfs/status` - Get HDFS status
- `DELETE /api/hdfs/files/<filename>` - Delete file (admin)

### Stats
- `GET /api/stats` - Dashboard statistics

## Project Structure

```
/workspace
├── backend/
│   ├── app.py           # Flask API server
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile
├── frontend/
│   ├── index.html       # Vue.js SPA
│   ├── nginx.conf       # Nginx configuration
│   └── Dockerfile
├── models/              # Trained model files
├── hdfs_storage/        # HDFS simulated storage
├── pipelines/           # Pipeline configurations
├── docker-compose.yml   # Docker orchestration
└── start.sh            # Startup script
```

## Usage Guide

### 1. Create a Pipeline

1. Login with admin/999999
2. Go to "Pipelines" tab
3. Click "Create Pipeline"
4. Enter name, select model type, configure parameters
5. Click "Create"

### 2. Run Training

1. Find your pipeline in the list
2. Click "Run" button
3. Watch real-time progress
4. View metrics when completed

### 3. Test Model

1. Go to "Models" tab
2. Click "Test Prediction" on any model
3. Enter 20 feature values (comma-separated)
4. Click "Predict" to see results

### 4. Manage HDFS Storage

1. Go to "HDFS Storage" tab
2. View all stored model files
3. See file sizes, checksums, metadata
4. Admin can delete files

## Supported Model Types

- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential tree building
- **Logistic Regression**: Linear classification
- **Neural Network**: Multi-layer perceptron

## Technology Stack

**Backend:**
- Python 3.10
- Flask (Web Framework)
- Scikit-learn (ML Library)
- PyJWT (Authentication)
- Joblib (Model Serialization)

**Frontend:**
- Vue.js 3
- TailwindCSS
- Font Awesome Icons

**Infrastructure:**
- Docker & Docker Compose
- Nginx (Reverse Proxy)
