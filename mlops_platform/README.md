# MLOps Platform with Real HDFS Integration

## 🎯 Features

- **Real HDFS Integration** - Full Hadoop cluster (NameNode + DataNode) for storing ML models
- **ML Pipeline Management** - Create, run, and monitor ML training pipelines
- **Model Repository** - Store and manage models in HDFS with metadata
- **Web Interface** - Modern Vue.js dashboard for all operations
- **External Model Upload** - Load models from URLs, Nexus repositories, or files
- **Model Testing** - Test models with sample data directly from UI
- **JWT Authentication** - Secure login system (admin/999999)

## 📁 Project Structure

```
mlops_platform/
├── backend/
│   ├── app.py              # Flask API server
│   ├── Dockerfile          # Backend container
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Vue.js SPA
│   ├── nginx.conf          # Nginx configuration
│   └── Dockerfile          # Frontend container
├── hdfs/
│   ├── Dockerfile.namenode # HDFS NameNode
│   └── Dockerfile.datanode # HDFS DataNode
├── docker-compose.yml      # Orchestration
└── README.md
```

## 🚀 Quick Start

### With Docker (Recommended)

```bash
cd /workspace/mlops_platform
docker-compose up --build
```

Wait 2-3 minutes for HDFS to initialize, then access:
- **Frontend**: http://localhost:8080
- **Backend API**: http://localhost:5000
- **HDFS NameNode UI**: http://localhost:9870

**Login**: admin / 999999

### Without Docker

```bash
# Install dependencies
pip install Flask flask-cors PyJWT numpy scikit-learn joblib requests pyarrow hdfs pandas

# Start backend
cd backend
python app.py

# Open frontend/index.html in browser or serve with any web server
```

## 📊 Usage Guide

### 1. Create ML Pipeline
1. Go to "Pipelines" tab
2. Click "Create Pipeline"
3. Enter name and select model type
4. Watch real-time progress through 7 stages

### 2. View Models
- Navigate to "Models" tab
- See all models from local storage and HDFS
- Test models inline with sample data

### 3. Browse HDFS
- Go to "HDFS Browser" tab
- Navigate directories
- Download models to local storage
- View file details and sizes

### 4. Upload External Models
- Go to "Upload Model" tab
- Enter URL (direct link or Nexus repository)
- Monitor upload progress
- Model automatically validated and saved to HDFS

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/login | POST | Authenticate user |
| /api/pipelines | GET | List all pipelines |
| /api/pipelines | POST | Create new pipeline |
| /api/models | GET | List all models |
| /api/models/:name/test | POST | Test model |
| /api/hdfs/status | GET | Get HDFS connection status |
| /api/hdfs/browse | POST | Browse HDFS directory |
| /api/hdfs/download | POST | Download file from HDFS |
| /api/upload/model | POST | Upload model from external source |

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Backend   │
│  (Vue.js)   │◀────│   (Flask)   │
│   :8080     │     │   :5000     │
└─────────────┘     └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼───┐ ┌────▼────┐
        │ NameNode  │ │DataNode│ │  Local  │
        │  :9870    │ │ :9864  │ │ Storage │
        │  :9000    │ │       │ │         │
        └───────────┘ └───────┘ └─────────┘
             HDFS Cluster
```

## 📦 Supported Model Types

- Random Forest
- Gradient Boosting
- Logistic Regression
- Neural Network (MLP)

## 🔐 Default Credentials

- **Username**: admin
- **Password**: 999999

## 🛠️ Troubleshooting

### HDFS not connecting
- Wait 2-3 minutes for HDFS initialization
- Check NameNode logs: `docker-compose logs namenode`
- Verify port 9870 is accessible

### Models not appearing
- Refresh the models list
- Check backend logs: `docker-compose logs backend`
- Ensure volumes are mounted correctly

### Build failures
- Clear Docker cache: `docker-compose build --no-cache`
- Check internet connection for base images
