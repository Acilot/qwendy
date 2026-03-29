# MLOps Platform - ML Pipeline with Hadoop Integration

A complete MLOps solution for training, deploying, and serving machine learning models with Hadoop integration.

## 🚀 Features

- **ML Pipeline**: Automated model training with scikit-learn
- **Inference Service**: REST API for model predictions
- **Web Interface**: User-friendly dashboard for model management
- **Hadoop Integration**: HDFS storage for model versioning
- **Docker Support**: Full containerization with docker-compose

## 📁 Project Structure

```
/workspace
├── ml_pipeline/           # Model training pipeline
│   ├── pipeline.py        # Main training code
│   ├── requirements.txt
│   └── Dockerfile
├── inference_service/     # Prediction API
│   ├── app.py            # Flask API server
│   ├── requirements.txt
│   └── Dockerfile
├── web_interface/         # Web dashboard
│   ├── app.py            # Main application
│   ├── index.html        # Frontend UI
│   ├── requirements.txt
│   └── Dockerfile
├── hadoop_integration/    # HDFS integration
│   ├── hdfs_client.py    # Hadoop client
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml     # Orchestration
└── README.md             # This file
```

## 🛠️ Quick Start

### Option 1: Run with Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Access the web interface
open http://localhost:8080
```

### Option 2: Run Locally (without Docker)

```bash
# Install dependencies
pip install -r ml_pipeline/requirements.txt
pip install -r inference_service/requirements.txt
pip install -r web_interface/requirements.txt

# Train a model
cd ml_pipeline && python pipeline.py

# Start the web interface
cd ../web_interface && python app.py

# Access http://localhost:8080
```

## 📊 API Endpoints

### Main Platform (Port 8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/health` | GET | Health check |
| `/api/predict` | POST | Make prediction |
| `/api/models` | GET | List all models |
| `/api/model/info` | GET | Current model info |
| `/api/model/load/<version>` | POST | Load specific model |
| `/api/train` | POST | Train new model |

### Inference Service (Port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/models` | GET | List all models |
| `/model/info` | GET | Current model info |

## 🔮 Making Predictions

### Via Web Interface
1. Open http://localhost:8080
2. Enter 10 feature values
3. Click "Get Prediction"

### Via API

```bash
curl -X POST http://localhost:8080/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.5, -0.3, 0.1, 0.8, -0.2, 0.4, -0.6, 0.9, -0.1, 0.3]
  }'
```

### Response Example

```json
{
  "success": true,
  "result": {
    "prediction": 1,
    "probabilities": {
      "class_0": 0.31,
      "class_1": 0.69
    }
  }
}
```

## 🎯 Training New Models

### Via Web Interface
Send POST request to `/api/train`:

```bash
curl -X POST http://localhost:8080/api/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest"}'
```

Supported model types:
- `random_forest` (default)
- `logistic_regression`

## 🐘 Hadoop Integration

The platform includes simulated HDFS integration for model storage:

```python
from hadoop_integration.hdfs_client import HadoopIntegration

hdfs = HadoopIntegration()

# Upload model to HDFS
hdfs.upload_model(
    local_model_path="/app/models/model_20240101_120000.pkl",
    local_info_path="/app/models/info_20240101_120000.json"
)

# Download model from HDFS
hdfs.download_model(version="20240101_120000")

# List all models in HDFS
models = hdfs.list_models()
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐
│  Web Interface  │────▶│  Inference API   │
│   (Port 8080)   │     │   (Port 5000)    │
└─────────────────┘     └────────┬─────────┘
                                 │
                          ┌──────▼──────┐
                          │   Models    │
                          │  Storage    │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │   Hadoop    │
                          │    HDFS     │
                          └─────────────┘
```

## 🧪 Testing

```bash
# Test ML Pipeline
cd ml_pipeline && python pipeline.py

# Test Hadoop Integration
cd hadoop_integration && python hdfs_client.py

# Test Inference Service
cd inference_service && python app.py &
curl http://localhost:5000/health
```

## 📝 License

MIT License
