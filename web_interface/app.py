"""
Main Application - Combines Web Interface and Inference API
"""
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline.pipeline import MLPipeline

app = Flask(__name__, 
            static_folder='web_interface',
            static_url_path='')

# Use environment variable or default to ./models
MODELS_DIR = os.environ.get('MODELS_DIR', './models')
pipeline = MLPipeline(models_dir=MODELS_DIR)

# Auto-load latest model on startup
try:
    pipeline.load_model("latest")
    print("Latest model loaded successfully")
except FileNotFoundError:
    print("No model found. Please train a model first.")


# Serve web interface
@app.route('/')
def serve_index():
    return send_from_directory('web_interface', 'index.html')


# API endpoints
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "mlops-platform"})


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using loaded model"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' in request body"}), 400
        
        features = data['features']
        
        if len(features) != 10:
            return jsonify({"error": "Features must be a list of 10 numbers"}), 400
        
        result = pipeline.predict(features)
        
        return jsonify({
            "success": True,
            "result": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available models"""
    try:
        models = pipeline.list_models()
        return jsonify({
            "success": True,
            "models": models,
            "count": len(models)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get current model info"""
    try:
        return jsonify({
            "success": True,
            "model_info": pipeline.model_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/load/<version>', methods=['POST'])
def load_model(version):
    """Load a specific model version"""
    try:
        pipeline.load_model(version)
        return jsonify({
            "success": True,
            "message": f"Model {version} loaded successfully",
            "model_info": pipeline.model_info
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/train', methods=['POST'])
def train_model():
    """Train a new model"""
    try:
        data = request.get_json() or {}
        model_type = data.get('model_type', 'random_forest')
        
        # Train
        info = pipeline.train(model_type)
        
        # Save
        version = pipeline.save_model()
        
        # Reload to make it active
        pipeline.load_model("latest")
        
        return jsonify({
            "success": True,
            "message": f"Model trained and saved successfully",
            "version": version,
            "model_info": info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
