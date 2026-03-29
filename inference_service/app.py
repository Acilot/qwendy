"""
Inference Service - REST API for model predictions
"""
import os
import sys
from flask import Flask, request, jsonify
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_pipeline.pipeline import MLPipeline

app = Flask(__name__)
pipeline = MLPipeline(models_dir="/app/models")

# Auto-load latest model on startup
try:
    pipeline.load_model("latest")
    print("Latest model loaded successfully")
except FileNotFoundError:
    print("No model found. Please train a model first.")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "inference"})


@app.route('/predict', methods=['POST'])
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


@app.route('/models', methods=['GET'])
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


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get current model info"""
    try:
        return jsonify({
            "success": True,
            "model_info": pipeline.model_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/model/load/<version>', methods=['POST'])
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
