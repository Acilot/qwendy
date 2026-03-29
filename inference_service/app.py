"""
Inference Service - REST API for model predictions
Autonomous service with embedded model loading logic
"""
import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path

class InferenceEngine:
    """Embedded inference engine for loading and using models"""
    
    def __init__(self, models_dir="/app/models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.model_info = {}
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def list_models(self):
        """List all available model versions"""
        if not self.models_dir.exists():
            return []
        
        models = []
        for model_file in self.models_dir.glob("model_*.joblib"):
            version = model_file.stem.replace("model_", "")
            info_file = model_file.with_suffix(".json")
            
            model_info = {"version": version, "file": str(model_file)}
            
            if info_file.exists():
                with open(info_file, 'r') as f:
                    model_info.update(json.load(f))
            
            models.append(model_info)
        
        # Check for latest symlink or file
        latest_file = self.models_dir / "model_latest.joblib"
        if latest_file.exists():
            for m in models:
                if m['version'] == 'latest':
                    m['is_latest'] = True
                    break
        
        return models
    
    def load_model(self, version="latest"):
        """Load a specific model version"""
        print(f"Attempting to load model: {version}, models_dir: {self.models_dir}")
        print(f"Files in models_dir: {list(self.models_dir.iterdir()) if self.models_dir.exists() else 'Directory does not exist'}")
        
        if version == "latest":
            model_path = self.models_dir / "model_latest.joblib"
            print(f"Looking for latest model at: {model_path}, exists: {model_path.exists()}")
            if not model_path.exists():
                # Find the most recent model
                models = sorted(self.models_dir.glob("model_*.joblib"), 
                              key=lambda x: x.stat().st_mtime, reverse=True)
                print(f"Found .joblib models: {models}")
                if models:
                    model_path = models[0]
                    print(f"Using most recent model: {model_path}")
                else:
                    raise FileNotFoundError("No models found")
        else:
            model_path = self.models_dir / f"model_{version}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {version} not found at {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load model info
        info_path = model_path.with_suffix(".json")
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)
        else:
            self.model_info = {"version": version, "path": str(model_path)}
        
        print(f"Model loaded successfully: {self.model_info}")
        return self.model_info
    
    def predict(self, features):
        """Make prediction with loaded model"""
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        probability = self.model.predict_proba(features_array)[0]
        
        return {
            "prediction": int(prediction),
            "confidence": float(max(probability)),
            "probabilities": [float(p) for p in probability]
        }

app = Flask(__name__)
# Use environment variable for models dir, default to /app/models for Docker, /workspace/models for local
models_dir = os.environ.get("MODELS_DIR", "/app/models")
engine = InferenceEngine(models_dir=models_dir)

# Auto-load latest model on startup
try:
    engine.load_model("latest")
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
        
        result = engine.predict(features)
        
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
        models = engine.list_models()
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
            "model_info": engine.model_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/model/load/<version>', methods=['POST'])
def load_model(version):
    """Load a specific model version"""
    try:
        engine.load_model(version)
        return jsonify({
            "success": True,
            "message": f"Model {version} loaded successfully",
            "model_info": engine.model_info
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
