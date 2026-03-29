"""
ML Pipeline - Training and Model Management
"""
import os
import joblib
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import json


class MLPipeline:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.model = None
        self.model_info = {}
        
    def prepare_data(self):
        """Generate sample data for training"""
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y
    
    def train(self, model_type="random_forest"):
        """Train a model"""
        print(f"Starting training with {model_type}...")
        X, y = self.prepare_data()
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.model_info = {
            "model_type": model_type,
            "accuracy": float(accuracy),
            "trained_at": datetime.now().isoformat(),
            "n_features": 10,
            "n_samples": len(X)
        }
        
        print(f"Training completed. Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        
        return self.model_info
    
    def save_model(self, version=None):
        """Save model to disk in both .pkl and .joblib formats for compatibility"""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save in .pkl format (legacy)
        model_path_pkl = os.path.join(self.models_dir, f"model_{version}.pkl")
        # Save in .joblib format (for inference service)
        model_path_joblib = os.path.join(self.models_dir, f"model_{version}.joblib")
        
        info_path = os.path.join(self.models_dir, f"info_{version}.json")
        
        joblib.dump(self.model, model_path_pkl)
        joblib.dump(self.model, model_path_joblib)
        
        with open(info_path, 'w') as f:
            json.dump(self.model_info, f, indent=2)
        
        # Save latest marker (point to joblib version)
        latest_info = {
            "latest_version": version, 
            "model_path": model_path_joblib,
            "model_path_pkl": model_path_pkl
        }
        with open(os.path.join(self.models_dir, "latest.json"), 'w') as f:
            json.dump(latest_info, f, indent=2)
        
        # Create symlink or copy for model_latest.joblib
        latest_model_path = os.path.join(self.models_dir, "model_latest.joblib")
        latest_info_path = os.path.join(self.models_dir, "model_latest_info.json")
        
        # Copy the latest model
        import shutil
        shutil.copy2(model_path_joblib, latest_model_path)
        shutil.copy2(info_path, latest_info_path)
        
        print(f"Model saved to {model_path_joblib} (and {model_path_pkl})")
        return version
    
    def load_model(self, version="latest"):
        """Load model from disk"""
        if version == "latest":
            latest_path = os.path.join(self.models_dir, "latest.json")
            if not os.path.exists(latest_path):
                raise FileNotFoundError("No latest model found. Train and save a model first.")
            with open(latest_path, 'r') as f:
                latest_info = json.load(f)
            version = latest_info["latest_version"]
        
        model_path = os.path.join(self.models_dir, f"model_{version}.pkl")
        info_path = os.path.join(self.models_dir, f"info_{version}.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {version} not found")
        
        self.model = joblib.load(model_path)
        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
        
        print(f"Loaded model {version} with accuracy {self.model_info['accuracy']:.4f}")
        return self.model
    
    def predict(self, features):
        """Make prediction"""
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        return {
            "prediction": int(prediction),
            "probabilities": {
                "class_0": float(probabilities[0]),
                "class_1": float(probabilities[1])
            }
        }
    
    def list_models(self):
        """List all available models"""
        models = []
        for f in os.listdir(self.models_dir):
            if f.startswith("model_") and f.endswith(".pkl"):
                version = f.replace("model_", "").replace(".pkl", "")
                info_path = os.path.join(self.models_dir, f"info_{version}.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as finfo:
                        info = json.load(finfo)
                    models.append({
                        "version": version,
                        "accuracy": info.get("accuracy", "N/A"),
                        "trained_at": info.get("trained_at", "N/A"),
                        "model_type": info.get("model_type", "N/A")
                    })
        return models


if __name__ == "__main__":
    # Test the pipeline
    pipeline = MLPipeline()
    
    # Train
    info = pipeline.train("random_forest")
    print(f"\nTraining info: {info}")
    
    # Save
    version = pipeline.save_model()
    print(f"\nSaved version: {version}")
    
    # Load
    pipeline.load_model("latest")
    
    # Predict
    test_features = [0.5, -0.3, 0.1, 0.8, -0.2, 0.4, -0.6, 0.9, -0.1, 0.3]
    result = pipeline.predict(test_features)
    print(f"\nPrediction result: {result}")
    
    # List models
    models = pipeline.list_models()
    print(f"\nAvailable models: {models}")
