"""
Hadoop Integration - Model Storage and Retrieval from HDFS
Simulated HDFS operations for environments without actual Hadoop
"""
import os
import shutil
import json
from datetime import datetime


class HadoopIntegration:
    """
    Integration with Hadoop HDFS for model storage.
    In production, this would use pyhdfs or hdfs libraries.
    For demo purposes, we simulate HDFS with local filesystem.
    """
    
    def __init__(self, hdfs_root="/tmp/hdfs", hadoop_host="localhost", hadoop_port=9870):
        self.hdfs_root = hdfs_root
        self.hadoop_host = hadoop_host
        self.hadoop_port = hadoop_port
        self.models_path = os.path.join(hdfs_root, "mlops", "models")
        
        # Create directories
        os.makedirs(self.models_path, exist_ok=True)
    
    def connect(self):
        """Simulate connection to HDFS"""
        print(f"Connecting to HDFS at {self.hadoop_host}:{self.hadoop_port}")
        # In production: client = hdfs.InsecureClient(f'http://{self.hadoop_host}:{self.hadoop_port}')
        return True
    
    def upload_model(self, local_model_path, local_info_path, version=None):
        """Upload model to HDFS"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        hdfs_model_path = os.path.join(self.models_path, f"model_{version}.pkl")
        hdfs_info_path = os.path.join(self.models_path, f"info_{version}.json")
        
        # Simulate HDFS upload (in production, use client.upload())
        shutil.copy2(local_model_path, hdfs_model_path)
        shutil.copy2(local_info_path, hdfs_info_path)
        
        print(f"Model uploaded to HDFS: {hdfs_model_path}")
        
        return {
            "version": version,
            "hdfs_path": hdfs_model_path,
            "status": "uploaded"
        }
    
    def download_model(self, version, local_dir="/app/models"):
        """Download model from HDFS"""
        hdfs_model_path = os.path.join(self.models_path, f"model_{version}.pkl")
        hdfs_info_path = os.path.join(self.models_path, f"info_{version}.json")
        
        if not os.path.exists(hdfs_model_path):
            raise FileNotFoundError(f"Model {version} not found in HDFS")
        
        local_model_path = os.path.join(local_dir, f"model_{version}.pkl")
        local_info_path = os.path.join(local_dir, f"info_{version}.json")
        
        # Simulate HDFS download (in production, use client.download())
        os.makedirs(local_dir, exist_ok=True)
        shutil.copy2(hdfs_model_path, local_model_path)
        shutil.copy2(hdfs_info_path, local_info_path)
        
        print(f"Model downloaded from HDFS: {local_model_path}")
        
        return {
            "version": version,
            "local_path": local_model_path,
            "status": "downloaded"
        }
    
    def list_models(self):
        """List all models in HDFS"""
        models = []
        for f in os.listdir(self.models_path):
            if f.startswith("model_") and f.endswith(".pkl"):
                version = f.replace("model_", "").replace(".pkl", "")
                info_path = os.path.join(self.models_path, f"info_{version}.json")
                if os.path.exists(info_path):
                    with open(info_path, 'r') as finfo:
                        info = json.load(finfo)
                    models.append({
                        "version": version,
                        "hdfs_path": os.path.join(self.models_path, f),
                        "accuracy": info.get("accuracy", "N/A"),
                        "trained_at": info.get("trained_at", "N/A"),
                        "model_type": info.get("model_type", "N/A")
                    })
        return models
    
    def get_model_status(self, version):
        """Get status of a specific model in HDFS"""
        hdfs_model_path = os.path.join(self.models_path, f"model_{version}.pkl")
        hdfs_info_path = os.path.join(self.models_path, f"info_{version}.json")
        
        exists = os.path.exists(hdfs_model_path)
        
        return {
            "version": version,
            "exists": exists,
            "hdfs_path": hdfs_model_path,
            "size_bytes": os.path.getsize(hdfs_model_path) if exists else 0
        }
    
    def delete_model(self, version):
        """Delete model from HDFS"""
        hdfs_model_path = os.path.join(self.models_path, f"model_{version}.pkl")
        hdfs_info_path = os.path.join(self.models_path, f"info_{version}.json")
        
        deleted = False
        if os.path.exists(hdfs_model_path):
            os.remove(hdfs_model_path)
            deleted = True
        if os.path.exists(hdfs_info_path):
            os.remove(hdfs_info_path)
            deleted = True
        
        return {
            "version": version,
            "deleted": deleted,
            "status": "deleted" if deleted else "not_found"
        }
    
    def health_check(self):
        """Check HDFS connection health"""
        try:
            # Check if HDFS root is accessible
            is_accessible = os.path.exists(self.hdfs_root)
            return {
                "status": "healthy" if is_accessible else "unhealthy",
                "hdfs_root": self.hdfs_root,
                "host": self.hadoop_host,
                "port": self.hadoop_port
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


if __name__ == "__main__":
    # Test the Hadoop integration
    hdfs = HadoopIntegration()
    
    # Health check
    health = hdfs.health_check()
    print(f"HDFS Health: {health}")
    
    # List models
    models = hdfs.list_models()
    print(f"\nModels in HDFS: {len(models)}")
    for m in models:
        print(f"  - {m['version']}: {m['model_type']} (accuracy: {m['accuracy']})")
