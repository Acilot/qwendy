"""
MLOps Platform - Backend API
Modern Flask-based backend with authentication, pipeline management, and HDFS integration
"""

import os
import json
import time
import uuid
import threading
import hashlib
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from functools import wraps

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
import joblib

# Configuration
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)
app.config['SECRET_KEY'] = 'mlops-secret-key-2024'
app.config['MODELS_DIR'] = Path('/workspace/models')
app.config['HDFS_DIR'] = Path('/workspace/hdfs_storage')
app.config['PIPELINES_DIR'] = Path('/workspace/pipelines')

# Ensure directories exist
for dir_path in [app.config['MODELS_DIR'], app.config['HDFS_DIR'], app.config['PIPELINES_DIR']]:
    dir_path.mkdir(parents=True, exist_ok=True)

# In-memory storage (in production, use database)
users = {
    'admin': {
        'password': generate_password_hash('999999'),
        'role': 'admin'
    }
}

pipelines = {}
models_registry = {}
hdfs_metadata = {}

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token missing'}), 401
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['username']
        except:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    @token_required
    def decorated(current_user, *args, **kwargs):
        if users.get(current_user, {}).get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

# Auth endpoints
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if username in users and check_password_hash(users[username]['password'], password):
        token = jwt.encode({
            'username': username,
            'role': users[username]['role'],
            'exp': datetime.utcnow().timestamp() + 86400
        }, app.config['SECRET_KEY'], algorithm='HS256')
        return jsonify({'token': token, 'username': username, 'role': users[username]['role']})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/users', methods=['POST'])
@admin_required
def create_user(current_user):
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if username in users:
        return jsonify({'error': 'User already exists'}), 400
    
    users[username] = {
        'password': generate_password_hash(password),
        'role': role
    }
    return jsonify({'message': f'User {username} created'})

# Pipeline management
@app.route('/api/pipelines', methods=['GET'])
@token_required
def get_pipelines(current_user):
    return jsonify(list(pipelines.values()))

@app.route('/api/pipelines', methods=['POST'])
@token_required
def create_pipeline(current_user):
    data = request.get_json()
    pipeline_id = str(uuid.uuid4())[:8]
    
    pipeline = {
        'id': pipeline_id,
        'name': data.get('name', f'Pipeline-{pipeline_id}'),
        'model_type': data.get('model_type', 'random_forest'),
        'parameters': data.get('parameters', {}),
        'status': 'created',
        'progress': 0,
        'created_at': datetime.utcnow().isoformat(),
        'created_by': current_user,
        'metrics': None,
        'model_path': None
    }
    
    pipelines[pipeline_id] = pipeline
    return jsonify(pipeline)

@app.route('/api/pipelines/<pipeline_id>/run', methods=['POST'])
@token_required
def run_pipeline(current_user, pipeline_id):
    if pipeline_id not in pipelines:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    def execute_pipeline():
        try:
            pipeline = pipelines[pipeline_id]
            pipeline['status'] = 'running'
            pipeline['progress'] = 10
            
            # Generate synthetic data
            X, y = make_classification(
                n_samples=1000, 
                n_features=20, 
                n_informative=15,
                n_redundant=5,
                random_state=42
            )
            pipeline['progress'] = 30
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipeline['progress'] = 50
            
            # Create model based on type
            model_type = pipeline['model_type']
            params = pipeline.get('parameters', {})
            
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 10),
                    random_state=42
                )
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 5),
                    random_state=42
                )
            elif model_type == 'logistic_regression':
                model = LogisticRegression(
                    max_iter=params.get('max_iter', 1000),
                    random_state=42
                )
            elif model_type == 'neural_network':
                model = MLPClassifier(
                    hidden_layer_sizes=params.get('hidden_layers', (100,)),
                    max_iter=params.get('max_iter', 500),
                    random_state=42
                )
            else:
                model = RandomForestClassifier(random_state=42)
            
            pipeline['progress'] = 60
            model.fit(X_train, y_train)
            pipeline['progress'] = 80
            
            # Evaluate
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1': float(f1_score(y_test, y_pred, average='weighted'))
            }
            
            # Save model
            model_filename = f"{pipeline['name']}_{pipeline_id}.joblib"
            model_path = app.config['MODELS_DIR'] / model_filename
            joblib.dump(model, model_path)
            
            # Save to HDFS
            hdfs_path = app.config['HDFS_DIR'] / model_filename
            joblib.dump(model, hdfs_path)
            hdfs_metadata[model_filename] = {
                'size': os.path.getsize(hdfs_path),
                'created_at': datetime.utcnow().isoformat(),
                'pipeline_id': pipeline_id,
                'checksum': hashlib.md5(open(hdfs_path, 'rb').read()).hexdigest()
            }
            
            pipeline['status'] = 'completed'
            pipeline['progress'] = 100
            pipeline['metrics'] = metrics
            pipeline['model_path'] = str(model_path)
            models_registry[pipeline_id] = {
                'name': pipeline['name'],
                'path': str(model_path),
                'metrics': metrics,
                'created_at': pipeline['created_at']
            }
            
        except Exception as e:
            pipeline['status'] = 'failed'
            pipeline['error'] = str(e)
    
    thread = threading.Thread(target=execute_pipeline)
    thread.start()
    
    return jsonify({'message': 'Pipeline started', 'pipeline_id': pipeline_id})

@app.route('/api/pipelines/<pipeline_id>', methods=['GET'])
@token_required
def get_pipeline(current_user, pipeline_id):
    if pipeline_id not in pipelines:
        return jsonify({'error': 'Pipeline not found'}), 404
    return jsonify(pipelines[pipeline_id])

@app.route('/api/pipelines/<pipeline_id>', methods=['DELETE'])
@admin_required
def delete_pipeline(current_user, pipeline_id):
    if pipeline_id not in pipelines:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    pipeline = pipelines[pipeline_id]
    if pipeline.get('model_path'):
        model_path = Path(pipeline['model_path'])
        if model_path.exists():
            model_path.unlink()
    
    del pipelines[pipeline_id]
    return jsonify({'message': 'Pipeline deleted'})

# Models management
@app.route('/api/models', methods=['GET'])
@token_required
def get_models(current_user):
    return jsonify(list(models_registry.values()))

@app.route('/api/models/<model_id>/predict', methods=['POST'])
@token_required
def predict(current_user, model_id):
    data = request.get_json()
    features = data.get('features')
    
    if model_id not in models_registry:
        return jsonify({'error': 'Model not found'}), 404
    
    model_path = Path(models_registry[model_id]['path'])
    if not model_path.exists():
        return jsonify({'error': 'Model file not found'}), 404
    
    model = joblib.load(model_path)
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0] if hasattr(model, 'predict_proba') else None
    
    return jsonify({
        'prediction': int(prediction),
        'probabilities': probabilities.tolist() if probabilities is not None else None,
        'model_name': models_registry[model_id]['name']
    })

# HDFS management
@app.route('/api/hdfs/status', methods=['GET'])
@token_required
def get_hdfs_status(current_user):
    hdfs_dir = app.config['HDFS_DIR']
    files = []
    total_size = 0
    
    for f in hdfs_dir.glob('*.joblib'):
        size = os.path.getsize(f)
        total_size += size
        files.append({
            'name': f.name,
            'size': size,
            'created_at': datetime.fromtimestamp(os.path.getctime(f)).isoformat(),
            'metadata': hdfs_metadata.get(f.name, {})
        })
    
    return jsonify({
        'status': 'healthy',
        'root': str(hdfs_dir),
        'total_files': len(files),
        'total_size': total_size,
        'files': files
    })

@app.route('/api/hdfs/files/<filename>', methods=['DELETE'])
@admin_required
def delete_hdfs_file(current_user, filename):
    hdfs_path = app.config['HDFS_DIR'] / filename
    if not hdfs_path.exists():
        return jsonify({'error': 'File not found'}), 404
    
    hdfs_path.unlink()
    if filename in hdfs_metadata:
        del hdfs_metadata[filename]
    
    return jsonify({'message': f'File {filename} deleted'})

# Dashboard stats
@app.route('/api/stats', methods=['GET'])
@token_required
def get_stats(current_user):
    total_pipelines = len(pipelines)
    completed = sum(1 for p in pipelines.values() if p['status'] == 'completed')
    running = sum(1 for p in pipelines.values() if p['status'] == 'running')
    failed = sum(1 for p in pipelines.values() if p['status'] == 'failed')
    
    avg_accuracy = 0
    if completed > 0:
        accuracies = [p['metrics']['accuracy'] for p in pipelines.values() if p.get('metrics')]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
    
    return jsonify({
        'total_pipelines': total_pipelines,
        'completed': completed,
        'running': running,
        'failed': failed,
        'total_models': len(models_registry),
        'avg_accuracy': round(avg_accuracy, 4),
        'hdfs_files': len(hdfs_metadata)
    })

# Serve frontend
@app.route('/')
@app.route('/<path:path>')
def serve_frontend(path='index.html'):
    if path == '' or path.endswith('/'):
        path += 'index.html'
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
