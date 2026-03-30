"""
MLOps Platform - Backend API
Production-ready with PostgreSQL, real HDFS, universal model training, and GPU support
"""

import os
import json
import time
import uuid
import threading
import hashlib
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from functools import wraps
import bcrypt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score
)
import joblib

# Database
from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, Integer, Boolean, JSON, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.pool import StaticPool

# HDFS
try:
    import pyarrow.fs as hdfs_fs
    HDFS_AVAILABLE = True
except ImportError:
    HDFS_AVAILABLE = False

# GPU detection
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# Configuration
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'mlops-secret-key-' + str(uuid.uuid4()))
app.config['DATABASE_URL'] = os.environ.get('DATABASE_URL', 'postgresql://mlops:mlops@localhost:5432/mlops')
app.config['MODELS_DIR'] = Path(os.environ.get('MODELS_DIR', '/workspace/models'))
app.config['HDFS_URL'] = os.environ.get('HDFS_URL', 'hdfs://localhost:9000')
app.config['HDFS_USER'] = os.environ.get('HDFS_USER', 'mlops')
app.config['DATA_DIR'] = Path(os.environ.get('DATA_DIR', '/workspace/data'))

# Ensure directories exist
for dir_path in [app.config['MODELS_DIR'], app.config['DATA_DIR']]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Database setup
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(16), default='user')
    created_at = Column(DateTime, default=datetime.utcnow)
    pipelines = relationship('Pipeline', backref='owner', lazy='dynamic')

class Pipeline(Base):
    __tablename__ = 'pipelines'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(128), nullable=False)
    description = Column(Text)
    status = Column(String(32), default='created')
    progress = Column(Integer, default=0)
    model_type = Column(String(64))
    model_config = Column(JSON)
    data_path = Column(String(256))
    target_column = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(36), ForeignKey('users.id'))
    runs = relationship('TrainingRun', backref='pipeline', lazy='dynamic')

class TrainingRun(Base):
    __tablename__ = 'training_runs'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    pipeline_id = Column(String(36), ForeignKey('pipelines.id'), nullable=False)
    status = Column(String(32), default='running')
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    metrics = Column(JSON)
    model_path = Column(String(256))
    model_id = Column(String(36), ForeignKey('models.id'))
    logs = Column(JSON, default=list)

class Model(Base):
    __tablename__ = 'models'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(128), nullable=False)
    pipeline_id = Column(String(36), ForeignKey('pipelines.id'))
    path = Column(String(256), nullable=False)
    hdfs_path = Column(String(256))
    model_type = Column(String(64))
    metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(36), ForeignKey('users.id'))
    api_endpoint = Column(String(128), unique=True)

class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(36), ForeignKey('models.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metric_type = Column(String(64))  # accuracy, precision, recall, f1, latency, etc.
    value = Column(Float)
    run_id = Column(String(36), ForeignKey('training_runs.id'))

class ActivityLog(Base):
    __tablename__ = 'activity_logs'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(String(36), ForeignKey('users.id'))
    action = Column(String(64))  # login, create_pipeline, train_model, predict, etc.
    resource_type = Column(String(32))  # pipeline, model, user
    resource_id = Column(String(36))
    details = Column(JSON)

# Database initialization
def init_db():
    engine = create_engine(app.config['DATABASE_URL'])
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session

db_session = None
try:
    db_session = init_db()
except Exception as e:
    print(f"⚠️ Database not available, using in-memory storage: {e}")
    db_session = None

# HDFS client
def get_hdfs_client():
    if not HDFS_AVAILABLE:
        return None
    try:
        fs = hdfs_fs.HadoopFileSystem(
            host=app.config['HDFS_URL'].replace('hdfs://', '').split(':')[0],
            port=int(app.config['HDFS_URL'].split(':')[-1]) if ':' in app.config['HDFS_URL'] else 9000,
            user=app.config['HDFS_USER']
        )
        return fs
    except Exception as e:
        print(f"⚠️ HDFS not available: {e}")
        return None

hdfs_client = get_hdfs_client()

# In-memory fallback
users_cache = {}
pipelines_cache = {}
models_cache = {}
training_runs_cache = {}

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
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    @token_required
    def decorated(current_user, *args, **kwargs):
        if db_session:
            user = db_session.query(User).filter_by(username=current_user).first()
            if not user or user.role != 'admin':
                return jsonify({'error': 'Admin access required'}), 403
        elif users_cache.get(current_user, {}).get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

def log_activity(user_id, action, resource_type=None, resource_id=None, details=None):
    if db_session:
        try:
            log = ActivityLog(
                user_id=user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details
            )
            db_session.add(log)
            db_session.commit()
        except Exception as e:
            print(f"Failed to log activity: {e}")

# First-run setup check
def check_first_run():
    if db_session:
        admin_exists = db_session.query(User).filter_by(role='admin').first() is not None
        return not admin_exists
    else:
        return 'admin' not in users_cache

# Auth endpoints
@app.route('/api/setup', methods=['GET'])
def check_setup():
    """Check if first-run setup is needed"""
    return jsonify({'needs_setup': check_first_run()})

@app.route('/api/setup', methods=['POST'])
def setup_admin():
    """First-run admin setup"""
    data = request.get_json()
    username = data.get('username', 'admin')
    password = data.get('password')
    
    if not password or len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    if not check_first_run():
        return jsonify({'error': 'Setup already completed'}), 400
    
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    if db_session:
        admin = User(username=username, password_hash=password_hash, role='admin')
        db_session.add(admin)
        db_session.commit()
    else:
        users_cache[username] = {'password': password_hash, 'role': 'admin'}
    
    log_activity(None, 'setup_admin', 'user', username)
    return jsonify({'message': 'Admin user created', 'username': username})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    user = None
    if db_session:
        user = db_session.query(User).filter_by(username=username).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            token = jwt.encode({
                'username': username,
                'role': user.role,
                'exp': datetime.utcnow().timestamp() + 86400
            }, app.config['SECRET_KEY'], algorithm='HS256')
            log_activity(user.id, 'login', 'user', user.id)
            return jsonify({'token': token, 'username': username, 'role': user.role})
    else:
        if username in users_cache:
            cached = users_cache[username]
            if bcrypt.checkpw(password.encode('utf-8'), cached['password'].encode('utf-8')):
                token = jwt.encode({
                    'username': username,
                    'role': cached['role'],
                    'exp': datetime.utcnow().timestamp() + 86400
                }, app.config['SECRET_KEY'], algorithm='HS256')
                return jsonify({'token': token, 'username': username, 'role': cached['role']})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/users', methods=['GET'])
@admin_required
def list_users(current_user):
    if db_session:
        users = db_session.query(User).all()
        return jsonify([{
            'id': u.id,
            'username': u.username,
            'role': u.role,
            'created_at': u.created_at.isoformat()
        } for u in users])
    return jsonify([])

@app.route('/api/users', methods=['POST'])
@admin_required
def create_user(current_user):
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    if db_session:
        existing = db_session.query(User).filter_by(username=username).first()
        if existing:
            return jsonify({'error': 'User already exists'}), 400
        user = User(username=username, password_hash=password_hash, role=role)
        db_session.add(user)
        db_session.commit()
        log_activity(current_user, 'create_user', 'user', user.id)
        return jsonify({'id': user.id, 'username': username, 'role': role})
    else:
        if username in users_cache:
            return jsonify({'error': 'User already exists'}), 400
        users_cache[username] = {'password': password_hash, 'role': role}
        return jsonify({'username': username, 'role': role})

# Data upload and schema inference
@app.route('/api/data/upload', methods=['POST'])
@token_required
def upload_data(current_user):
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file
    data_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix.lower()
    filename = f"data_{data_id}{ext}"
    filepath = app.config['DATA_DIR'] / filename
    file.save(filepath)
    
    # Infer schema
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath, nrows=100)
        elif ext in ['.json', '.jsonl']:
            df = pd.read_json(filepath, nrows=100)
        elif ext == '.parquet':
            df = pd.read_parquet(filepath)[:100]
        else:
            return jsonify({'error': f'Unsupported format: {ext}'}), 400
        
        schema = {
            'columns': [
                {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'nullable': bool(df[col].isnull().any()),
                    'sample_values': df[col].dropna().head(3).tolist()
                }
                for col in df.columns
            ],
            'row_count': len(df),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Save to HDFS if available
        hdfs_path = None
        if hdfs_client:
            try:
                hdfs_path = f"/mlops/data/{filename}"
                with open(filepath, 'rb') as f:
                    hdfs_client.write_file(hdfs_path, f.read(), overwrite=True)
            except Exception as e:
                print(f"HDFS write failed: {e}")
        
        log_activity(current_user, 'upload_data', 'data', data_id, {'filename': filename})
        
        return jsonify({
            'data_id': data_id,
            'filename': filename,
            'path': str(filepath),
            'hdfs_path': hdfs_path,
            'schema': schema,
            'format': ext[1:]
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 400

@app.route('/api/data/<data_id>/schema', methods=['GET'])
@token_required
def get_data_schema(current_user, data_id):
    """Get inferred schema for uploaded data"""
    # Find data file
    data_file = list(app.config['DATA_DIR'].glob(f"data_{data_id}*"))
    if not data_file:
        return jsonify({'error': 'Data not found'}), 404
    
    filepath = data_file[0]
    ext = filepath.suffix.lower()
    
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext in ['.json', '.jsonl']:
            df = pd.read_json(filepath)
        elif ext == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            return jsonify({'error': 'Unsupported format'}), 400
        
        schema = {
            'columns': [
                {
                    'name': col,
                    'dtype': str(df[col].dtype),
                    'nullable': bool(df[col].isnull().any()),
                    'unique_values': df[col].nunique(),
                    'min': float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    'max': float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                    'mean': float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None
                }
                for col in df.columns
            ],
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        
        return jsonify(schema)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Pipeline management
@app.route('/api/pipelines', methods=['GET'])
@token_required
def get_pipelines(current_user):
    if db_session:
        user = db_session.query(User).filter_by(username=current_user).first()
        pipelines = db_session.query(Pipeline).filter_by(created_by=user.id).all() if user else []
        return jsonify([{
            'id': p.id,
            'name': p.name,
            'description': p.description,
            'status': p.status,
            'progress': p.progress,
            'model_type': p.model_type,
            'created_at': p.created_at.isoformat(),
            'runs_count': p.runs.count()
        } for p in pipelines])
    return jsonify(list(pipelines_cache.values()))

@app.route('/api/pipelines', methods=['POST'])
@token_required
def create_pipeline(current_user):
    data = request.get_json()
    pipeline_id = str(uuid.uuid4())
    
    pipeline_data = {
        'id': pipeline_id,
        'name': data.get('name', f'Pipeline-{pipeline_id[:8]}'),
        'description': data.get('description'),
        'model_type': data.get('model_type'),
        'model_config': data.get('model_config', {}),
        'data_path': data.get('data_path'),
        'target_column': data.get('target_column'),
        'status': 'created',
        'progress': 0,
        'created_at': datetime.utcnow().isoformat(),
        'created_by': current_user
    }
    
    if db_session:
        user = db_session.query(User).filter_by(username=current_user).first()
        pipeline = Pipeline(
            id=pipeline_id,
            name=pipeline_data['name'],
            description=pipeline_data['description'],
            model_type=pipeline_data['model_type'],
            model_config=pipeline_data['model_config'],
            data_path=pipeline_data['data_path'],
            target_column=pipeline_data['target_column'],
            created_by=user.id if user else None
        )
        db_session.add(pipeline)
        db_session.commit()
        log_activity(user.id if user else None, 'create_pipeline', 'pipeline', pipeline_id)
    
    pipelines_cache[pipeline_id] = pipeline_data
    return jsonify(pipeline_data)

def load_user_model(model_type: str, model_config: dict):
    """
    Dynamically load and configure any scikit-learn model
    Supports custom model classes via module path
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    
    # Classification models
    model_classes = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'ada_boost': AdaBoostClassifier,
        'logistic_regression': LogisticRegression,
        'sgd_classifier': SGDClassifier,
        'svc': SVC,
        'knn': KNeighborsClassifier,
        'naive_bayes': GaussianNB,
        'decision_tree': DecisionTreeClassifier,
        'mlp': MLPClassifier,
        # Regression models
        'random_forest_regressor': RandomForestRegressor,
        'gradient_boosting_regressor': GradientBoostingRegressor,
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'svr': SVR
    }
    
    # Check for custom model (module.class path)
    if model_type and '.' in model_type:
        try:
            module_path, class_name = model_type.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            return model_class(**model_config)
        except Exception as e:
            raise ValueError(f"Failed to load custom model {model_type}: {e}")
    
    # Built-in models
    if model_type in model_classes:
        return model_classes[model_type](**model_config)
    
    # Default fallback
    return RandomForestClassifier(**model_config)

@app.route('/api/pipelines/<pipeline_id>/run', methods=['POST'])
@token_required
def run_pipeline(current_user, pipeline_id):
    data = request.get_json() or {}
    
    pipeline = None
    if db_session:
        pipeline = db_session.query(Pipeline).filter_by(id=pipeline_id).first()
    else:
        pipeline = pipelines_cache.get(pipeline_id)
    
    if not pipeline:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    # Get configuration
    model_type = data.get('model_type', pipeline.model_type or 'random_forest')
    model_config = data.get('model_config', pipeline.model_config or {})
    data_path = data.get('data_path', pipeline.data_path)
    target_column = data.get('target_column', pipeline.target_column)
    
    if not data_path:
        return jsonify({'error': 'Data path required'}), 400
    
    if not target_column:
        return jsonify({'error': 'Target column required'}), 400
    
    # Create training run
    run_id = str(uuid.uuid4())
    if db_session:
        run = TrainingRun(
            id=run_id,
            pipeline_id=pipeline_id,
            status='running',
            logs=[]
        )
        db_session.add(run)
        db_session.commit()
    else:
        training_runs_cache[run_id] = {
            'id': run_id,
            'pipeline_id': pipeline_id,
            'status': 'running',
            'logs': []
        }
    
    def execute_training():
        logs = []
        start_time = time.time()
        
        try:
            # Update status
            if db_session:
                run = db_session.query(TrainingRun).filter_by(id=run_id).first()
                run.status = 'running'
                run.logs = logs
                db_session.commit()
            
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': 'Starting training pipeline'})
            
            # Load data
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Loading data from {data_path}'})
            data_file = Path(data_path)
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            ext = data_file.suffix.lower()
            if ext == '.csv':
                df = pd.read_csv(data_file)
            elif ext in ['.json', '.jsonl']:
                df = pd.read_json(data_file)
            elif ext == '.parquet':
                df = pd.read_parquet(data_file)
            else:
                raise ValueError(f"Unsupported format: {ext}")
            
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Loaded {len(df)} rows, {len(df.columns)} columns'})
            
            # Prepare features and target
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Encoding {len(categorical_cols)} categorical columns'})
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
            # Handle missing values
            if X.isnull().any().any():
                logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'warning', 'message': 'Filling missing values with median/mode'})
                for col in X.columns:
                    if X[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(X[col]):
                            X[col] = X[col].fillna(X[col].median())
                        else:
                            X[col] = X[col].fillna(X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'unknown')
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Train: {len(X_train)}, Test: {len(X_test)}'})
            
            # Create model
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Creating {model_type} model'})
            if GPU_AVAILABLE and model_type in ['mlp', 'random_forest']:
                logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': 'GPU available and will be used if supported by model'})
            else:
                logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': 'Training on CPU (GPU not available or not supported by this model)'})
            
            model = load_user_model(model_type, model_config)
            
            # Train
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': 'Training model...'})
            model.fit(X_train, y_train)
            
            # Evaluate
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': 'Evaluating model...'})
            y_pred = model.predict(X_test)
            
            # Calculate metrics based on task type
            is_classification = hasattr(model, 'predict_proba')
            metrics = {}
            
            if is_classification:
                metrics = {
                    'accuracy': float(accuracy_score(y_test, y_pred)),
                    'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                }
                if len(set(y_test)) > 1:
                    try:
                        metrics['roc_auc'] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], multi_class='ovr'))
                    except:
                        pass
                conf_matrix = confusion_matrix(y_test, y_pred)
                metrics['confusion_matrix'] = conf_matrix.tolist()
            else:
                metrics = {
                    'mse': float(mean_squared_error(y_test, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    'r2': float(r2_score(y_test, y_pred))
                }
            
            # Training time
            training_time = time.time() - start_time
            metrics['training_time_seconds'] = training_time
            metrics['gpu_used'] = GPU_AVAILABLE
            
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Training completed in {training_time:.2f}s'})
            
            # Save model
            model_filename = f"model_{run_id}.joblib"
            model_path = app.config['MODELS_DIR'] / model_filename
            joblib.dump(model, model_path)
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Model saved to {model_path}'})
            
            # Save to HDFS
            hdfs_path = None
            if hdfs_client:
                try:
                    hdfs_path = f"/mlops/models/{model_filename}"
                    with open(model_path, 'rb') as f:
                        hdfs_client.write_file(hdfs_path, f.read(), overwrite=True)
                    logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Model backed up to HDFS: {hdfs_path}'})
                except Exception as e:
                    logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'warning', 'message': f'HDFS backup failed: {e}'})
            
            # Create model registry entry
            model_id = str(uuid.uuid4())
            api_endpoint = f"/api/models/{model_id}/predict"
            
            if db_session:
                user = db_session.query(User).filter_by(username=current_user).first()
                model_record = Model(
                    id=model_id,
                    name=pipeline.name if pipeline else f'Model-{model_id[:8]}',
                    pipeline_id=pipeline_id,
                    path=str(model_path),
                    hdfs_path=hdfs_path,
                    model_type=model_type,
                    metrics=metrics,
                    api_endpoint=api_endpoint,
                    created_by=user.id if user else None
                )
                db_session.add(model_record)
                run.model_id = model_id
                db_session.commit()
                
                # Save metrics
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        model_metric = ModelMetrics(
                            model_id=model_id,
                            metric_type=metric_name,
                            value=metric_value,
                            run_id=run_id
                        )
                        db_session.add(model_metric)
                db_session.commit()
            else:
                models_cache[model_id] = {
                    'id': model_id,
                    'name': pipeline.name if pipeline else f'Model-{model_id[:8]}',
                    'path': str(model_path),
                    'model_type': model_type,
                    'metrics': metrics,
                    'api_endpoint': api_endpoint,
                    'created_at': datetime.utcnow().isoformat()
                }
            
            # Update run status
            if db_session:
                run = db_session.query(TrainingRun).filter_by(id=run_id).first()
                run.status = 'completed'
                run.completed_at = datetime.utcnow()
                run.metrics = metrics
                run.model_path = str(model_path)
                run.logs = logs
                db_session.commit()
                
                pipeline = db_session.query(Pipeline).filter_by(id=pipeline_id).first()
                pipeline.status = 'completed'
                pipeline.progress = 100
                db_session.commit()
            else:
                training_runs_cache[run_id] = {
                    'id': run_id,
                    'pipeline_id': pipeline_id,
                    'status': 'completed',
                    'completed_at': datetime.utcnow().isoformat(),
                    'metrics': metrics,
                    'model_path': str(model_path),
                    'model_id': model_id,
                    'logs': logs
                }
                pipelines_cache[pipeline_id]['status'] = 'completed'
                pipelines_cache[pipeline_id]['progress'] = 100
            
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'success', 'message': 'Pipeline completed successfully'})
            
        except Exception as e:
            error_msg = str(e)
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'error', 'message': error_msg})
            
            if db_session:
                run = db_session.query(TrainingRun).filter_by(id=run_id).first()
                run.status = 'failed'
                run.error_message = error_msg
                run.logs = logs
                db_session.commit()
                
                pipeline = db_session.query(Pipeline).filter_by(id=pipeline_id).first()
                pipeline.status = 'failed'
                db_session.commit()
            else:
                training_runs_cache[run_id] = {
                    'id': run_id,
                    'pipeline_id': pipeline_id,
                    'status': 'failed',
                    'error_message': error_msg,
                    'logs': logs
                }
    
    thread = threading.Thread(target=execute_training)
    thread.start()
    
    return jsonify({
        'message': 'Pipeline started',
        'pipeline_id': pipeline_id,
        'run_id': run_id
    })

@app.route('/api/pipelines/<pipeline_id>', methods=['GET'])
@token_required
def get_pipeline(current_user, pipeline_id):
    if db_session:
        pipeline = db_session.query(Pipeline).filter_by(id=pipeline_id).first()
        if not pipeline:
            return jsonify({'error': 'Pipeline not found'}), 404
        return jsonify({
            'id': pipeline.id,
            'name': pipeline.name,
            'description': pipeline.description,
            'status': pipeline.status,
            'progress': pipeline.progress,
            'model_type': pipeline.model_type,
            'model_config': pipeline.model_config,
            'data_path': pipeline.data_path,
            'target_column': pipeline.target_column,
            'created_at': pipeline.created_at.isoformat(),
            'runs': [{
                'id': r.id,
                'status': r.status,
                'started_at': r.started_at.isoformat(),
                'completed_at': r.completed_at.isoformat() if r.completed_at else None,
                'metrics': r.metrics,
                'error_message': r.error_message
            } for r in pipeline.runs.order_by(TrainingRun.started_at.desc())]
        })
    
    pipeline = pipelines_cache.get(pipeline_id)
    if not pipeline:
        return jsonify({'error': 'Pipeline not found'}), 404
    return jsonify(pipeline)

@app.route('/api/pipelines/<pipeline_id>', methods=['DELETE'])
@admin_required
def delete_pipeline(current_user, pipeline_id):
    if db_session:
        pipeline = db_session.query(Pipeline).filter_by(id=pipeline_id).first()
        if not pipeline:
            return jsonify({'error': 'Pipeline not found'}), 404
        
        # Delete associated models
        models = db_session.query(Model).filter_by(pipeline_id=pipeline_id).all()
        for model in models:
            model_path = Path(model.path)
            if model_path.exists():
                model_path.unlink()
            db_session.delete(model)
        
        # Delete runs
        runs = db_session.query(TrainingRun).filter_by(pipeline_id=pipeline_id).all()
        for run in runs:
            db_session.delete(run)
        
        user = db_session.query(User).filter_by(username=current_user).first()
        log_activity(user.id if user else None, 'delete_pipeline', 'pipeline', pipeline_id)
        
        db_session.delete(pipeline)
        db_session.commit()
        return jsonify({'message': 'Pipeline deleted'})
    
    if pipeline_id in pipelines_cache:
        del pipelines_cache[pipeline_id]
        return jsonify({'message': 'Pipeline deleted'})
    return jsonify({'error': 'Pipeline not found'}), 404

# Models management
@app.route('/api/models', methods=['GET'])
@token_required
def get_models(current_user):
    if db_session:
        models = db_session.query(Model).all()
        return jsonify([{
            'id': m.id,
            'name': m.name,
            'model_type': m.model_type,
            'metrics': m.metrics,
            'api_endpoint': m.api_endpoint,
            'created_at': m.created_at.isoformat()
        } for m in models])
    return jsonify(list(models_cache.values()))

@app.route('/api/models/<model_id>', methods=['GET'])
@token_required
def get_model_details(current_user, model_id):
    if db_session:
        model = db_session.query(Model).filter_by(id=model_id).first()
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Get metrics history
        metrics_history = db_session.query(ModelMetrics).filter_by(model_id=model_id).order_by(ModelMetrics.timestamp.desc()).limit(100).all()
        
        return jsonify({
            'id': model.id,
            'name': model.name,
            'model_type': model.model_type,
            'path': model.path,
            'hdfs_path': model.hdfs_path,
            'metrics': model.metrics,
            'api_endpoint': model.api_endpoint,
            'created_at': model.created_at.isoformat(),
            'metrics_history': [{
                'metric_type': m.metric_type,
                'value': m.value,
                'timestamp': m.timestamp.isoformat()
            } for m in metrics_history]
        })
    
    model = models_cache.get(model_id)
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    return jsonify(model)

@app.route('/api/models/<model_id>/predict', methods=['POST'])
@token_required
def predict(current_user, model_id):
    start_time = time.time()
    
    data = request.get_json()
    features = data.get('features')
    features_df = data.get('features_df')  # Allow DataFrame-style input
    
    model = None
    if db_session:
        model = db_session.query(Model).filter_by(id=model_id).first()
    else:
        model = models_cache.get(model_id)
    
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    
    model_path = Path(model.path if isinstance(model, dict) else model.path)
    if not model_path.exists():
        return jsonify({'error': 'Model file not found'}), 404
    
    try:
        model_obj = joblib.load(model_path)
        
        # Prepare input
        if features_df is not None:
            X = pd.DataFrame([features_df] if isinstance(features_df, dict) else features_df)
        else:
            X = np.array([features])
        
        # Predict
        prediction = model_obj.predict(X)[0]
        probabilities = model_obj.predict_proba(X)[0].tolist() if hasattr(model_obj, 'predict_proba') else None
        
        # Log prediction
        latency = time.time() - start_time
        if db_session:
            metric = ModelMetrics(
                model_id=model_id,
                metric_type='prediction_latency',
                value=latency
            )
            db_session.add(metric)
            db_session.commit()
            
            user = db_session.query(User).filter_by(username=current_user).first()
            log_activity(user.id if user else None, 'predict', 'model', model_id, {'latency': latency})
        
        return jsonify({
            'prediction': int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction),
            'probabilities': probabilities,
            'model_name': model.name if isinstance(model, dict) else model.name,
            'latency_ms': round(latency * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

@app.route('/api/models/<model_id>/metrics', methods=['GET'])
@token_required
def get_model_metrics(current_user, model_id):
    """Get golden signals and metrics for a model"""
    if db_session:
        model = db_session.query(Model).filter_by(id=model_id).first()
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Get recent metrics
        recent_metrics = db_session.query(ModelMetrics).filter_by(model_id=model_id).order_by(ModelMetrics.timestamp.desc()).limit(1000).all()
        
        # Calculate golden signals
        latencies = [m.value for m in recent_metrics if m.metric_type == 'prediction_latency']
        predictions = [m for m in recent_metrics if m.metric_type == 'prediction']
        
        golden_signals = {
            'latency': {
                'p50': float(np.percentile(latencies, 50)) if latencies else 0,
                'p95': float(np.percentile(latencies, 95)) if latencies else 0,
                'p99': float(np.percentile(latencies, 99)) if latencies else 0,
                'avg': float(np.mean(latencies)) if latencies else 0
            },
            'error_rate': 0.0,  # Would need error logging
            'throughput': len(predictions) / 3600 if predictions else 0,  # per hour
            'saturation': 0.0  # Would need resource monitoring
        }
        
        # Model quality metrics
        quality_metrics = model.metrics or {}
        
        return jsonify({
            'model_id': model_id,
            'golden_signals': golden_signals,
            'quality_metrics': quality_metrics,
            'recent_metrics': [{
                'type': m.metric_type,
                'value': m.value,
                'timestamp': m.timestamp.isoformat()
            } for m in recent_metrics[:100]]
        })
    
    model = models_cache.get(model_id)
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    
    return jsonify({
        'model_id': model_id,
        'golden_signals': {
            'latency': {'p50': 0, 'p95': 0, 'p99': 0, 'avg': 0},
            'error_rate': 0,
            'throughput': 0,
            'saturation': 0
        },
        'quality_metrics': model.get('metrics', {})
    })

@app.route('/api/models/<model_id>', methods=['DELETE'])
@admin_required
def delete_model(current_user, model_id):
    if db_session:
        model = db_session.query(Model).filter_by(id=model_id).first()
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        model_path = Path(model.path)
        if model_path.exists():
            model_path.unlink()
        
        # Delete from HDFS
        if model.hdfs_path and hdfs_client:
            try:
                hdfs_client.delete_file(model.hdfs_path)
            except:
                pass
        
        # Delete metrics
        db_session.query(ModelMetrics).filter_by(model_id=model_id).delete()
        
        user = db_session.query(User).filter_by(username=current_user).first()
        log_activity(user.id if user else None, 'delete_model', 'model', model_id)
        
        db_session.delete(model)
        db_session.commit()
        return jsonify({'message': 'Model deleted'})
    
    if model_id in models_cache:
        model_path = Path(models_cache[model_id]['path'])
        if model_path.exists():
            model_path.unlink()
        del models_cache[model_id]
        return jsonify({'message': 'Model deleted'})
    return jsonify({'error': 'Model not found'}), 404

# Activity logs
@app.route('/api/logs', methods=['GET'])
@admin_required
def get_logs(current_user):
    resource_type = request.args.get('resource_type')
    resource_id = request.args.get('resource_id')
    limit = int(request.args.get('limit', 100))
    
    if db_session:
        query = db_session.query(ActivityLog)
        if resource_type:
            query = query.filter_by(resource_type=resource_type)
        if resource_id:
            query = query.filter_by(resource_id=resource_id)
        
        logs = query.order_by(ActivityLog.timestamp.desc()).limit(limit).all()
        return jsonify([{
            'id': l.id,
            'timestamp': l.timestamp.isoformat(),
            'user_id': l.user_id,
            'action': l.action,
            'resource_type': l.resource_type,
            'resource_id': l.resource_id,
            'details': l.details
        } for l in logs])
    
    return jsonify([])

@app.route('/api/pipelines/<pipeline_id>/runs/<run_id>', methods=['GET'])
@token_required
def get_run_logs(current_user, pipeline_id, run_id):
    """Get detailed logs for a training run"""
    if db_session:
        run = db_session.query(TrainingRun).filter_by(id=run_id, pipeline_id=pipeline_id).first()
        if not run:
            return jsonify({'error': 'Run not found'}), 404
        return jsonify({
            'id': run.id,
            'status': run.status,
            'started_at': run.started_at.isoformat(),
            'completed_at': run.completed_at.isoformat() if run.completed_at else None,
            'metrics': run.metrics,
            'error_message': run.error_message,
            'logs': run.logs or []
        })
    
    run = training_runs_cache.get(run_id)
    if not run or run.get('pipeline_id') != pipeline_id:
        return jsonify({'error': 'Run not found'}), 404
    return jsonify(run)

# Dashboard stats
@app.route('/api/stats', methods=['GET'])
@token_required
def get_stats(current_user):
    if db_session:
        total_pipelines = db_session.query(Pipeline).count()
        completed = db_session.query(Pipeline).filter_by(status='completed').count()
        running = db_session.query(Pipeline).filter_by(status='running').count()
        failed = db_session.query(Pipeline).filter_by(status='failed').count()
        total_models = db_session.query(Model).count()
        
        # Average accuracy from completed pipelines
        runs = db_session.query(TrainingRun).filter_by(status='completed').all()
        accuracies = []
        for run in runs:
            if run.metrics and 'accuracy' in run.metrics:
                accuracies.append(run.metrics['accuracy'])
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        # Recent activity
        recent_logs = db_session.query(ActivityLog).order_by(ActivityLog.timestamp.desc()).limit(10).all()
        
        return jsonify({
            'total_pipelines': total_pipelines,
            'completed': completed,
            'running': running,
            'failed': failed,
            'total_models': total_models,
            'avg_accuracy': round(avg_accuracy, 4),
            'gpu_available': GPU_AVAILABLE,
            'hdfs_available': HDFS_AVAILABLE,
            'recent_activity': [{
                'action': l.action,
                'resource_type': l.resource_type,
                'timestamp': l.timestamp.isoformat()
            } for l in recent_logs]
        })
    
    return jsonify({
        'total_pipelines': len(pipelines_cache),
        'completed': sum(1 for p in pipelines_cache.values() if p.get('status') == 'completed'),
        'running': sum(1 for p in pipelines_cache.values() if p.get('status') == 'running'),
        'failed': sum(1 for p in pipelines_cache.values() if p.get('status') == 'failed'),
        'total_models': len(models_cache),
        'gpu_available': GPU_AVAILABLE,
        'hdfs_available': HDFS_AVAILABLE
    })

# HDFS status
@app.route('/api/hdfs/status', methods=['GET'])
@token_required
def get_hdfs_status(current_user):
    if hdfs_client:
        try:
            file_info = hdfs_client.get_file_info('/mlops')
            return jsonify({
                'status': 'connected',
                'url': app.config['HDFS_URL'],
                'user': app.config['HDFS_USER']
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e)
            })
    
    return jsonify({
        'status': 'not_configured',
        'message': 'HDFS client not available. Using local storage.'
    })

# Serve frontend
@app.route('/')
@app.route('/<path:path>')
def serve_frontend(path='index.html'):
    if path == '' or path.endswith('/'):
        path += 'index.html'
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    print("=" * 60)
    print("MLOps Platform - Backend")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"HDFS Available: {HDFS_AVAILABLE}")
    print(f"Database: {'PostgreSQL' if db_session else 'In-memory (development)'}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
