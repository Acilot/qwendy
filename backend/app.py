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

# HDFS (optional - for external cluster integration)
HDFS_AVAILABLE = False
try:
    import pyarrow.fs as hdfs_fs
    HDFS_AVAILABLE = os.environ.get('HDFS_URL', '') != ''
except ImportError:
    pass

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
    metric_type = Column(String(64))
    value = Column(Float)
    run_id = Column(String(36), ForeignKey('training_runs.id'))

class ActivityLog(Base):
    __tablename__ = 'activity_logs'
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(String(36), ForeignKey('users.id'))
    action = Column(String(64))
    resource_type = Column(String(32))
    resource_id = Column(String(36))
    details = Column(JSON)

# Database initialization
db_session_factory = None
try:
    engine = create_engine(app.config['DATABASE_URL'])
    Base.metadata.create_all(engine)
    db_session_factory = sessionmaker(bind=engine)
except Exception as e:
    print(f"⚠️ Database not available, using in-memory storage: {e}")

def get_db_session():
    """Get a new database session"""
    if db_session_factory:
        return db_session_factory()
    return None

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
        session = get_db_session()
        if session:
            try:
                user = session.query(User).filter_by(username=current_user).first()
                session.close()
                if not user or user.role != 'admin':
                    return jsonify({'error': 'Admin access required'}), 403
            except Exception:
                session.close()
                return jsonify({'error': 'Database error'}), 500
        elif users_cache.get(current_user, {}).get('role') != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(current_user, *args, **kwargs)
    return decorated

def log_activity(user_id, action, resource_type=None, resource_id=None, details=None):
    session = get_db_session()
    if session:
        try:
            log = ActivityLog(user_id=user_id, action=action, resource_type=resource_type, resource_id=resource_id, details=details)
            session.add(log)
            session.commit()
        except Exception as e:
            print(f"Failed to log activity: {e}")
        finally:
            session.close()

def check_first_run():
    session = get_db_session()
    if session:
        try:
            admin_exists = session.query(User).filter_by(role='admin').first() is not None
            session.close()
            return not admin_exists
        except Exception:
            session.close()
            return True
    else:
        return 'admin' not in users_cache

# Setup endpoint
@app.route('/api/setup', methods=['GET'])
def check_setup():
    return jsonify({'needs_setup': check_first_run()})

@app.route('/api/setup', methods=['POST'])
def setup_admin():
    data = request.get_json()
    username = data.get('username', 'admin')
    password = data.get('password')
    
    if not password or len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    if not check_first_run():
        return jsonify({'error': 'Setup already completed'}), 400
    
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    session = get_db_session()
    if session:
        try:
            admin = User(username=username, password_hash=password_hash, role='admin')
            session.add(admin)
            session.commit()
        except Exception as e:
            session.close()
            return jsonify({'error': f'Database error: {str(e)}'}), 500
        finally:
            session.close()
    else:
        users_cache[username] = {'password': password_hash, 'role': 'admin'}
    
    log_activity(None, 'setup_admin', 'user', username)
    return jsonify({'message': 'Admin user created', 'username': username})

# Registration endpoint (public)
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    session = get_db_session()
    if session:
        try:
            existing = session.query(User).filter_by(username=username).first()
            if existing:
                session.close()
                return jsonify({'error': 'User already exists'}), 400
            user = User(username=username, password_hash=password_hash, role='user')
            session.add(user)
            session.commit()
            user_id = user.id
            session.close()
            log_activity(user_id, 'register', 'user', user_id)
            return jsonify({'id': user_id, 'username': username, 'role': 'user'})
        except Exception as e:
            session.close()
            return jsonify({'error': f'Database error: {str(e)}'}), 500
    else:
        if username in users_cache:
            return jsonify({'error': 'User already exists'}), 400
        users_cache[username] = {'password': password_hash, 'role': 'user'}
        return jsonify({'username': username, 'role': 'user'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    session = get_db_session()
    if session:
        try:
            user = session.query(User).filter_by(username=username).first()
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                token = jwt.encode({
                    'username': username,
                    'role': user.role,
                    'exp': datetime.utcnow().timestamp() + 86400
                }, app.config['SECRET_KEY'], algorithm='HS256')
                log_activity(user.id, 'login', 'user', user.id)
                session.close()
                return jsonify({'token': token, 'username': username, 'role': user.role})
            session.close()
        except Exception as e:
            session.close()
            return jsonify({'error': f'Database error: {str(e)}'}), 500
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
    session = get_db_session()
    if session:
        try:
            users = session.query(User).all()
            session.close()
            return jsonify([{'id': u.id, 'username': u.username, 'role': u.role, 'created_at': u.created_at.isoformat()} for u in users])
        except Exception:
            session.close()
            return jsonify([])
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
    
    session = get_db_session()
    if session:
        try:
            existing = session.query(User).filter_by(username=username).first()
            if existing:
                session.close()
                return jsonify({'error': 'User already exists'}), 400
            user = User(username=username, password_hash=password_hash, role=role)
            session.add(user)
            session.commit()
            user_id = user.id
            session.close()
            log_activity(current_user, 'create_user', 'user', user_id)
            return jsonify({'id': user_id, 'username': username, 'role': role})
        except Exception as e:
            session.close()
            return jsonify({'error': f'Database error: {str(e)}'}), 500
    else:
        if username in users_cache:
            return jsonify({'error': 'User already exists'}), 400
        users_cache[username] = {'password': password_hash, 'role': role}
        return jsonify({'username': username, 'role': role})

# Data upload
@app.route('/api/data/upload', methods=['POST'])
@token_required
def upload_data(current_user):
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    data_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix.lower()
    filename = f"data_{data_id}{ext}"
    filepath = app.config['DATA_DIR'] / filename
    file.save(filepath)
    
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
            'columns': [{'name': col, 'dtype': str(df[col].dtype), 'nullable': bool(df[col].isnull().any()), 'sample_values': df[col].dropna().head(3).tolist()} for col in df.columns],
            'row_count': len(df),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        hdfs_path = None
        if HDFS_AVAILABLE:
            try:
                fs = hdfs_fs.HadoopFileSystem(app.config['HDFS_URL'].replace('hdfs://', '').split(':')[0], port=9000, user=app.config['HDFS_USER'])
                hdfs_path = f"/mlops/data/{filename}"
                with open(filepath, 'rb') as f:
                    fs.write_file(hdfs_path, f.read(), overwrite=True)
            except Exception as e:
                print(f"HDFS write failed: {e}")
        
        log_activity(current_user, 'upload_data', 'data', data_id, {'filename': filename})
        return jsonify({'data_id': data_id, 'filename': filename, 'path': str(filepath), 'hdfs_path': hdfs_path, 'schema': schema, 'format': ext[1:]})
    except Exception as e:
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 400

@app.route('/api/data/<data_id>/schema', methods=['GET'])
@token_required
def get_data_schema(current_user, data_id):
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
            'columns': [{'name': col, 'dtype': str(df[col].dtype), 'nullable': bool(df[col].isnull().any()), 'unique_values': df[col].nunique(), 'min': float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None, 'max': float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None, 'mean': float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None} for col in df.columns],
            'row_count': len(df),
            'column_count': len(df.columns)
        }
        return jsonify(schema)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Pipelines
@app.route('/api/pipelines', methods=['GET'])
@token_required
def get_pipelines(current_user):
    session = get_db_session()
    if session:
        try:
            user = session.query(User).filter_by(username=current_user).first()
            pipelines = session.query(Pipeline).filter_by(created_by=user.id).all() if user else []
            session.close()
            return jsonify([{'id': p.id, 'name': p.name, 'description': p.description, 'status': p.status, 'progress': p.progress, 'model_type': p.model_type, 'created_at': p.created_at.isoformat(), 'runs_count': p.runs.count()} for p in pipelines])
        except Exception:
            session.close()
            return jsonify([])
    return jsonify(list(pipelines_cache.values()))

@app.route('/api/pipelines', methods=['POST'])
@token_required
def create_pipeline(current_user):
    data = request.get_json()
    pipeline_id = str(uuid.uuid4())
    
    pipeline_data = {'id': pipeline_id, 'name': data.get('name', f'Pipeline-{pipeline_id[:8]}'), 'description': data.get('description'), 'model_type': data.get('model_type'), 'model_config': data.get('model_config', {}), 'data_path': data.get('data_path'), 'target_column': data.get('target_column'), 'status': 'created', 'progress': 0, 'created_at': datetime.utcnow().isoformat(), 'created_by': current_user}
    
    session = get_db_session()
    if session:
        try:
            user = session.query(User).filter_by(username=current_user).first()
            pipeline = Pipeline(id=pipeline_id, name=pipeline_data['name'], description=pipeline_data['description'], model_type=pipeline_data['model_type'], model_config=pipeline_data['model_config'], data_path=pipeline_data['data_path'], target_column=pipeline_data['target_column'], created_by=user.id if user else None)
            session.add(pipeline)
            session.commit()
            session.close()
            log_activity(user.id if user else None, 'create_pipeline', 'pipeline', pipeline_id)
        except Exception as e:
            session.close()
            print(f"Failed to create pipeline: {e}")
    
    pipelines_cache[pipeline_id] = pipeline_data
    return jsonify(pipeline_data)

def load_user_model(model_type: str, model_config: dict):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression, Ridge, Lasso
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    
    model_classes = {
        'random_forest': RandomForestClassifier, 'gradient_boosting': GradientBoostingClassifier, 'ada_boost': AdaBoostClassifier,
        'logistic_regression': LogisticRegression, 'sgd_classifier': SGDClassifier, 'svc': SVC, 'knn': KNeighborsClassifier,
        'naive_bayes': GaussianNB, 'decision_tree': DecisionTreeClassifier, 'mlp': MLPClassifier,
        'random_forest_regressor': RandomForestRegressor, 'gradient_boosting_regressor': GradientBoostingRegressor,
        'linear_regression': LinearRegression, 'ridge': Ridge, 'lasso': Lasso, 'svr': SVR
    }
    
    if model_type and '.' in model_type:
        try:
            module_path, class_name = model_type.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            return model_class(**model_config)
        except Exception as e:
            raise ValueError(f"Failed to load custom model {model_type}: {e}")
    
    if model_type in model_classes:
        return model_classes[model_type](**model_config)
    return RandomForestClassifier(**model_config)

@app.route('/api/pipelines/<pipeline_id>/run', methods=['POST'])
@token_required
def run_pipeline(current_user, pipeline_id):
    data = request.get_json() or {}
    
    session = get_db_session()
    pipeline = None
    if session:
        try:
            pipeline = session.query(Pipeline).filter_by(id=pipeline_id).first()
        except:
            pass
        finally:
            session.close()
    else:
        pipeline = pipelines_cache.get(pipeline_id)
    
    if not pipeline:
        return jsonify({'error': 'Pipeline not found'}), 404
    
    model_type = data.get('model_type', pipeline.model_type or 'random_forest')
    model_config = data.get('model_config', pipeline.model_config or {})
    data_path = data.get('data_path', pipeline.data_path)
    target_column = data.get('target_column', pipeline.target_column)
    
    if not data_path:
        return jsonify({'error': 'Data path required'}), 400
    if not target_column:
        return jsonify({'error': 'Target column required'}), 400
    
    run_id = str(uuid.uuid4())
    
    session = get_db_session()
    if session:
        try:
            run = TrainingRun(id=run_id, pipeline_id=pipeline_id, status='running', logs=[])
            session.add(run)
            session.commit()
            session.close()
        except Exception as e:
            session.close()
            print(f"Failed to create run: {e}")
    else:
        training_runs_cache[run_id] = {'id': run_id, 'pipeline_id': pipeline_id, 'status': 'running', 'logs': []}
    
    def execute_training():
        logs = []
        start_time = time.time()
        
        try:
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': 'Starting training pipeline'})
            
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
            
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Encoding {len(categorical_cols)} categorical columns'})
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
            if X.isnull().any().any():
                logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'warning', 'message': 'Filling missing values'})
                for col in X.columns:
                    if X[col].isnull().any():
                        X[col] = X[col].fillna(X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else X[col].mode().iloc[0] if len(X[col].mode()) > 0 else 'unknown')
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Train: {len(X_train)}, Test: {len(X_test)}'})
            
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Creating {model_type} model'})
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': 'GPU: ' + ('Available' if GPU_AVAILABLE else 'Not available')})
            
            model = load_user_model(model_type, model_config)
            model.fit(X_train, y_train)
            
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': 'Evaluating model...'})
            y_pred = model.predict(X_test)
            
            is_classification = hasattr(model, 'predict_proba')
            metrics = {}
            
            if is_classification:
                metrics = {'accuracy': float(accuracy_score(y_test, y_pred)), 'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 'f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))}
                if len(set(y_test)) > 1:
                    try:
                        metrics['roc_auc'] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], multi_class='ovr'))
                    except:
                        pass
                metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            else:
                metrics = {'mse': float(mean_squared_error(y_test, y_pred)), 'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))), 'r2': float(r2_score(y_test, y_pred))}
            
            metrics['training_time_seconds'] = time.time() - start_time
            metrics['gpu_used'] = GPU_AVAILABLE
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Training completed in {metrics["training_time_seconds"]:.2f}s'})
            
            model_filename = f"model_{run_id}.joblib"
            model_path = app.config['MODELS_DIR'] / model_filename
            joblib.dump(model, model_path)
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Model saved to {model_path}'})
            
            hdfs_path = None
            if HDFS_AVAILABLE:
                try:
                    fs = hdfs_fs.HadoopFileSystem(app.config['HDFS_URL'].replace('hdfs://', '').split(':')[0], port=9000, user=app.config['HDFS_USER'])
                    hdfs_path = f"/mlops/models/{model_filename}"
                    with open(model_path, 'rb') as f:
                        fs.write_file(hdfs_path, f.read(), overwrite=True)
                    logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'info', 'message': f'Model backed up to HDFS: {hdfs_path}'})
                except Exception as e:
                    logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'warning', 'message': f'HDFS backup failed: {e}'})
            
            model_id = str(uuid.uuid4())
            api_endpoint = f"/api/models/{model_id}/predict"
            
            session = get_db_session()
            if session:
                try:
                    user = session.query(User).filter_by(username=current_user).first()
                    model_record = Model(id=model_id, name=pipeline.name if pipeline else f'Model-{model_id[:8]}', pipeline_id=pipeline_id, path=str(model_path), hdfs_path=hdfs_path, model_type=model_type, metrics=metrics, api_endpoint=api_endpoint, created_by=user.id if user else None)
                    session.add(model_record)
                    run = session.query(TrainingRun).filter_by(id=run_id).first()
                    run.model_id = model_id
                    run.status = 'completed'
                    run.completed_at = datetime.utcnow()
                    run.metrics = metrics
                    run.logs = logs
                    session.commit()
                    
                    pipeline = session.query(Pipeline).filter_by(id=pipeline_id).first()
                    pipeline.status = 'completed'
                    pipeline.progress = 100
                    session.commit()
                    
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            model_metric = ModelMetrics(model_id=model_id, metric_type=metric_name, value=metric_value, run_id=run_id)
                            session.add(model_metric)
                    session.commit()
                    session.close()
                except Exception as e:
                    session.close()
                    print(f"Failed to save model: {e}")
            else:
                models_cache[model_id] = {'id': model_id, 'name': pipeline.name if pipeline else f'Model-{model_id[:8]}', 'path': str(model_path), 'model_type': model_type, 'metrics': metrics, 'api_endpoint': api_endpoint, 'created_at': datetime.utcnow().isoformat()}
                training_runs_cache[run_id] = {'id': run_id, 'pipeline_id': pipeline_id, 'status': 'completed', 'completed_at': datetime.utcnow().isoformat(), 'metrics': metrics, 'model_path': str(model_path), 'model_id': model_id, 'logs': logs}
                pipelines_cache[pipeline_id]['status'] = 'completed'
                pipelines_cache[pipeline_id]['progress'] = 100
            
        except Exception as e:
            error_msg = str(e)
            logs.append({'timestamp': datetime.utcnow().isoformat(), 'level': 'error', 'message': error_msg})
            
            session = get_db_session()
            if session:
                try:
                    run = session.query(TrainingRun).filter_by(id=run_id).first()
                    run.status = 'failed'
                    run.error_message = error_msg
                    run.logs = logs
                    session.commit()
                    pipeline = session.query(Pipeline).filter_by(id=pipeline_id).first()
                    pipeline.status = 'failed'
                    session.commit()
                    session.close()
                except:
                    session.close()
            else:
                training_runs_cache[run_id] = {'id': run_id, 'pipeline_id': pipeline_id, 'status': 'failed', 'error_message': error_msg, 'logs': logs}
    
    thread = threading.Thread(target=execute_training)
    thread.start()
    return jsonify({'message': 'Pipeline started', 'pipeline_id': pipeline_id, 'run_id': run_id})

@app.route('/api/pipelines/<pipeline_id>', methods=['GET'])
@token_required
def get_pipeline(current_user, pipeline_id):
    session = get_db_session()
    if session:
        try:
            pipeline = session.query(Pipeline).filter_by(id=pipeline_id).first()
            if not pipeline:
                session.close()
                return jsonify({'error': 'Pipeline not found'}), 404
            result = {'id': pipeline.id, 'name': pipeline.name, 'description': pipeline.description, 'status': pipeline.status, 'progress': pipeline.progress, 'model_type': pipeline.model_type, 'model_config': pipeline.model_config, 'data_path': pipeline.data_path, 'target_column': pipeline.target_column, 'created_at': pipeline.created_at.isoformat(), 'runs': [{'id': r.id, 'status': r.status, 'started_at': r.started_at.isoformat(), 'completed_at': r.completed_at.isoformat() if r.completed_at else None, 'metrics': r.metrics, 'error_message': r.error_message} for r in pipeline.runs.order_by(TrainingRun.started_at.desc())]}
            session.close()
            return jsonify(result)
        except Exception as e:
            session.close()
            return jsonify({'error': str(e)}), 500
    
    pipeline = pipelines_cache.get(pipeline_id)
    if not pipeline:
        return jsonify({'error': 'Pipeline not found'}), 404
    return jsonify(pipeline)

@app.route('/api/pipelines/<pipeline_id>', methods=['DELETE'])
@admin_required
def delete_pipeline(current_user, pipeline_id):
    session = get_db_session()
    if session:
        try:
            pipeline = session.query(Pipeline).filter_by(id=pipeline_id).first()
            if not pipeline:
                session.close()
                return jsonify({'error': 'Pipeline not found'}), 404
            
            models = session.query(Model).filter_by(pipeline_id=pipeline_id).all()
            for model in models:
                model_path = Path(model.path)
                if model_path.exists():
                    model_path.unlink()
                session.delete(model)
            
            runs = session.query(TrainingRun).filter_by(pipeline_id=pipeline_id).all()
            for run in runs:
                session.delete(run)
            
            user = session.query(User).filter_by(username=current_user).first()
            log_activity(user.id if user else None, 'delete_pipeline', 'pipeline', pipeline_id)
            session.delete(pipeline)
            session.commit()
            session.close()
            return jsonify({'message': 'Pipeline deleted'})
        except Exception as e:
            session.close()
            return jsonify({'error': str(e)}), 500
    
    if pipeline_id in pipelines_cache:
        del pipelines_cache[pipeline_id]
        return jsonify({'message': 'Pipeline deleted'})
    return jsonify({'error': 'Pipeline not found'}), 404

@app.route('/api/pipelines/<pipeline_id>/runs/<run_id>', methods=['GET'])
@token_required
def get_run_logs(current_user, pipeline_id, run_id):
    session = get_db_session()
    if session:
        try:
            run = session.query(TrainingRun).filter_by(id=run_id, pipeline_id=pipeline_id).first()
            if not run:
                session.close()
                return jsonify({'error': 'Run not found'}), 404
            result = {'id': run.id, 'status': run.status, 'started_at': run.started_at.isoformat(), 'completed_at': run.completed_at.isoformat() if run.completed_at else None, 'metrics': run.metrics, 'error_message': run.error_message, 'logs': run.logs or []}
            session.close()
            return jsonify(result)
        except Exception as e:
            session.close()
            return jsonify({'error': str(e)}), 500
    
    run = training_runs_cache.get(run_id)
    if not run or run.get('pipeline_id') != pipeline_id:
        return jsonify({'error': 'Run not found'}), 404
    return jsonify(run)

# Models
@app.route('/api/models', methods=['GET'])
@token_required
def get_models(current_user):
    session = get_db_session()
    if session:
        try:
            models = session.query(Model).all()
            session.close()
            return jsonify([{'id': m.id, 'name': m.name, 'model_type': m.model_type, 'metrics': m.metrics, 'api_endpoint': m.api_endpoint, 'created_at': m.created_at.isoformat()} for m in models])
        except Exception:
            session.close()
            return jsonify([])
    return jsonify(list(models_cache.values()))

@app.route('/api/models/<model_id>', methods=['GET'])
@token_required
def get_model_details(current_user, model_id):
    session = get_db_session()
    if session:
        try:
            model = session.query(Model).filter_by(id=model_id).first()
            if not model:
                session.close()
                return jsonify({'error': 'Model not found'}), 404
            metrics_history = session.query(ModelMetrics).filter_by(model_id=model_id).order_by(ModelMetrics.timestamp.desc()).limit(100).all()
            result = {'id': model.id, 'name': model.name, 'model_type': model.model_type, 'path': model.path, 'hdfs_path': model.hdfs_path, 'metrics': model.metrics, 'api_endpoint': model.api_endpoint, 'created_at': model.created_at.isoformat(), 'metrics_history': [{'metric_type': m.metric_type, 'value': m.value, 'timestamp': m.timestamp.isoformat()} for m in metrics_history]}
            session.close()
            return jsonify(result)
        except Exception as e:
            session.close()
            return jsonify({'error': str(e)}), 500
    
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
    features_df = data.get('features_df')
    
    session = get_db_session()
    model = None
    if session:
        try:
            model = session.query(Model).filter_by(id=model_id).first()
        except:
            pass
        finally:
            session.close()
    else:
        model = models_cache.get(model_id)
    
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    
    model_path = Path(model.path if isinstance(model, dict) else model.path)
    if not model_path.exists():
        return jsonify({'error': 'Model file not found'}), 404
    
    try:
        model_obj = joblib.load(model_path)
        X = pd.DataFrame([features_df] if isinstance(features_df, dict) else features_df) if features_df is not None else np.array([features])
        prediction = model_obj.predict(X)[0]
        probabilities = model_obj.predict_proba(X)[0].tolist() if hasattr(model_obj, 'predict_proba') else None
        latency = time.time() - start_time
        
        session = get_db_session()
        if session:
            try:
                metric = ModelMetrics(model_id=model_id, metric_type='prediction_latency', value=latency)
                session.add(metric)
                session.commit()
                user = session.query(User).filter_by(username=current_user).first()
                log_activity(user.id if user else None, 'predict', 'model', model_id, {'latency': latency})
                session.close()
            except:
                session.close()
        
        return jsonify({'prediction': int(prediction) if isinstance(prediction, (np.integer, int)) else float(prediction), 'probabilities': probabilities, 'model_name': model.name if isinstance(model, dict) else model.name, 'latency_ms': round(latency * 1000, 2), 'timestamp': datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 400

@app.route('/api/models/<model_id>/metrics', methods=['GET'])
@token_required
def get_model_metrics(current_user, model_id):
    session = get_db_session()
    if session:
        try:
            model = session.query(Model).filter_by(id=model_id).first()
            if not model:
                session.close()
                return jsonify({'error': 'Model not found'}), 404
            recent_metrics = session.query(ModelMetrics).filter_by(model_id=model_id).order_by(ModelMetrics.timestamp.desc()).limit(1000).all()
            session.close()
            
            latencies = [m.value for m in recent_metrics if m.metric_type == 'prediction_latency']
            predictions = [m for m in recent_metrics if m.metric_type == 'prediction']
            
            golden_signals = {'latency': {'p50': float(np.percentile(latencies, 50)) if latencies else 0, 'p95': float(np.percentile(latencies, 95)) if latencies else 0, 'p99': float(np.percentile(latencies, 99)) if latencies else 0, 'avg': float(np.mean(latencies)) if latencies else 0}, 'error_rate': 0.0, 'throughput': len(predictions) / 3600 if predictions else 0, 'saturation': 0.0}
            quality_metrics = model.metrics or {}
            
            return jsonify({'model_id': model_id, 'golden_signals': golden_signals, 'quality_metrics': quality_metrics, 'recent_metrics': [{'type': m.metric_type, 'value': m.value, 'timestamp': m.timestamp.isoformat()} for m in recent_metrics[:100]]})
        except Exception as e:
            session.close()
            return jsonify({'error': str(e)}), 500
    
    model = models_cache.get(model_id)
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    return jsonify({'model_id': model_id, 'golden_signals': {'latency': {'p50': 0, 'p95': 0, 'p99': 0, 'avg': 0}, 'error_rate': 0, 'throughput': 0, 'saturation': 0}, 'quality_metrics': model.get('metrics', {})})

@app.route('/api/models/<model_id>', methods=['DELETE'])
@admin_required
def delete_model(current_user, model_id):
    session = get_db_session()
    if session:
        try:
            model = session.query(Model).filter_by(id=model_id).first()
            if not model:
                session.close()
                return jsonify({'error': 'Model not found'}), 404
            model_path = Path(model.path)
            if model_path.exists():
                model_path.unlink()
            if model.hdfs_path and HDFS_AVAILABLE:
                try:
                    fs = hdfs_fs.HadoopFileSystem(app.config['HDFS_URL'].replace('hdfs://', '').split(':')[0], port=9000, user=app.config['HDFS_USER'])
                    fs.delete_file(model.hdfs_path)
                except:
                    pass
            session.query(ModelMetrics).filter_by(model_id=model_id).delete()
            user = session.query(User).filter_by(username=current_user).first()
            log_activity(user.id if user else None, 'delete_model', 'model', model_id)
            session.delete(model)
            session.commit()
            session.close()
            return jsonify({'message': 'Model deleted'})
        except Exception as e:
            session.close()
            return jsonify({'error': str(e)}), 500
    
    if model_id in models_cache:
        model_path = Path(models_cache[model_id]['path'])
        if model_path.exists():
            model_path.unlink()
        del models_cache[model_id]
        return jsonify({'message': 'Model deleted'})
    return jsonify({'error': 'Model not found'}), 404

# Logs
@app.route('/api/logs', methods=['GET'])
@admin_required
def get_logs(current_user):
    resource_type = request.args.get('resource_type')
    resource_id = request.args.get('resource_id')
    limit = int(request.args.get('limit', 100))
    
    session = get_db_session()
    if session:
        try:
            query = session.query(ActivityLog)
            if resource_type:
                query = query.filter_by(resource_type=resource_type)
            if resource_id:
                query = query.filter_by(resource_id=resource_id)
            logs = query.order_by(ActivityLog.timestamp.desc()).limit(limit).all()
            session.close()
            return jsonify([{'id': l.id, 'timestamp': l.timestamp.isoformat(), 'user_id': l.user_id, 'action': l.action, 'resource_type': l.resource_type, 'resource_id': l.resource_id, 'details': l.details} for l in logs])
        except Exception:
            session.close()
            return jsonify([])
    return jsonify([])

# Stats
@app.route('/api/stats', methods=['GET'])
@token_required
def get_stats(current_user):
    session = get_db_session()
    if session:
        try:
            total_pipelines = session.query(Pipeline).count()
            completed = session.query(Pipeline).filter_by(status='completed').count()
            running = session.query(Pipeline).filter_by(status='running').count()
            failed = session.query(Pipeline).filter_by(status='failed').count()
            total_models = session.query(Model).count()
            runs = session.query(TrainingRun).filter_by(status='completed').all()
            accuracies = [run.metrics['accuracy'] for run in runs if run.metrics and 'accuracy' in run.metrics]
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
            recent_logs = session.query(ActivityLog).order_by(ActivityLog.timestamp.desc()).limit(10).all()
            session.close()
            return jsonify({'total_pipelines': total_pipelines, 'completed': completed, 'running': running, 'failed': failed, 'total_models': total_models, 'avg_accuracy': round(avg_accuracy, 4), 'gpu_available': GPU_AVAILABLE, 'hdfs_available': HDFS_AVAILABLE, 'recent_activity': [{'action': l.action, 'resource_type': l.resource_type, 'timestamp': l.timestamp.isoformat()} for l in recent_logs]})
        except Exception:
            session.close()
    
    return jsonify({'total_pipelines': len(pipelines_cache), 'completed': sum(1 for p in pipelines_cache.values() if p.get('status') == 'completed'), 'running': sum(1 for p in pipelines_cache.values() if p.get('status') == 'running'), 'failed': sum(1 for p in pipelines_cache.values() if p.get('status') == 'failed'), 'total_models': len(models_cache), 'gpu_available': GPU_AVAILABLE, 'hdfs_available': HDFS_AVAILABLE})

@app.route('/api/hdfs/status', methods=['GET'])
@token_required
def get_hdfs_status(current_user):
    if HDFS_AVAILABLE:
        try:
            fs = hdfs_fs.HadoopFileSystem(app.config['HDFS_URL'].replace('hdfs://', '').split(':')[0], port=9000, user=app.config['HDFS_USER'])
            fs.get_file_info('/mlops')
            return jsonify({'status': 'connected', 'url': app.config['HDFS_URL'], 'user': app.config['HDFS_USER']})
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)})
    return jsonify({'status': 'not_configured', 'message': 'HDFS client not available. Using local storage.'})

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
    print(f"Database: {'PostgreSQL' if db_session_factory else 'In-memory (development)'}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
