import os
import time
import uuid
import hashlib
import threading
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import jwt
from functools import wraps
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from hdfs import InsecureClient
import json

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)
app.config['SECRET_KEY'] = 'mlops-secret-key-2024'
app.config['UPLOAD_FOLDER'] = '/workspace/models'
app.config['HDFS_STORAGE'] = '/workspace/hdfs_data'

# HDFS Configuration
HDFS_NAMENODE = os.getenv('HDFS_NAMENODE', 'http://namenode:9870')
HDFS_CLIENT = None

def get_hdfs_client():
    global HDFS_CLIENT
    try:
        if HDFS_CLIENT is None:
            HDFS_CLIENT = InsecureClient(HDFS_NAMENODE)
        HDFS_CLIENT.status('/')
        return HDFS_CLIENT
    except Exception as e:
        print(f"HDFS connection error: {e}")
        return None

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HDFS_STORAGE'], exist_ok=True)

pipelines = {}
models_metadata = {}
upload_tasks = {}

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
            return jsonify({'error': 'Token invalid'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if username == 'admin' and password == '999999':
        token = jwt.encode({
            'username': username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'], algorithm='HS256')
        return jsonify({'token': token, 'username': username, 'role': 'admin'})
    
    return jsonify({'error': 'Invalid credentials'}), 401

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
        'status': 'created',
        'progress': 0,
        'current_stage': 'initialized',
        'stages': [
            {'name': 'Data Loading', 'status': 'pending', 'logs': [], 'timestamp': None},
            {'name': 'Data Preprocessing', 'status': 'pending', 'logs': [], 'timestamp': None},
            {'name': 'Feature Engineering', 'status': 'pending', 'logs': [], 'timestamp': None},
            {'name': 'Model Training', 'status': 'pending', 'logs': [], 'timestamp': None},
            {'name': 'Model Evaluation', 'status': 'pending', 'logs': [], 'timestamp': None},
            {'name': 'Save to Local', 'status': 'pending', 'logs': [], 'timestamp': None},
            {'name': 'Upload to HDFS', 'status': 'pending', 'logs': [], 'timestamp': None}
        ],
        'metrics': {},
        'model_path': None,
        'hdfs_path': None,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    pipelines[pipeline_id] = pipeline
    thread = threading.Thread(target=run_pipeline, args=(pipeline_id,))
    thread.daemon = True
    thread.start()
    
    return jsonify(pipeline)

def update_stage(pipeline_id, stage_index, status, log_message, metrics=None):
    if pipeline_id in pipelines:
        pipeline = pipelines[pipeline_id]
        stage = pipeline['stages'][stage_index]
        stage['status'] = status
        stage['logs'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {log_message}")
        stage['timestamp'] = datetime.now().isoformat()
        
        if metrics:
            pipeline['metrics'].update(metrics)
        
        completed = sum(1 for s in pipeline['stages'] if s['status'] in ['completed', 'failed'])
        pipeline['progress'] = int((completed / len(pipeline['stages'])) * 100)
        
        if status == 'running':
            pipeline['current_stage'] = stage['name']
        elif status == 'completed' and stage_index == len(pipeline['stages']) - 1:
            pipeline['status'] = 'completed'
            pipeline['current_stage'] = 'finished'
        elif status == 'failed':
            pipeline['status'] = 'failed'
        
        pipeline['updated_at'] = datetime.now().isoformat()

def run_pipeline(pipeline_id):
    pipeline = pipelines[pipeline_id]
    model_type = pipeline['model_type']
    
    try:
        update_stage(pipeline_id, 0, 'running', 'Loading dataset...')
        time.sleep(1)
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
        update_stage(pipeline_id, 0, 'completed', f'Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features')
        
        update_stage(pipeline_id, 1, 'running', 'Preprocessing data...')
        time.sleep(1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        update_stage(pipeline_id, 1, 'completed', f'Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}')
        
        update_stage(pipeline_id, 2, 'running', 'Engineering features...')
        time.sleep(1)
        update_stage(pipeline_id, 2, 'completed', 'Feature engineering completed')
        
        update_stage(pipeline_id, 3, 'running', f'Training {model_type}...')
        time.sleep(2)
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'neural_network':
            model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        update_stage(pipeline_id, 3, 'completed', f'Model trained: {model_type}')
        
        update_stage(pipeline_id, 4, 'running', 'Evaluating model...')
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1': float(f1_score(y_test, y_pred, average='weighted'))
        }
        update_stage(pipeline_id, 4, 'completed', f'Evaluation complete', metrics)
        
        update_stage(pipeline_id, 5, 'running', 'Saving model locally...')
        time.sleep(1)
        model_filename = f"model_{pipeline_id}_{model_type}.joblib"
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
        joblib.dump(model, model_path)
        
        metadata = {
            'pipeline_id': pipeline_id,
            'model_type': model_type,
            'metrics': metrics,
            'feature_count': X.shape[1],
            'created_at': datetime.now().isoformat()
        }
        metadata_path = model_path.replace('.joblib', '.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        pipeline['model_path'] = model_path
        models_metadata[model_filename] = metadata
        update_stage(pipeline_id, 5, 'completed', f'Model saved: {model_filename}')
        
        update_stage(pipeline_id, 6, 'running', 'Uploading to HDFS...')
        hdfs_path = f'/models/{model_filename}'
        hdfs_metadata_path = f'/models/{model_filename.replace(".joblib", ".json")}'
        
        try:
            client = get_hdfs_client()
            if client:
                with open(model_path, 'rb') as local_file:
                    client.write(hdfs_path, local_file, overwrite=True)
                
                with open(metadata_path, 'rb') as meta_file:
                    client.write(hdfs_metadata_path, meta_file, overwrite=True)
                
                file_status = client.status(hdfs_path)
                pipeline['hdfs_path'] = hdfs_path
                update_stage(pipeline_id, 6, 'completed', 
                           f'Model uploaded to HDFS: {hdfs_path} (size: {file_status["length"]} bytes)')
            else:
                update_stage(pipeline_id, 6, 'failed', 'HDFS connection failed')
        except Exception as e:
            update_stage(pipeline_id, 6, 'failed', f'HDFS upload error: {str(e)}')
        
    except Exception as e:
        update_stage(pipeline_id, 0, 'failed', f'Pipeline error: {str(e)}')
        if pipeline_id in pipelines:
            pipelines[pipeline_id]['status'] = 'failed'

@app.route('/api/pipelines/<pipeline_id>', methods=['GET'])
@token_required
def get_pipeline(current_user, pipeline_id):
    if pipeline_id in pipelines:
        return jsonify(pipelines[pipeline_id])
    return jsonify({'error': 'Pipeline not found'}), 404

@app.route('/api/models', methods=['GET'])
@token_required
def get_models(current_user):
    models = []
    
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith('.joblib'):
            metadata_path = filename.replace('.joblib', '.json')
            metadata = {}
            if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], metadata_path)):
                with open(os.path.join(app.config['UPLOAD_FOLDER'], metadata_path)) as f:
                    metadata = json.load(f)
            
            models.append({
                'name': filename,
                'path': os.path.join(app.config['UPLOAD_FOLDER'], filename),
                'source': 'local',
                'size': os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], filename)),
                'metadata': metadata
            })
    
    try:
        client = get_hdfs_client()
        if client:
            try:
                hdfs_models = client.list('/models')
                for hdfs_model in hdfs_models:
                    if hdfs_model.endswith('.joblib'):
                        hdfs_path = f'/models/{hdfs_model}'
                        status = client.status(hdfs_path)
                        
                        metadata = {}
                        try:
                            meta_path = hdfs_path.replace('.joblib', '.json')
                            with client.read(meta_path) as reader:
                                metadata = json.loads(reader.read().decode())
                        except:
                            pass
                        
                        models.append({
                            'name': hdfs_model,
                            'path': hdfs_path,
                            'source': 'hdfs',
                            'size': status['length'],
                            'metadata': metadata
                        })
            except:
                pass
    except Exception as e:
        print(f"Error listing HDFS models: {e}")
    
    return jsonify(models)

@app.route('/api/hdfs/status', methods=['GET'])
@token_required
def get_hdfs_status(current_user):
    status = {
        'connected': False,
        'namenode': HDFS_NAMENODE,
        'files': [],
        'total_size': 0
    }
    
    try:
        client = get_hdfs_client()
        if client:
            status['connected'] = True
            
            try:
                files = client.list('/models')
                for f in files:
                    try:
                        file_status = client.status(f'/models/{f}')
                        status['files'].append({
                            'name': f,
                            'path': f'/models/{f}',
                            'size': file_status['length'],
                            'replication': file_status.get('replication', 1),
                            'timestamp': datetime.fromtimestamp(file_status['modificationTime']/1000).isoformat()
                        })
                        status['total_size'] += file_status['length']
                    except:
                        pass
            except:
                pass
            
            try:
                response = requests.get(f"{HDFS_NAMENODE}/jmx?qry=Hadoop:service=NameNode,name=NameNodeInfo", timeout=5)
                if response.status_code == 200:
                    jmx_data = response.json()
                    if 'beans' in jmx_data and len(jmx_data['beans']) > 0:
                        bean = jmx_data['beans'][0]
                        status['cluster_info'] = {
                            'used': bean.get('Used', 0),
                            'capacity': bean.get('Capacity', 0),
                            'free': bean.get('Remaining', 0)
                        }
            except Exception as e:
                print(f"Could not get cluster info: {e}")
    
    except Exception as e:
        status['error'] = str(e)
    
    return jsonify(status)

@app.route('/api/hdfs/browse', methods=['POST'])
@token_required
def browse_hdfs(current_user):
    data = request.get_json()
    path = data.get('path', '/')
    
    result = {'path': path, 'entries': [], 'error': None}
    
    try:
        client = get_hdfs_client()
        if not client:
            result['error'] = 'Not connected to HDFS'
            return jsonify(result)
        
        try:
            entries = client.list(path)
            for entry in entries:
                entry_path = f"{path}/{entry}" if path != '/' else f"/{entry}"
                try:
                    status = client.status(entry_path)
                    is_dir = status['type'] == 'DIRECTORY'
                    result['entries'].append({
                        'name': entry,
                        'path': entry_path,
                        'type': 'directory' if is_dir else 'file',
                        'size': status.get('length', 0) if not is_dir else 0,
                        'modified': datetime.fromtimestamp(status['modificationTime']/1000).isoformat()
                    })
                except:
                    pass
        except Exception as e:
            result['error'] = str(e)
    
    except Exception as e:
        result['error'] = str(e)
    
    return jsonify(result)

@app.route('/api/hdfs/download', methods=['POST'])
@token_required
def download_from_hdfs(current_user):
    data = request.get_json()
    hdfs_path = data.get('path')
    
    if not hdfs_path:
        return jsonify({'error': 'Path required'}), 400
    
    try:
        client = get_hdfs_client()
        if not client:
            return jsonify({'error': 'Not connected to HDFS'}), 503
        
        with client.read(hdfs_path) as reader:
            content = reader.read()
        
        filename = os.path.basename(hdfs_path)
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(local_path, 'wb') as f:
            f.write(content)
        
        return jsonify({
            'success': True,
            'message': f'Downloaded {filename} to local storage',
            'local_path': local_path,
            'size': len(content)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/inference', methods=['POST'])
@token_required
def inference(current_user):
    data = request.get_json()
    model_name = data.get('model_name')
    features = data.get('features')
    
    if not model_name or not features:
        return jsonify({'error': 'model_name and features required'}), 400
    
    model_path = None
    source = 'local'
    
    local_path = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
    if os.path.exists(local_path):
        model_path = local_path
        source = 'local'
    else:
        try:
            client = get_hdfs_client()
            if client:
                hdfs_path = f'/models/{model_name}'
                with client.read(hdfs_path) as reader:
                    content = reader.read()
                
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{model_name}')
                with open(temp_path, 'wb') as f:
                    f.write(content)
                model_path = temp_path
                source = 'hdfs'
        except Exception as e:
            return jsonify({'error': f'Model not found: {str(e)}'}), 404
    
    try:
        model = joblib.load(model_path)
        features_array = np.array(features).reshape(1, -1)
        
        prediction = model.predict(features_array)[0]
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_array)[0]
            probabilities = {f'class_{i}': float(p) for i, p in enumerate(proba)}
        
        result = {
            'prediction': int(prediction),
            'probabilities': probabilities,
            'model_name': model_name,
            'source': source
        }
        
        if source == 'hdfs' and os.path.exists(model_path):
            os.remove(model_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload/model', methods=['POST'])
@token_required
def upload_model(current_user):
    data = request.get_json()
    source_type = data.get('source_type')
    source_url = data.get('source_url')
    
    task_id = str(uuid.uuid4())[:8]
    upload_tasks[task_id] = {
        'id': task_id,
        'status': 'pending',
        'progress': 0,
        'message': 'Initializing upload...',
        'source_type': source_type,
        'source_url': source_url,
        'created_at': datetime.now().isoformat()
    }
    
    thread = threading.Thread(target=process_model_upload, args=(task_id, data))
    thread.daemon = True
    thread.start()
    
    return jsonify(upload_tasks[task_id])

def process_model_upload(task_id, data):
    task = upload_tasks[task_id]
    
    try:
        source_type = data.get('source_type')
        model_filename = data.get('filename', f'model_external_{task_id}.joblib')
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
        
        if source_type in ['url', 'nexus']:
            task['status'] = 'downloading'
            task['progress'] = 10
            task['message'] = f'Downloading from {data.get("source_url")}...'
            
            response = requests.get(data['source_url'], stream=True)
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            task['progress'] = 10 + int((downloaded / total_size) * 40)
                
                task['progress'] = 50
                task['message'] = 'Download complete, validating...'
            else:
                raise Exception(f'Failed to download: HTTP {response.status_code}')
        
        try:
            model = joblib.load(model_path)
            task['progress'] = 70
            task['message'] = 'Model validated successfully'
            
            metadata = {
                'source': 'external',
                'source_type': source_type,
                'source_url': data.get('source_url'),
                'uploaded_at': datetime.now().isoformat(),
                'model_type': type(model).__name__
            }
            
            metadata_path = model_path.replace('.joblib', '.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            task['status'] = 'uploading_hdfs'
            task['progress'] = 80
            task['message'] = 'Uploading to HDFS...'
            
            client = get_hdfs_client()
            if client:
                hdfs_path = f'/models/{model_filename}'
                with open(model_path, 'rb') as local_file:
                    client.write(hdfs_path, local_file, overwrite=True)
                
                meta_hdfs_path = hdfs_path.replace('.joblib', '.json')
                with open(metadata_path, 'rb') as meta_file:
                    client.write(meta_hdfs_path, meta_file, overwrite=True)
                
                task['hdfs_path'] = hdfs_path
                task['progress'] = 100
                task['message'] = f'Successfully uploaded to HDFS: {hdfs_path}'
            else:
                task['progress'] = 90
                task['message'] = 'Model saved locally (HDFS unavailable)'
            
            task['status'] = 'completed'
            task['model_path'] = model_path
            task['filename'] = model_filename
            
        except Exception as e:
            task['status'] = 'failed'
            task['message'] = f'Model validation failed: {str(e)}'
            if os.path.exists(model_path):
                os.remove(model_path)
    
    except Exception as e:
        task['status'] = 'failed'
        task['message'] = f'Upload failed: {str(e)}'

@app.route('/api/upload/tasks/<task_id>', methods=['GET'])
@token_required
def get_upload_task(current_user, task_id):
    if task_id in upload_tasks:
        return jsonify(upload_tasks[task_id])
    return jsonify({'error': 'Task not found'}), 404

@app.route('/api/models/<model_name>/test', methods=['POST'])
@token_required
def test_model(current_user, model_name):
    try:
        model_path = None
        source = 'local'
        
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
        if os.path.exists(local_path):
            model_path = local_path
            source = 'local'
        else:
            try:
                client = get_hdfs_client()
                if client:
                    hdfs_path = f'/models/{model_name}'
                    with client.read(hdfs_path) as reader:
                        content = reader.read()
                    
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'test_{model_name}')
                    with open(temp_path, 'wb') as f:
                        f.write(content)
                    model_path = temp_path
                    source = 'hdfs'
            except:
                pass
        
        if not model_path:
            return jsonify({'error': 'Model not found'}), 404
        
        model = joblib.load(model_path)
        
        try:
            metadata_path = model_path.replace('.joblib', '.json')
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                feature_count = metadata.get('feature_count', 20)
            else:
                feature_count = 20
        except:
            feature_count = 20
        
        test_samples = 5
        results = []
        
        for i in range(test_samples):
            np.random.seed(i)
            test_features = np.random.randn(1, feature_count).tolist()[0]
            
            prediction = model.predict([test_features])[0]
            probabilities = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([test_features])[0]
                probabilities = {f'class_{j}': float(p) for j, p in enumerate(proba)}
            
            results.append({
                'sample_id': i + 1,
                'features': test_features[:5],
                'prediction': int(prediction),
                'probabilities': probabilities
            })
        
        if source == 'hdfs' and os.path.exists(model_path):
            os.remove(model_path)
        
        return jsonify({
            'model_name': model_name,
            'source': source,
            'test_results': results,
            'samples_tested': test_samples
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting MLOps Platform Backend...")
    print(f"HDFS Namenode: {HDFS_NAMENODE}")
    
    for i in range(30):
        try:
            client = get_hdfs_client()
            if client:
                print("✓ Connected to HDFS")
                break
        except:
            pass
        time.sleep(2)
    else:
        print("⚠ HDFS not available, will retry on demand")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
