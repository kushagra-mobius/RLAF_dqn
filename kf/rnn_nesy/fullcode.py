import torch
import torch.nn as nn
import numpy as np
import sys
import os
import json
import requests
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.nesy_factory.RNNs.simple_rnn import SimpleRNN
from src.nesy_factory.RNNs.gru import GRU

# ==============================================================================
# 1. DATA FETCHING AND PROCESSING
# ==============================================================================

def get_bearer_token(token_url, username, password, product_id):
    """Gets a bearer token from the authentication endpoint."""
    headers = {
        'content-type': 'application/json',
        'mask': 'false'
    }
    auth_data = {
        "userName": username,
        "password": password,
        "productId": product_id,
        "requestType": "TENANT"
    }
    response = requests.post(token_url, headers=headers, json=auth_data)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.json()["accessToken"]

def fetch_data_from_api(schema_id, token):
    """Fetches a list of JSON objects from a URL using a bearer token."""
    url = f'https://igs.gov-cloud.ai/pi-entity-instances-service/v2.0/schemas/{schema_id}/instances/list?page=0&size=2000&showDBaaSReservedKeywords=true'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    data = {
      "dbType": "TIDB",
      "ownedOnly": True,
      "filter": {}
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    return response.json()

def process_total_metrics_data(json_data):
    """Processes the Total Metrics data, extracting nested fields."""
    records = []
    for item in json_data:
        record = {
            'totalallocatable.cpu': float(item['totalallocatable']['cpu'].replace('c', '')),
            'totalallocatable.memory': float(item['totalallocatable']['memory'].replace('TB', '')) * 1e12, # Convert TB to bytes
            'totalallocatable.storage': float(item['totalallocatable']['storage'].replace('TB', '')) * 1e12, # Convert TB to bytes
            'timestamp': item['timestamp']
        }
        records.append(record)
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    return df

def create_sequences_with_labels(data, labels, seq_length):
    """
    Creates sequences from data and assigns a label to each sequence.
    The label for a sequence is the label of its last element.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(labels[i+seq_length-1])
            
    return np.array(X), np.array(y)

# ==============================================================================
# 2. SCRIPT EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate RNN models for sequence classification and forecasting.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the JSON configuration file')
    args = parser.parse_args()

    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config = json.load(f)

    print(f"--- Running Use Case: {config['use_case']} ---")

    # --- Authentication ---
    TOKEN_URL = "https://ig.mobiusdtaas.ai/mobius-iam-service/v1.0/login"
    USERNAME = "aidtaas@gaiansolutions.com"
    PASSWORD = "Gaian@123"
    PRODUCT_ID = "c2255be4-ddf6-449e-a1e0-b4f7f9a2b636"
    
    print("Getting bearer token...")
    bearer_token = get_bearer_token(TOKEN_URL, USERNAME, PASSWORD, PRODUCT_ID)

    # --- Data Loading ---
    print("Fetching and processing data...")
    try:
        json_data = fetch_data_from_api(config['schema_id'], bearer_token)
        if config['use_case'] == "Edge Heartbeat Anomaly Filtering":
            df = pd.DataFrame(json_data)
        elif config['use_case'] == "Fine-Grained Build Cost Metering Loops":
            df = process_total_metrics_data(json_data)
        else:
            df = pd.DataFrame(json_data)

        print(f"Successfully fetched {len(df)} records from the API.")
    except requests.exceptions.HTTPError as e:
        print(f"API request failed with error: {e}")
        return
    
    # --- Feature Engineering ---
    all_columns = config['feature_columns'] + [config['target_column']]
    for col in all_columns:
        if '.' in col: # Handle nested columns for forecasting
            # The processing function already handled this
            pass
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=all_columns, inplace=True)
    
    # --- Sequence Creation and Splitting ---
    features = config['feature_columns']
    
    scaler = MinMaxScaler()
    data_for_sequencing = scaler.fit_transform(df[features].astype(float).values)
    y_labels = df[config['target_column']].astype(float).values
    
    X, y = create_sequences_with_labels(data_for_sequencing, y_labels, config['seq_length'])
    print(f"Created {len(X)} sequences of length {config['seq_length']}.")
    
    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Dataset created with {len(X_train)} training samples and {len(X_test)} test samples.")

    # --- Model Configuration and Training ---
    model_config = {
        'input_dim': len(features),
        'hidden_dim': config['hidden_dim'],
        'output_dim': 1,
        'num_layers': config['num_layers'],
        'dropout': config['dropout'],
        'optimizer': 'adam',
        'learning_rate': config['learning_rate'],
        'epochs': config['epochs'],
        'loss_function': config['loss_function']
    }
    
    if config['model_type'] == 'SimpleRNN':
        model = SimpleRNN(model_config)
    elif config['model_type'] == 'GRU':
        model = GRU(model_config)
    else:
        raise ValueError("Invalid model type specified")

    model._init_optimizer_and_criterion()
    print(f"\n{config['model_type']} model created:", model)
    
    print("\n--- Starting Model Training ---")
    for epoch in range(model_config['epochs']):
        model.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            loss = model.train_step((inputs, labels))
            total_train_loss += loss
        
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d}/{model_config['epochs']} | Train Loss: {avg_train_loss:.6f}")

    print("\n--- Finished Model Training ---")

    print("\n--- Starting Final Evaluation ---")
    eval_metrics = model.eval_step(test_loader)

    if model.loss_function_type == 'bcewithlogitsloss':
        print(f"Test Accuracy: {eval_metrics['accuracy']:.4f} | "
              f"Precision: {eval_metrics['precision']:.4f} | "
              f"Recall: {eval_metrics['recall']:.4f} | "
              f"F1-Score: {eval_metrics['f1_score']:.4f}")
    else:
        print(f"Test Loss (MSE): {eval_metrics['loss']:.6f} | "
              f"Test MSE: {eval_metrics['mse']:.6f} | "
              f"Test MAE: {eval_metrics['mae']:.6f}")
    print("\n--- Finished Final Evaluation ---")

if __name__ == "__main__":
    main()