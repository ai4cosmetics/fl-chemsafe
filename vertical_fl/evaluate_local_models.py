"""Evaluate non-IID client models."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import json
from pathlib import Path
import os, sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from vertical_fl.task import SMILESCNNModel

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# CPU optimizations
torch.set_num_threads(4)


def load_data():
    """Load non-IID split data."""
    data_path = Path("data/noniid_split.npz")
    vocab_path = Path("data/vocab.json")
    
    if not data_path.exists() or not vocab_path.exists():
        raise FileNotFoundError("Data files not found. Run data_preparation.py first.")
    
    data = np.load(data_path)
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    return {
        'client1_sequences': torch.tensor(data['client1_sequences'], dtype=torch.long),
        'client1_labels': torch.tensor(data['client1_labels'], dtype=torch.float32),
        'client2_sequences': torch.tensor(data['client2_sequences'], dtype=torch.long),
        'client2_labels': torch.tensor(data['client2_labels'], dtype=torch.float32),
        'test_sequences': torch.tensor(data['test_sequences'], dtype=torch.long),
        'test_labels': torch.tensor(data['test_labels'], dtype=torch.float32),
        'vocab_size': len(vocab)
    }


def train_model(model, train_data, train_labels, epochs=30):
    """Train CNN model."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(train_data)
        loss = criterion(logits.view(-1), train_labels)
        loss.backward()
        optimizer.step()


def evaluate_model(model, test_data, test_labels):
    """Evaluate model performance."""
    model.eval()
    
    with torch.no_grad():
        logits = model(test_data)
        probs = torch.sigmoid(logits).squeeze()
        binary = (probs > 0.5).float().numpy()
        
        acc = accuracy_score(test_labels.numpy(), binary)
        auc = roc_auc_score(test_labels.numpy(), probs.numpy())
        precision = precision_score(test_labels.numpy(), binary, zero_division=0)
        recall = recall_score(test_labels.numpy(), binary, zero_division=0)
        f1 = f1_score(test_labels.numpy(), binary, zero_division=0)
    
    return {
        'accuracy': acc,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    """Evaluate non-IID client models."""
    
    data = load_data()
    
    # Initialize and train models
    results = {}
    
    for client_id, (sequences, labels, name) in enumerate([
        (data['client1_sequences'], data['client1_labels'], "Client A (~80% mut)"),
        (data['client2_sequences'], data['client2_labels'], "Client B (~20% mut)")
    ]):
        model = SMILESCNNModel(vocab_size=data['vocab_size'])
        train_model(model, sequences, labels)
        results[name] = evaluate_model(model, data['test_sequences'], data['test_labels'])
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    with open("results/local_model_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    main() 