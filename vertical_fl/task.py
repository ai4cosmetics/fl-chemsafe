"""Task for federated learning with SMILES CNNs."""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


class SMILESCNNModel(nn.Module):
    """CNN for SMILES sequences."""
    
    def __init__(self, vocab_size=36, embedding_dim=32):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        
        x = torch.relu(self.conv1(embedded))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def get_model_params(model: torch.nn.Module) -> list:
    """Get model parameters as numpy arrays."""
    return [val.detach().cpu().numpy() for val in model.parameters()]


def set_model_params(model: torch.nn.Module, params: list):
    """Set model parameters from numpy arrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def load_data(client_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Load data for specific client."""
    data_path = Path("data/noniid_split.npz")
    vocab_path = Path("data/vocab.json")
    
    data = np.load(data_path)
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    if client_id == 0:
        train_data = torch.tensor(data['client1_sequences'], dtype=torch.long)
        train_labels = torch.tensor(data['client1_labels'], dtype=torch.float32)
    else:
        train_data = torch.tensor(data['client2_sequences'], dtype=torch.long)
        train_labels = torch.tensor(data['client2_labels'], dtype=torch.float32)
    
    test_data = torch.tensor(data['test_sequences'], dtype=torch.long)
    test_labels = torch.tensor(data['test_labels'], dtype=torch.float32)
    
    return train_data, train_labels, test_data, test_labels, len(vocab)


def train_model(model: torch.nn.Module, train_data: torch.Tensor, train_labels: torch.Tensor, epochs: int = 5):
    """Train model for specified epochs."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(train_data)
        loss = criterion(logits.view(-1), train_labels)
        loss.backward()
        optimizer.step()
    
    return loss.item()


def evaluate_model(model: torch.nn.Module, test_data: torch.Tensor, test_labels: torch.Tensor) -> Dict[str, float]:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    
    with torch.no_grad():
        logits = model(test_data)
        probs = torch.sigmoid(logits).squeeze()
        predictions = (probs > 0.5).float()
        
        # Convert to numpy for sklearn metrics
        y_true = test_labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_prob = probs.cpu().numpy()
        
        # Compute all metrics
        accuracy = (predictions == test_labels).float().mean().item()
        loss = nn.BCEWithLogitsLoss()(logits.view(-1), test_labels).item()
        
        # Handle edge cases for sklearn metrics
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5  # Default AUC if only one class present
            
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "loss": loss,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    } 