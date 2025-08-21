"""Flower client app for federated SMILES CNN training."""

import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vertical_fl.task import SMILESCNNModel, load_data, get_model_params, set_model_params, train_model, evaluate_model


class SMILESClient(NumPyClient):
    """Flower client for SMILES CNN."""
    
    def __init__(self, client_id: int):
        self.client_id = client_id
        
        try:
            # Load data
            self.train_data, self.train_labels, self.test_data, self.test_labels, vocab_size = load_data(client_id)
            
            # Initialize model
            self.model = SMILESCNNModel(vocab_size=vocab_size)
            
        except Exception as e:
            print(f"Client {client_id} initialization failed: {e}")
            raise
    
    def get_parameters(self, config):
        """Return model parameters."""
        try:
            params = get_model_params(self.model)
            return params
        except Exception as e:
            print(f"Client {self.client_id} get_parameters failed: {e}")
            raise
    
    def fit(self, parameters, config):
        """Train model with current parameters."""
        try:
            # Set parameters from server
            set_model_params(self.model, parameters)
            
            # Train model
            epochs = config.get("epochs", 5)
            loss = train_model(self.model, self.train_data, self.train_labels, epochs)
            
            return get_model_params(self.model), len(self.train_data), {"loss": loss}
            
        except Exception as e:
            print(f"Client {self.client_id} fit failed: {e}")
            raise
    
    def evaluate(self, parameters, config):
        """Evaluate model with current parameters."""
        try:
            # Set parameters from server
            set_model_params(self.model, parameters)
            
            # Evaluate model
            metrics = evaluate_model(self.model, self.test_data, self.test_labels)
            
            return metrics["loss"], len(self.test_data), metrics
            
        except Exception as e:
            print(f"Client {self.client_id} evaluate failed: {e}")
            raise


def client_fn(context: Context) -> SMILESClient:
    """Construct a client given context."""
    try:
        client_id = int(context.node_config["partition-id"])
        return SMILESClient(client_id)
    except Exception as e:
        print(f"Client creation failed: {e}")
        raise


# Flower ClientApp
app = ClientApp(client_fn=client_fn) 