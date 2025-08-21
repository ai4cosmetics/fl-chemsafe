"""Flower server app for federated SMILES CNN training."""

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
import json
from pathlib import Path

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from vertical_fl.task import (
    SMILESCNNModel, 
    get_model_params, 
    set_model_params,
    load_data, 
    evaluate_model
)


def weighted_average(metrics):
    """Aggregate evaluation metrics weighted by number of examples."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}


def evaluate_global_model(server_round, parameters, config):
    """Evaluate global model and save comprehensive metrics."""
    # Get vocab size
    base_dir = Path(__file__).parent.parent
    vocab_path = base_dir / "data/vocab.json"
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
        vocab_size = len(vocab)
    
    # Load test data
    _, _, test_data, test_labels, _ = load_data(0)
    
    # Create model and set parameters
    model = SMILESCNNModel(vocab_size=vocab_size)
    
    # Convert Parameters to list format for set_model_params
    if isinstance(parameters, Parameters):
        params_list = parameters_to_ndarrays(parameters)
    else:
        params_list = parameters
    
    set_model_params(model, params_list)
    
    # Evaluate with comprehensive metrics
    metrics = evaluate_model(model, test_data, test_labels)
    
    print(f"   Round {server_round} - Global Model: AUC={metrics['auc']:.3f}, Acc={metrics['accuracy']:.3f}")
    
    # Save final round results
    if server_round == 10:  # Final round
        results_path = base_dir / "results/federated_results.json"
        results_path.parent.mkdir(exist_ok=True)
        
        final_results = {
            "Federated Model": metrics,
            "round": server_round,
            "timestamp": str(Path().cwd())
        }
        
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nFinal federated results saved to {results_path}")
    
    return metrics["accuracy"], metrics


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    
    # Get correct vocab size from data
    base_dir = Path(__file__).parent.parent
    vocab_path = base_dir / "data/vocab.json"
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
            vocab_size = len(vocab)
    else:
        vocab_size = 36  # Default fallback
    
    # Initialize model to get initial parameters
    model = SMILESCNNModel(vocab_size=vocab_size)
    initial_parameters = ndarrays_to_parameters(get_model_params(model))
    
    print(f"Server initialized with vocab_size={vocab_size}, model params: {len(get_model_params(model))}")
    
    # Configure strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=2,  # Minimum 2 clients for training
        min_evaluate_clients=2,  # Minimum 2 clients for evaluation
        min_available_clients=2,  # Wait for 2 clients
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=evaluate_global_model,
        initial_parameters=initial_parameters,
    )
    
    # Server config
    config = ServerConfig(num_rounds=10)
    
    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)