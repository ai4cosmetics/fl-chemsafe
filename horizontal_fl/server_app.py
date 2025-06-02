"""
Flower federated learning server coordinating multiple organizations.
Aggregates model updates using FedXgbBagging strategy.
"""

import flwr as fl
import os
from flwr.server.strategy import FedXgbBagging
from flwr.common import Parameters


def config_func(rnd: int):
    """Return a configuration with global round number."""
    config = {
        "global_round": str(rnd),
        "local_epochs": "1",
    }
    return config


# Configure strategy
strategy = FedXgbBagging(
    fraction_fit=1.0,
    fraction_evaluate=0.0,  # No evaluation during training
    min_fit_clients=2,
    min_evaluate_clients=0,
    min_available_clients=2,
    on_fit_config_fn=config_func,
    initial_parameters=Parameters(tensor_type="", tensors=[bytes()]),
)

# Start server
if __name__ == "__main__":
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )
    
    # Save final global model in models folder
    os.makedirs('models', exist_ok=True)
    if hasattr(strategy, 'global_model') and strategy.global_model is not None:
        with open('models/global_federated_model.json', 'wb') as f:
            f.write(strategy.global_model)
