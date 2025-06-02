import flwr as fl
import numpy as np
import os
from task import plot_federated_histogram_comparison

bins = np.linspace(-12, 0, 20)
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

class HistogramStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(min_fit_clients=3, min_available_clients=3)
        self.agg_hist = {"Epidermis": None, "Dermis": None}
        self.noisy_hist = {"Epidermis": None, "Dermis": None}

    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""
        config = {"round": server_round}
        
        # Sample clients
        sample_size = max(self.min_fit_clients, 1)
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_available_clients,
        )
        
        # Return client instructions
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]

    def aggregate_fit(self, rnd, results, failures):
        num_bins = len(bins) - 1
        agg_hist_epidermis = np.zeros(num_bins)
        agg_hist_dermis = np.zeros(num_bins)
        
        for _, fit_res in results:
            metrics = fit_res.metrics
            for i in range(num_bins):
                agg_hist_epidermis[i] += metrics.get(f"hist_epidermis_{i}", 0.0)
                agg_hist_dermis[i] += metrics.get(f"hist_dermis_{i}", 0.0)
        
        if rnd == 1:
            self.agg_hist["Epidermis"] = agg_hist_epidermis
            self.agg_hist["Dermis"] = agg_hist_dermis
            os.makedirs(PLOT_DIR, exist_ok=True)
            return None, {}
        else:  # rnd == 2
            self.noisy_hist["Epidermis"] = agg_hist_epidermis
            self.noisy_hist["Dermis"] = agg_hist_dermis
            
            os.makedirs(PLOT_DIR, exist_ok=True)
            plot_federated_histogram_comparison(self.agg_hist, self.noisy_hist, bins, PLOT_DIR)
            return None, {"stop": True}  # Stop after round 2
        
        return None, {}

if __name__ == "__main__":
    strategy = HistogramStrategy()
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy
    )