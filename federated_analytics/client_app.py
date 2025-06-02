import flwr as fl
import numpy as np
import pandas as pd
import os
from task import add_gaussian_noise_to_datapoints

local_path = os.environ.get("LOCAL_DATASET", "federated_analytics/data/HuskinDB_clean.csv")
df = pd.read_csv(local_path)
bins = np.linspace(-12, 0, 20)

class HistogramClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        current_round = config.get("round", 1)
        
        # Get original data
        epidermis_data = df['LogKp Epidermis (cm/s)'].dropna().values
        dermis_data = df['LogKp Dermis (cm/s)'].dropna().values
        
        # Add noise to original datapoints in round 2
        if current_round == 2:
            epidermis_data = add_gaussian_noise_to_datapoints(epidermis_data, epsilon=1)
            dermis_data = add_gaussian_noise_to_datapoints(dermis_data, epsilon=1)

        # Compute histograms from (potentially noisy) datapoints
        hist_epidermis, _ = np.histogram(epidermis_data, bins=bins)
        hist_dermis, _ = np.histogram(dermis_data, bins=bins)

        metrics = {}
        for i, v in enumerate(hist_epidermis):
            metrics[f"hist_epidermis_{i}"] = float(v)
        for i, v in enumerate(hist_dermis):
            metrics[f"hist_dermis_{i}"] = float(v)
        return [], 0, metrics

    def evaluate(self, _parameters, _config):
        return 0.0, len(df), {"accuracy": 1.0}

if __name__ == "__main__":
    client = HistogramClient()
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)