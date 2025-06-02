| tags         | property | framework |
|--------------|----------|-----------|
| analytics    | logKp    | Flower    |
| tabular      |          |           |

# Federated Analytics: Histogram Aggregation with Gaussian Noise on Datapoints

This project demonstrates federated analytics using Flower, where each client computes local histograms of dermal permeability data and shares them with a central server. The server aggregates the histograms and visualises the results, comparing the original data with data that has Gaussian noise added to the original datapoints (for differential privacy).

## Setup

1. Install dependencies:
   ```bash
   pip install flwr numpy pandas matplotlib plotly
   ```

2. Ensure your data files are in `federated_analytics/data/` (e.g., `HuskinDB_clean.csv`, `SkinPiX_clean.csv`, `NCSU_USEPA_clean.csv`).

## Usage

### 1. Start the server (in one terminal):
```bash
python federated_analytics/server_app.py
```

### 2. In three other terminals, run each client:
```bash
LOCAL_DATASET=federated_analytics/data/HuskinDB_clean.csv python federated_analytics/client_app.py

LOCAL_DATASET=federated_analytics/data/SkinPiX_clean.csv python federated_analytics/client_app.py

LOCAL_DATASET=federated_analytics/data/NCSU_USEPA_clean.csv python federated_analytics/client_app.py
```

### 3. View results
The comparison plot will be saved as:
```
federated_analytics/plots/federated_histogram_comparison.png
```

### 4. (Optional) Generate initial dataset plots
To generate plots of all datasets and individual histograms, run:
```
python federated_analytics/task.py
```
The plots will be saved in the `federated_analytics/plots/` directory.


> Usually, this is not the case. Given that it is a simulated use case, we have access to the original data and can perform such operations.

## Privacy Implementation Details
- **Round 1**: Clients send histograms computed from original datapoints
- **Round 2**: Clients add Gaussian noise to original datapoints before computing histograms (ε=1)
- **Result**: Side-by-side comparison showing the impact of federated aggregation on data utility

## Files
- `client_app.py`: Client logic for computing and sending histograms. Adds Gaussian noise to datapoints in round 2.
- `server_app.py`: Server logic for aggregating histograms and generating comparison plots.
- `task.py`: Plotting utilities and Gaussian noise functions.
- `data/`: Directory for local dataset CSV files.
- `plots/`: Output directory for generated plots.

## Notes
- The left panel shows the aggregated histogram without noise.
- The right panel shows the result with Gaussian noise applied to original datapoints (ε=1).
- The server runs for 2 rounds: first collecting clean histograms, then collecting histograms from noisy datapoints.
- Clients automatically handle both rounds without manual intervention.
- The server stops automatically after completing both rounds and generating the comparison plot.
- To experiment with privacy levels, modify the `epsilon` parameter in `client_app.py` lines 23-24.