| tags         | property | framework |
|--------------|----------|-----------|
| XGBoost      | LLNA     | Flower     

# Horizontal Federated Learning for Chemical Safety Prediction

This project implements and compares **three distinct ML approaches** for skin sensitisation prediction using LLNA (Local Lymph Node Assay) data from AI4Cosmetics and SkinDoctorCP datasets:

1. **Individual Models** - Organisations train separate models on their own private data
2. **Centralised Model** - All data is pooled together for optimal performance  
3. **Horizontal Federated Learning** - Distributed training without sharing raw data

## Setup

1. Install the project and all dependencies:
```bash copy
pip install -e .
```

2. Ensure your data files are in `horizontal_fl/data/`.

## Usage

### 1. Prepare Data
```bash copy
cd horizontal_fl
python data_preparation.py
```

### 2. Run Federated Learning

**Start the server (in one terminal):**
```bash copy
python horizontal_fl/server_app.py
```

**In two other terminals, run each client:**
```bash copy
python horizontal_fl/client_app.py ai4cosmetics

python horizontal_fl/client_app.py skindoctorcp
```

*Note: `task.py` is automatically imported by `client_app.py` to provide XGBoost configuration and data loading utilities.*

### 3. Run Complete Evaluation (Baseline + Federated)
```bash copy
python baseline_models_evaluation.py
```

### 4. Create Visualizations
```bash copy
python create_performance_plot.py
python create_umap_plot.py
```

### 5. View results
Results are saved as:
```
horizontal_fl/results/complete_comparison.csv
horizontal_fl/results/global_test_predictions.csv
horizontal_fl/plots/model_comparison.html
horizontal_fl/plots/chemical_space_umap.html
```

## Federated Learning Implementation Details
- **Strategy**: FedXgbBagging (tree-based model aggregation)
- **Rounds**: 1 round of training
- **Clients**: 2 organizations (AI4Cosmetics, SkinDoctorCP)
- **Privacy**: Only model parameters are shared, never raw data
- **Evaluation**: Global test set for fair comparison

## Files

### Data Preparation
- `data_preparation.py`: Data loading and preprocessing utilities

### Federated Learning
- `horizontal_fl/server_app.py`: Server logic for coordinating federated learning and saving global model
- `horizontal_fl/client_app.py`: Client logic for local XGBoost training on private data
- `horizontal_fl/task.py`: XGBoost model configuration and utilities for federated learning

### Baseline Models & Evaluation
- `baseline_models_evaluation.py`: Trains baselines, evaluates federated model, saves results

### Visualisation
- `create_comparison_plot.py`: Performance comparison visualisation
- `create_umap_plot.py`: Chemical space UMAP visualisation with predictions

### Directories
- `data/`: Directory for local dataset files
- `results/`: Output directory for generated results and metrics
- `plots/`: Output directory for visualisations

## Notes
- **Execution Order**: Run federated learning first to generate the global model, then evaluation
- **Server Configuration**: The server runs for 1 round with 2 clients automatically
- **Evaluation**: Individual models evaluated on local test sets, centralised and federated models on global test set

## Data Source
- Wilm A., Norinder U., Agea M.I., de Bruyn Kops C., Stork C., Kühnl J., Kirchmair J. Skin Doctor CP: Conformal prediction of the skin sensitization potential of small organic molecules. Chem Res Toxicol. 2021, 34(2):330-344. doi: 10.1021/acs.chemrestox.0c00253.