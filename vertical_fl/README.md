| tags         | property | framework |
|--------------|----------|-----------|
| SMILES CNN   | Mutagenicity | Flower     

# Federated Learning for Chemical Mutagenicity Prediction

This project implements and compares **federated vs local approaches** for mutagenicity prediction using SMILES molecular representations with CNN models:

1. **Local Models** - Individual clients train separate CNN models on their non-IID data distributions
2. **Federated Model** - Distributed CNN training using FedAvg aggregation without sharing raw data

## Setup

1. Install dependencies:
   ```bash
   pip install flwr torch scikit-learn pandas numpy plotly rdkit kaleido
   ```

2. Ensure your data files are prepared in `vertical_fl/data/`.

## Usage

### 1. Prepare Data
```bash
cd vertical_fl
python data_preparation.py
```

### 2. Run Federated Learning
```bash
flwr run .
```

### 3. Evaluate Local Models
```bash
python evaluate_local_models.py
```

### 4. Create Visualizations
```bash
python create_performance_visualization.py
```

### 5. View results
Results are saved as:
```
vertical_fl/results/federated_results.json
vertical_fl/results/local_model_results.json
vertical_fl/plots/performance_comparison.png
vertical_fl/plots/performance_radar.png
```

## Federated Learning Implementation Details
- **Strategy**: FedAvg (parameter averaging)
- **Rounds**: 10 rounds of training
- **Clients**: 2 clients with non-IID data distributions
- **Model**: SMILES CNN (embedding + 2 conv layers + global pooling)
- **Data Split**: Client A (~80% mutagenic), Client B (~20% mutagenic)
- **Privacy**: Only model parameters are shared, never raw molecular data
- **Evaluation**: Shared test set for fair comparison

## Files

### Data Preparation
- `data_preparation.py`: SMILES tokenization and non-IID data splitting

### Federated Learning
- `server_app.py`: Flower server for coordinating federated CNN training
- `client_app.py`: Flower client for local CNN training on private SMILES data
- `task.py`: SMILES CNN model definition and training utilities
- `pyproject.toml`: Flower project configuration

### Local Model Evaluation
- `evaluate_local_models.py`: Trains and evaluates individual client models

### Visualisation
- `create_performance_visualization.py`: Performance comparison charts (bar + radar)

### Directories
- `data/`: Directory for SMILES datasets and vocabulary
- `results/`: Output directory for federated and local model metrics
- `plots/`: Output directory for performance visualizations

## Model Architecture
- **Input**: SMILES sequences (molecular representations)
- **Embedding**: 32-dimensional character embeddings
- **Conv Layers**: 2 Conv1D layers (64, 128 filters) with ReLU and MaxPooling
- **Output**: Global max pooling + fully connected layer for binary classification
- **Loss**: BCEWithLogitsLoss for mutagenicity prediction

## Notes
- **Non-IID Setting**: Clients have different distributions of mutagenic vs non-mutagenic compounds
- **Convergence**: 10 rounds typically sufficient for stable federated performance
- **Evaluation Metrics**: AUC, Accuracy, Precision, Recall, F1-score
- **Results**: Federated model achieves balanced performance, outperforming individual local models
