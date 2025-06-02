# Federated Learning for Chemical Safety Assessment

This repository contains **three distinct federated learning implementations** for hazard and exposure chemical safety assessment.

## 🧪 Projects

### [Federated Analytics](./federated_analytics/)
- **Model**: Histogram aggregation
- **Task**: Dermal permeability (logKp) analysis
- **Data**: HuskinDB + SkinPiX + NCSU_USEPA datasets  
- **Approach**: Privacy-preserving analytics with differential privacy

### [Horizontal Federated Learning](./horizontal_fl/)
- **Model**: XGBoost 
- **Task**: Skin sensitisation prediction (LLNA)
- **Data**: AI4Cosmetics + SkinDoctorCP datasets
- **Approach**: Organisations with different datasets collaborate

### [Vertical Federated Learning](./vertical_fl/)
- **Model**: SMILES CNN
- **Task**: Mutagenicity prediction 
- **Data**: Non-IID molecular data splits
- **Approach**: Clients with different data distributions collaborate

## 🚀 Quick Start

Each project is self-contained with its own setup and usage instructions:

```bash
# For federated analytics (Histograms)
cd federated_analytics/
# Follow federated_analytics/README.md

# For horizontal FL (XGBoost)
cd horizontal_fl/
# Follow horizontal_fl/README.md

# For vertical FL (SMILES CNN)  
cd vertical_fl/
# Follow vertical_fl/README.md
```

## 📁 Repository Structure

```
fl-chemsafe/
├── federated_analytics/                     # Privacy-preserving histogram aggregation
│   ├── README.md                            # Setup and usage instructions
│   ├── client_app.py                        # Flower client implementation
│   ├── server_app.py                        # Flower server implementation  
│   ├── task.py                              # Data loading and model logic
│   ├── pyproject.toml                       # Python dependencies
│   ├── data/                                # Datasets
│   └── plots/                               # Generated visualisations
├── horizontal_fl/                           # XGBoost federated learning
│   ├── README.md                            # Setup and usage instructions
│   ├── client_app.py                        # Flower client implementation
│   ├── server_app.py                        # Flower server implementation
│   ├── task.py                              # XGBoost model logic
│   ├── data_preparation.py                  # Dataset preprocessing
│   ├── baseline_models_evaluation.py        # Performance comparison
│   ├── create_performance_plot.py           # Performance visualisation
│   ├── create_umap_plot.py                  # UMAP embedding plots
│   ├── pyproject.toml                       # Python dependencies
│   ├── data/                                # Datasets
│   ├── models/                              # Saved model files
│   ├── results/                             # Evaluation results
│   └── plots/                               # Generated visualisations
├── vertical_fl/                             # SMILES CNN federated learning
│   ├── README.md                            # Setup and usage instructions  
│   ├── client_app.py                        # Flower client implementation
│   ├── server_app.py                        # Flower server implementation
│   ├── task.py                              # CNN model and data logic
│   ├── data_preparation.py                  # Non-IID data splits
│   ├── evaluate_local_models.py             # Local model evaluation
│   ├── create_performance_visualization.py  # Performance plots
│   ├── pyproject.toml                       # Python dependencies
│   ├── data/                                # Datasets
│   ├── results/                             # Evaluation results
│   └── plots/                               # Generated visualisations
└── README.md                                # This file
```