"""
Trains individual/centralised models, loads federated model, evaluates all, saves results and predictions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import warnings
import os
import pickle
import json
from pathlib import Path
from data_preparation import load_data_splits

warnings.filterwarnings('ignore')

def train_xgb_model(X_train, y_train, X_test, y_test, model_name):
    """Train XGBoost model with hyperparameter optimisation"""
    if len(X_train) == 0 or len(np.unique(y_train)) < 2:
        return [model_name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, len(X_train), len(X_test), {}, None]
    
    scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train) if np.sum(y_train) > 0 else 1.0
    param_grid = {
        'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 6], 'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0.9], 'scale_pos_weight': [1.0, scale_pos_weight]
    }
    
    base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    
    try:
        search = RandomizedSearchCV(base_model, param_grid, n_iter=20, scoring='roc_auc',
                                  cv=StratifiedKFold(3, shuffle=True, random_state=42), random_state=42, verbose=0)
        search.fit(X_train, y_train)
        best_model, best_params = search.best_estimator_, search.best_params_
    except:
        best_model, best_params = base_model, {'scale_pos_weight': scale_pos_weight}
        best_model.fit(X_train, y_train)
    
    try:
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = best_model.predict(X_test)
        metrics = [roc_auc_score(y_test, y_pred_proba), accuracy_score(y_test, y_pred),
                  precision_score(y_test, y_pred, zero_division=0), recall_score(y_test, y_pred, zero_division=0),
                  f1_score(y_test, y_pred, zero_division=0), matthews_corrcoef(y_test, y_pred)]
    except:
        metrics = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    return [model_name] + metrics + [len(X_train), len(X_test), best_params, best_model]

def evaluate_federated_model(data_splits):
    """Load and evaluate federated model"""
    model_file = Path("models/global_federated_model.json")
    if not model_file.exists():
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([])]
    
    try:
        federated_model = xgb.Booster()
        federated_model.load_model(model_file)
        X_test, y_test = data_splits['global_test'][0], data_splits['global_test'][1]
        
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = federated_model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        metrics = [roc_auc_score(y_test, y_pred_proba), accuracy_score(y_test, y_pred),
                  precision_score(y_test, y_pred, zero_division=0), recall_score(y_test, y_pred, zero_division=0),
                  f1_score(y_test, y_pred, zero_division=0), matthews_corrcoef(y_test, y_pred)]
        return metrics + [y_pred_proba]
    except:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.array([])]

def run_comprehensive_evaluation():
    """Train baselines, load federated model, evaluate all, save results and predictions"""
    data_splits = load_data_splits()
    results = []
    
    # Train Individual Models
    for dataset_name in ['ai4cosmetics', 'skindoctorcp']:
        if len(data_splits[dataset_name]['train'][0]) > 0:
            if dataset_name == 'ai4cosmetics':
                model_name = "AI4Cosmetics Local"
            else:
                model_name = "SkinDoctorCP Local"
            result = train_xgb_model(data_splits[dataset_name]['train'][0], data_splits[dataset_name]['train'][1],
                                   data_splits[dataset_name]['test'][0], data_splits[dataset_name]['test'][1], model_name)
            results.append(result)
    
    # Train Centralised Model
    centralised_result = train_xgb_model(data_splits['combined_train'][0], data_splits['combined_train'][1],
                                       data_splits['global_test'][0], data_splits['global_test'][1], "Centralised Learning")
    results.append(centralised_result)
    
    # Evaluate Federated Model
    fed_metrics = evaluate_federated_model(data_splits)
    fed_predictions = fed_metrics[-1]
    fed_result = ['Federated Learning'] + fed_metrics[:-1] + [
        len(data_splits['ai4cosmetics']['train'][0]) + len(data_splits['skindoctorcp']['train'][0]),
        len(data_splits['global_test'][0]), {"strategy": "FedXgbBagging"}, None
    ]
    results.append(fed_result)
    
    # Save Results
    os.makedirs("results", exist_ok=True)
    columns = ['Model', 'AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'Train_Samples', 'Test_Samples', 'Best_Params']
    df_data = [{col: (json.dumps(row[i]) if col == 'Best_Params' else row[i]) for i, col in enumerate(columns)} for row in results]
    pd.DataFrame(df_data).to_csv("results/complete_comparison.csv", index=False)
    
    # Save centralized model for UMAP visualization
    os.makedirs("models", exist_ok=True)
    if centralised_result[-1] is not None:
        with open("models/centralised_combined_model.pkl", 'wb') as f:
            pickle.dump(centralised_result[-1], f)
    
    # Save predictions for global test set (centralised and federated only)
    X_test, y_test = data_splits['global_test'][0], data_splits['global_test'][1]
    predictions_data = {'true_labels': y_test}
    
    if centralised_result[-1] is not None:
        predictions_data['centralised_predictions'] = centralised_result[-1].predict_proba(X_test)[:, 1]
    if len(fed_predictions) > 0:
        predictions_data['federated_predictions'] = fed_predictions
    
    pd.DataFrame(predictions_data).to_csv("results/global_test_predictions.csv", index=False)
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_evaluation() 