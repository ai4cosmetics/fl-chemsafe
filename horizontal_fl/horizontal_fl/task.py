"""
XGBoost model configuration and utilities for federated learning.
Provides consistent model parameters and data loading functions for clients.
"""

import numpy as np
import os, sys
current_dir = os.path.dirname(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from data_preparation import load_data_splits


def get_xgboost_params(y_train):
    """Get XGBoost parameters matching baseline models for comparison"""
    # Calculate class imbalance (same as baseline models)
    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    # Use same parameters as baseline models for comparison
    return {
        'objective': 'binary:logistic',
        'n_estimators': 200,           
        'max_depth': 6,                
        'learning_rate': 0.05,            
        'min_child_weight': 1,         
        'subsample': 0.8,             
        'colsample_bytree': 0.8,      
        'reg_alpha': 0.1,             
        'reg_lambda': 1.0,             
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,            
        'eval_metric': 'logloss',      
        'verbosity': 0                 
    }


def get_client_data(client_name):
    """Get training data for a specific client"""
    data_splits = load_data_splits()
    
    if client_name == 'ai4cosmetics':
        return data_splits['ai4cosmetics']['train'][0], data_splits['ai4cosmetics']['train'][1]
    elif client_name == 'skindoctorcp':
        return data_splits['skindoctorcp']['train'][0], data_splits['skindoctorcp']['train'][1]
    else:
        raise ValueError(f"Unknown client: {client_name}")


def get_global_test_data():
    """Get global test data for evaluation"""
    splits = load_data_splits()
    return splits['global_test'][0], splits['global_test'][1]