"""
Flower federated learning client for individual organizations.
Trains XGBoost locally on private data without sharing raw data.
"""

import argparse
import xgboost as xgb
import flwr as fl
from flwr.client import Client
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from task import get_client_data, get_xgboost_params


class XGBoostClient(Client):
    
    def __init__(self, client_name, config):
        # Load data using task utility
        self.X_train, self.y_train = get_client_data(client_name)
        
        # Get XGBoost parameters
        self.params = get_xgboost_params(self.y_train)
        self.num_local_round = int(config.get("local_epochs", 1))
        self.model = None
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        # Return empty bytes if no model yet
        if self.model is None:
            local_model_bytes = bytes()
        else:
            # Save model as JSON bytes
            local_model = self.model.get_booster().save_raw("json")
            local_model_bytes = bytes(local_model)
        
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
        )
    
    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config.get("global_round", 1))
        
        if global_round == 1:
            # First round: train new model with consistent parameters
            self.model = xgb.XGBClassifier(**self.params)
            self.model.fit(self.X_train, self.y_train)
        else:
            # Load global model from server and continue training
            if ins.parameters.tensors and len(ins.parameters.tensors[0]) > 0:
                # Load global model
                booster = xgb.Booster()
                global_model = bytearray(ins.parameters.tensors[0])
                booster.load_model(global_model)
                
                # Create new model with additional trees
                self.model = xgb.XGBClassifier(**self.params)
                self.model.fit(self.X_train, self.y_train, xgb_model=booster)
            else:
                # Fallback: train from scratch
                self.model = xgb.XGBClassifier(**self.params)
                self.model.fit(self.X_train, self.y_train)
        
        # Save model as JSON bytes
        local_model = self.model.get_booster().save_raw("json")
        local_model_bytes = bytes(local_model)
        
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=len(self.X_train),
            metrics={},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('client_name', choices=['ai4cosmetics', 'skindoctorcp'])
    parser.add_argument('--server-address', default='127.0.0.1:8080')
    args = parser.parse_args()
    
    # Configuration for the client
    config = {"local_epochs": 1}
    
    fl.client.start_client(
        server_address=args.server_address,
        client=XGBoostClient(args.client_name, config)
    )
