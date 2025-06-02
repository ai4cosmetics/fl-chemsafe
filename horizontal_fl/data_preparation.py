"""
Data loading and preprocessing for horizontal federated learning.
Creates data splits for individual clients, centralized model, and global test set.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from molvs import standardize_smiles
import pickle, os, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

class DataPreparator:
    def __init__(self, random_state=42, outlier_std_threshold=3.0, variance_threshold=0.01, correlation_threshold=0.95):
        self.random_state = random_state
        self.outlier_std_threshold = outlier_std_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.scaler = StandardScaler()
        self.feature_names = None

    def standardize_smiles(self, smiles_list):
        result = []
        for smiles in smiles_list:
            try:
                std_smiles = standardize_smiles(smiles)
                mol = Chem.MolFromSmiles(std_smiles) if std_smiles else None
                result.append(Chem.MolToSmiles(mol, canonical=True) if mol else None)
            except:
                result.append(None)
        return result

    def generate_features(self, smiles_list):
        features, maccs_len = [], 167
        for smiles in smiles_list:
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        maccs = np.array(rdMolDescriptors.GetMACCSKeysFingerprint(mol).ToList(), dtype=np.float32)
                        desc_vals = [0.0 if (val := func(mol)) is None or np.isnan(val) or np.isinf(val) else float(val) for _, func in Descriptors._descList]
                        features.append(np.hstack([maccs, desc_vals]))
                    else:
                        features.append(np.zeros(maccs_len + len(Descriptors._descList), dtype=np.float32))
                except:
                    features.append(np.zeros(maccs_len + len(Descriptors._descList), dtype=np.float32))
            else:
                features.append(np.zeros(maccs_len + len(Descriptors._descList), dtype=np.float32))
        if self.feature_names is None:
            self.feature_names = [f'MACCS_{i}' for i in range(maccs_len)] + [name for name, _ in Descriptors._descList]
        return np.array(features, dtype=np.float32)

    def clean_and_filter(self, X):
        X_clean = X.copy()
        X_clean[np.isinf(X_clean)] = np.sign(X_clean[np.isinf(X_clean)]) * 1e10
        X_clean[np.isnan(X_clean)] = 0.0
        for col in range(X_clean.shape[1]):
            mean_val, std_val = np.mean(X_clean[:, col]), np.std(X_clean[:, col])
            if std_val > 0:
                X_clean[:, col] = np.clip(X_clean[:, col], mean_val - self.outlier_std_threshold * std_val, mean_val + self.outlier_std_threshold * std_val)
        df = pd.DataFrame(X_clean, columns=self.feature_names)
        df = df.loc[:, df.nunique() > 1]
        selector = VarianceThreshold(self.variance_threshold)
        X_var = selector.fit_transform(df.values)
        var_feats = df.columns[selector.get_support()].tolist()
        df_var = pd.DataFrame(X_var, columns=var_feats)
        corr_matrix = df_var.corr().abs()
        high_corr = {corr_matrix.columns[i] if df_var[corr_matrix.columns[i]].var() < df_var[corr_matrix.columns[j]].var() else corr_matrix.columns[j]
                    for i in range(len(corr_matrix.columns)) for j in range(i+1, len(corr_matrix.columns)) 
                    if corr_matrix.iloc[i, j] > self.correlation_threshold}
        final_feats = [col for col in var_feats if col not in high_corr]
        return df_var[final_feats].values, final_feats

    def load_and_prepare_data(self, ai4_path='data/ai4cosmetics_data.csv', skin_path='data/wilm_et_al_skindoctorcp_data.csv'):
        if not Path(ai4_path).exists() or not Path(skin_path).exists():
            raise FileNotFoundError(f"Data files not found: {ai4_path}, {skin_path}")
        ai4_df, skin_df = pd.read_csv(ai4_path), pd.read_csv(skin_path)
        if 'Canonical SMILES' not in ai4_df.columns or 'SMILES' not in skin_df.columns or 'LLNA' not in ai4_df.columns or 'LLNA' not in skin_df.columns:
            raise ValueError("Missing required columns")
        ai4_df['std_smiles'] = self.standardize_smiles(ai4_df['Canonical SMILES'].values)
        skin_df['std_smiles'] = self.standardize_smiles(skin_df['SMILES'].values)
        ai4_df = ai4_df[ai4_df['std_smiles'].notna()].drop_duplicates(subset=['std_smiles']).reset_index(drop=True)
        skin_df = skin_df[skin_df['std_smiles'].notna()].drop_duplicates(subset=['std_smiles']).reset_index(drop=True)
        ai4_feats, skin_feats = self.generate_features(ai4_df['std_smiles'].values), self.generate_features(skin_df['std_smiles'].values)
        filt_feats, final_names = self.clean_and_filter(np.vstack([ai4_feats, skin_feats]))
        ai4_idx = len(ai4_df)
        ai4_data = pd.DataFrame(filt_feats[:ai4_idx], columns=final_names)
        ai4_data['LLNA'] = ai4_df['LLNA'].values
        ai4_data['source'] = 'ai4cosmetics'
        ai4_data['std_smiles'] = ai4_df['std_smiles'].values
        skin_data = pd.DataFrame(filt_feats[ai4_idx:], columns=final_names)
        skin_data['LLNA'] = skin_df['LLNA'].values
        skin_data['source'] = 'skindoctorcp'
        skin_data['std_smiles'] = skin_df['std_smiles'].values
        combined = pd.concat([ai4_data, skin_data], ignore_index=True).drop_duplicates(subset=['std_smiles']).reset_index(drop=True)
        self.ai4_full, self.skin_full, self.feature_names = ai4_data, skin_data, final_names
        return combined, ai4_data, skin_data

    def create_data_splits(self, combined):
        def split_client_data(data):
            if len(data) == 0:
                return tuple(np.array([]).reshape(0, len(self.feature_names)) if i % 2 == 0 else np.array([]) for i in range(4))
            unique_mols = data.groupby('std_smiles')['LLNA'].first().reset_index()
            try:
                train_smiles, test_smiles = train_test_split(unique_mols['std_smiles'].values, test_size=0.2, random_state=self.random_state, 
                                                           stratify=unique_mols['LLNA'].values if unique_mols['LLNA'].value_counts().min() >= 2 else None)
            except:
                train_smiles, test_smiles = train_test_split(unique_mols['std_smiles'].values, test_size=0.2, random_state=self.random_state)
            train_data, test_data = data[data['std_smiles'].isin(train_smiles)], data[data['std_smiles'].isin(test_smiles)]
            return train_data[self.feature_names].values, train_data['LLNA'].values, test_data[self.feature_names].values, test_data['LLNA'].values

        ai4_splits, skin_splits = split_client_data(self.ai4_full), split_client_data(self.skin_full)
        X_combined_train = np.vstack([d for d in [ai4_splits[0], skin_splits[0]] if len(d) > 0]) if any(len(d) > 0 for d in [ai4_splits[0], skin_splits[0]]) else np.array([]).reshape(0, len(self.feature_names))
        y_combined_train = np.hstack([l for l in [ai4_splits[1], skin_splits[1]] if len(l) > 0]) if any(len(l) > 0 for l in [ai4_splits[1], skin_splits[1]]) else np.array([])
        X_global_test = np.vstack([d for d in [ai4_splits[2], skin_splits[2]] if len(d) > 0]) if any(len(d) > 0 for d in [ai4_splits[2], skin_splits[2]]) else np.array([]).reshape(0, len(self.feature_names))
        y_global_test = np.hstack([l for l in [ai4_splits[3], skin_splits[3]] if len(l) > 0]) if any(len(l) > 0 for l in [ai4_splits[3], skin_splits[3]]) else np.array([])
        
        if len(X_combined_train) > 0:
            self.scaler.fit(X_combined_train)
            ai4_splits = (self.scaler.transform(ai4_splits[0]) if len(ai4_splits[0]) > 0 else ai4_splits[0], ai4_splits[1],
                         self.scaler.transform(ai4_splits[2]) if len(ai4_splits[2]) > 0 else ai4_splits[2], ai4_splits[3])
            skin_splits = (self.scaler.transform(skin_splits[0]) if len(skin_splits[0]) > 0 else skin_splits[0], skin_splits[1],
                          self.scaler.transform(skin_splits[2]) if len(skin_splits[2]) > 0 else skin_splits[2], skin_splits[3])
            X_combined_train, X_global_test = self.scaler.transform(X_combined_train), self.scaler.transform(X_global_test)
        
        return {'global_test': (X_global_test, y_global_test), 'ai4cosmetics': {'train': (ai4_splits[0], ai4_splits[1]), 'test': (ai4_splits[2], ai4_splits[3])}, 
                'skindoctorcp': {'train': (skin_splits[0], skin_splits[1]), 'test': (skin_splits[2], skin_splits[3])}, 'combined_train': (X_combined_train, y_combined_train), 'feature_names': self.feature_names}

def prepare_data(force_reprocess=False, **kwargs):
    cache_file = Path('data/processed_data_splits.pkl')
    if not force_reprocess and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f: return pickle.load(f)
        except: pass
    prep = DataPreparator(**kwargs)
    combined, ai4, skin = prep.load_and_prepare_data()
    splits = prep.create_data_splits(combined)
    try:
        os.makedirs('data', exist_ok=True)
        with open(cache_file, 'wb') as f: pickle.dump(splits, f)
    except: pass
    return splits

def load_data_splits(): return prepare_data(False)

if __name__ == "__main__":
    try:
        data = prepare_data()
        print(f"Data loaded successfully:")
        print(f"  AI4Cosmetics - Train: {len(data['ai4cosmetics']['train'][0])}, Test: {len(data['ai4cosmetics']['test'][0])}")
        print(f"  SkinDoctorCP - Train: {len(data['skindoctorcp']['train'][0])}, Test: {len(data['skindoctorcp']['test'][0])}")
        print(f"  Combined training set: {len(data['combined_train'][0])} samples")
        print(f"  Global test set: {len(data['global_test'][0])} samples")
        print(f"  Features: {len(data['feature_names'])}")
    except Exception as e:
        print(f"Error: {e}")
        raise 