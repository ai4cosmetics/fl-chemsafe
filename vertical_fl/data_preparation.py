"""Data preparation for vertical federated learning with SMILES and MACCS."""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from molvs import standardize_smiles
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
import json
import warnings
warnings.filterwarnings('ignore')

# === DATA VALIDATION ===
def clean_and_validate_smiles(smiles):
    """Standardize and validate SMILES in one step."""
    try:
        standardized = standardize_smiles(smiles)
        mol = Chem.MolFromSmiles(standardized)
        return standardized if mol is not None else None
    except:
        return None

# === FEATURE CONVERSION ===
def smiles_to_sequence(smiles, vocab, max_len=100):
    sequence = [vocab.get(char, vocab['<UNK>']) for char in smiles[:max_len]]
    sequence += [vocab['<PAD>']] * (max_len - len(sequence))
    return np.array(sequence[:max_len])

def smiles_to_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)  # MACCS has 167 bits
    try:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        return np.array([int(x) for x in maccs.ToBitString()])
    except:
        return np.zeros(167)  # Handle any MACCS generation errors

def create_vocab(smiles_list):
    chars = set(''.join(smiles_list))
    vocab = {'<PAD>': 0, '<UNK>': 1}
    vocab.update({char: i+2 for i, char in enumerate(sorted(chars))})
    return vocab

# === FEATURE CLEANING ===
def clean_maccs_features(train_maccs, test_maccs, variance_threshold=0.01, correlation_threshold=0.95):
    """Clean MACCS features consistently for both train and test data."""
    # Step 1: Remove low-variance features (fit on train, apply to both)
    selector = VarianceThreshold(threshold=variance_threshold)
    train_filtered = selector.fit_transform(train_maccs)
    test_filtered = selector.transform(test_maccs)
    
    # Step 2: Remove highly correlated features (based on train data)
    if train_filtered.shape[1] > 1:
        corr_matrix = np.corrcoef(train_filtered.T)
        upper_tri = np.triu(np.abs(corr_matrix), k=1)
        to_remove = np.where(upper_tri > correlation_threshold)[1]
        keep_indices = [i for i in range(train_filtered.shape[1]) if i not in to_remove]
        
        train_cleaned = train_filtered[:, keep_indices]
        test_cleaned = test_filtered[:, keep_indices]
    else:
        train_cleaned = train_filtered
        test_cleaned = test_filtered
    
    return train_cleaned, test_cleaned, selector

# === DATA PROCESSING ===
def process_dataset():
    # Load and clean data
    df = pd.read_csv('data/xu2012_data.csv')
    df['label'] = df['Labels'].map({'mutagens': 1, 'non-mutagens': 0})
    df = df.dropna(subset=['label'])
    
    # Standardize and validate SMILES in one step
    df['smiles'] = df['SMILES'].apply(clean_and_validate_smiles)
    df = df.dropna(subset=['smiles'])  # Remove invalid SMILES
    df = df.drop_duplicates(subset=['smiles'])[['smiles', 'label']]
    
    return df

def create_noniid_split(df):
    """Create non-IID horizontal splits based on label distribution."""
    # Create vocabulary
    vocab = create_vocab(df['smiles'].tolist())
    
    # Separate by labels
    mutagenic_df = df[df['label'] == 1].copy()
    non_mutagenic_df = df[df['label'] == 0].copy()
    
    print(f"Original distribution: {len(mutagenic_df)} mutagenic, {len(non_mutagenic_df)} non-mutagenic")
    
    # Split each class into train/test first
    mut_train, mut_test = train_test_split(mutagenic_df, test_size=0.2, random_state=42)
    non_mut_train, non_mut_test = train_test_split(non_mutagenic_df, test_size=0.2, random_state=42)
    
    # Client 1: 80% mutagenic, 20% non-mutagenic
    client1_mut_size = int(0.8 * len(mut_train))
    client1_non_mut_size = int(0.2 * len(non_mut_train))
    
    client1_mut = mut_train.iloc[:client1_mut_size]
    client1_non_mut = non_mut_train.iloc[:client1_non_mut_size]
    client1_df = pd.concat([client1_mut, client1_non_mut]).sample(frac=1, random_state=42)
    
    # Client 2: 20% mutagenic, 80% non-mutagenic  
    client2_mut_size = int(0.2 * len(mut_train))
    client2_non_mut_size = int(0.8 * len(non_mut_train))
    
    client2_mut = mut_train.iloc[client1_mut_size:client1_mut_size + client2_mut_size]
    client2_non_mut = non_mut_train.iloc[client1_non_mut_size:client1_non_mut_size + client2_non_mut_size]
    client2_df = pd.concat([client2_mut, client2_non_mut]).sample(frac=1, random_state=42)
    
    # Global test set
    test_df = pd.concat([mut_test, non_mut_test]).sample(frac=1, random_state=42)
    
    # Convert to sequences
    client1_sequences = np.array([smiles_to_sequence(s, vocab) for s in client1_df['smiles']])
    client2_sequences = np.array([smiles_to_sequence(s, vocab) for s in client2_df['smiles']])
    test_sequences = np.array([smiles_to_sequence(s, vocab) for s in test_df['smiles']])
    
    print(f"Client 1: {len(client1_df)} samples ({np.mean(client1_df['label']):.1%} mutagenic)")
    print(f"Client 2: {len(client2_df)} samples ({np.mean(client2_df['label']):.1%} mutagenic)")
    print(f"Test: {len(test_df)} samples ({np.mean(test_df['label']):.1%} mutagenic)")
    
    return {
        'client1_sequences': client1_sequences,
        'client1_labels': client1_df['label'].values,
        'client2_sequences': client2_sequences, 
        'client2_labels': client2_df['label'].values,
        'test_sequences': test_sequences,
        'test_labels': test_df['label'].values,
        'vocab': vocab
    }

def save_noniid_data(data):
    """Save non-IID federated learning data."""
    np.savez('data/noniid_split.npz',
             client1_sequences=data['client1_sequences'],
             client1_labels=data['client1_labels'],
             client2_sequences=data['client2_sequences'],
             client2_labels=data['client2_labels'], 
             test_sequences=data['test_sequences'],
             test_labels=data['test_labels'])
    
    with open('data/vocab.json', 'w') as f:
        json.dump(data['vocab'], f)

def main():
    df = process_dataset()
    data = create_noniid_split(df)
    save_noniid_data(data)
    
    print(f"Processed {len(df)} molecules")
    print(f"Client 1: {len(data['client1_labels'])} samples")
    print(f"Client 2: {len(data['client2_labels'])} samples") 
    print(f"Test: {len(data['test_labels'])} samples")
    print(f"SMILES vocab: {len(data['vocab'])} characters")

if __name__ == "__main__":
    main() 