"""
UMAP visualisation of chemical space with federated learning predictions.
Shows AI4Cosmetics (green), SkinDoctorCP (blue), and global test set (black) with predictions.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import umap
import xgboost as xgb
from data_preparation import load_data_splits
from rdkit import Chem
from rdkit.Chem import Draw
import base64
from io import BytesIO
import warnings
from pathlib import Path
import pickle
import glob
import os
from molvs import standardize_smiles
warnings.filterwarnings('ignore')

def mol_to_base64_image(smiles, size=(100, 100)):
    """Convert SMILES to base64 encoded molecular structure image"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            img = Draw.MolToImage(mol, size=size)
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    except:
        pass
    return None

def create_chemical_metadata():
    """Create metadata mapping for chemicals"""
    # Load original CSVs directly
    ai4_df = pd.read_csv('data/ai4cosmetics_data.csv')
    skin_df = pd.read_csv('data/wilm_et_al_skindoctorcp_data.csv')
    
    # Create metadata with real names and CASRN
    ai4_metadata = {}
    for _, row in ai4_df.iterrows():
        try:
            std_smiles = standardize_smiles(row['Canonical SMILES'])
            ai4_metadata[std_smiles] = {
                'name': row['Name'],
                'casrn': row['CASRN'],
                'source': 'AI4Cosmetics',
                'smiles': std_smiles
            }
        except:
            pass
    
    skin_metadata = {}
    for _, row in skin_df.iterrows():
        try:
            std_smiles = standardize_smiles(row['SMILES'])
            skin_metadata[std_smiles] = {
                'name': row['Names'], 
                'casrn': row['CASRN'],
                'source': 'SkinDoctorCP',
                'smiles': std_smiles
            }
        except:
            pass
    
    # Save metadata
    os.makedirs('data', exist_ok=True)
    with open('data/chemical_metadata.pkl', 'wb') as f:
        pickle.dump({'ai4cosmetics': ai4_metadata, 'skindoctorcp': skin_metadata}, f)
    
    print("Chemical metadata saved to data/chemical_metadata.pkl")

def load_chemical_metadata():
    """Load chemical metadata"""
    if not os.path.exists('data/chemical_metadata.pkl'):
        print("Creating chemical metadata...")
        create_chemical_metadata()
    
    with open('data/chemical_metadata.pkl', 'rb') as f:
        return pickle.load(f)

def smiles_to_features_map(data_splits):
    """Create mapping from standardised SMILES to feature index using cached data"""
    # Build mappings from cached splits (no reprocessing needed!)
    feature_to_smiles = {}
    test_to_smiles = {}
    
    # Training data mapping (combined order: ai4 first, then skin)
    feature_idx = 0
    for smiles in data_splits['ai4cosmetics']['train'][2]:  # SMILES are at index 2
        feature_to_smiles[feature_idx] = smiles
        feature_idx += 1
    for smiles in data_splits['skindoctorcp']['train'][2]:
        feature_to_smiles[feature_idx] = smiles
        feature_idx += 1
    
    # Test data mapping (combined order: ai4 first, then skin)
    test_feature_idx = 0
    for smiles in data_splits['ai4cosmetics']['test'][2]:
        test_to_smiles[test_feature_idx] = smiles
        test_feature_idx += 1
    for smiles in data_splits['skindoctorcp']['test'][2]:
        test_to_smiles[test_feature_idx] = smiles
        test_feature_idx += 1
    
    return feature_to_smiles, test_to_smiles

def load_models_and_data():
    """Load federated/centralised models and data splits"""
    data_splits = load_data_splits()
    
    # Load centralised model
    cent_files = glob.glob("models/*centralised*.pkl") + glob.glob("models/*combined*.pkl")
    centralised_model = None
    if cent_files:
        with open(cent_files[0], 'rb') as f:
            centralised_model = pickle.load(f)
    
    # Load federated model
    federated_model = None
    if os.path.exists('models/global_federated_model.json'):
        federated_model = xgb.Booster()
        federated_model.load_model('models/global_federated_model.json')
    
    return data_splits, centralised_model, federated_model

def prepare_hover_text(smiles, meta_dict, llna_label, predictions=None):
    """Create hover text for a single data point"""
    meta = meta_dict.get(smiles, {})
    name = meta.get('name', 'Unknown')
    casrn = meta.get('casrn', 'Unknown')
    
    # Base hover text without molecular structure
    hover_text = f"<b>{name}</b><br>CASRN: {casrn}<br>SMILES: {smiles}<br>True LLNA: {llna_label}"
    
    # Add predictions if provided
    if predictions:
        cent_proba, fed_proba = predictions
        cent_pred_label = "Positive" if cent_proba > 0.5 else "Negative"
        fed_pred_label = "Positive" if fed_proba > 0.5 else "Negative"
        hover_text += f"<br><b>Centralised Pred:</b> {cent_pred_label} ({cent_proba:.3f})"
        hover_text += f"<br><b>Federated Pred:</b> {fed_pred_label} ({fed_proba:.3f})"
    
    return hover_text

def create_training_traces(train_embedding, y_train, ai4_train_size, train_to_smiles, metadata):
    """Create scatter traces for training data"""
    traces = []
    
    # AI4Cosmetics training data
    ai4_embed = train_embedding[:ai4_train_size]
    ai4_labels = y_train[:ai4_train_size]
    ai4_hover_text = []
    
    for i in range(ai4_train_size):
        llna_label = "Positive" if ai4_labels[i] == 1 else "Negative"
        if i in train_to_smiles:
            smiles = train_to_smiles[i]
            hover_text = prepare_hover_text(smiles, metadata['ai4cosmetics'], llna_label)
        else:
            hover_text = f"True LLNA: {llna_label}"
        ai4_hover_text.append(hover_text)
    
    traces.append(go.Scatter(
        x=ai4_embed[:, 0], y=ai4_embed[:, 1],
        mode='markers',
        marker=dict(color='#ff69b4', size=12, opacity=0.8, line=dict(width=1, color='white')),
        name='AI4Cosmetics Training',
        text=ai4_hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # SkinDoctorCP training data
    skin_embed = train_embedding[ai4_train_size:]
    skin_labels = y_train[ai4_train_size:]
    skin_hover_text = []
    
    for i in range(len(skin_embed)):
        llna_label = "Positive" if skin_labels[i] == 1 else "Negative"
        train_idx = ai4_train_size + i
        if train_idx in train_to_smiles:
            smiles = train_to_smiles[train_idx]
            hover_text = prepare_hover_text(smiles, metadata['skindoctorcp'], llna_label)
        else:
            hover_text = f"True LLNA: {llna_label}"
        skin_hover_text.append(hover_text)
    
    traces.append(go.Scatter(
        x=skin_embed[:, 0], y=skin_embed[:, 1],
        mode='markers',
        marker=dict(color='#037bff', size=12, opacity=0.8, line=dict(width=1, color='white')),
        name='SkinDoctorCP Training',
        text=skin_hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    return traces

def create_test_trace(test_embedding, y_test, X_test, test_to_smiles, metadata, centralised_model, federated_model):
    """Create scatter trace for test data with predictions"""
    if not (centralised_model and federated_model):
        return None
    
    # Get predictions
    cent_proba = centralised_model.predict_proba(X_test)[:, 1]
    dtest = xgb.DMatrix(X_test)
    fed_proba = federated_model.predict(dtest)
    
    test_hover_text = []
    for i in range(len(test_embedding)):
        true_llna_label = "Positive" if y_test[i] == 1 else "Negative"
        
        if i in test_to_smiles:
            smiles = test_to_smiles[i]
            # Find metadata from either source
            if smiles in metadata['ai4cosmetics']:
                meta_dict = metadata['ai4cosmetics']
            elif smiles in metadata['skindoctorcp']:
                meta_dict = metadata['skindoctorcp']
            else:
                meta_dict = {}
            
            hover_text = prepare_hover_text(smiles, meta_dict, true_llna_label, 
                                          (cent_proba[i], fed_proba[i]))
        else:
            hover_text = (f"True LLNA: {true_llna_label}<br>"
                         f"Centralised Pred: {cent_proba[i]:.3f}<br>"
                         f"Federated Pred: {fed_proba[i]:.3f}")
        test_hover_text.append(hover_text)
    
    return go.Scatter(
        x=test_embedding[:, 0], y=test_embedding[:, 1],
        mode='markers',
        marker=dict(color='black', size=12, opacity=0.9, line=dict(width=1, color='white')),
        name='Global Test Set',
        text=test_hover_text,
        hovertemplate='%{text}<extra></extra>'
    )

def create_umap_visualisation():
    """Create UMAP visualisation with training data and test predictions"""
    # Load data and models
    data_splits, centralised_model, federated_model = load_models_and_data()
    metadata = load_chemical_metadata()
    train_to_smiles, test_to_smiles = smiles_to_features_map(data_splits)
    
    # Get training and test data
    X_train = data_splits['combined_train'][0]
    y_train = data_splits['combined_train'][1]
    X_test = data_splits['global_test'][0] 
    y_test = data_splits['global_test'][1]
    
    # Fit UMAP
    X_combined = np.vstack([X_train, X_test])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_combined)
    
    # Split embeddings
    train_embedding = embedding[:len(X_train)]
    test_embedding = embedding[len(X_train):]
    
    # Create figure
    fig = go.Figure()
    
    # Add training traces
    ai4_train_size = len(data_splits['ai4cosmetics']['train'][0])
    training_traces = create_training_traces(train_embedding, y_train, ai4_train_size, 
                                           train_to_smiles, metadata)
    for trace in training_traces:
        fig.add_trace(trace)
    
    # Add test trace
    test_trace = create_test_trace(test_embedding, y_test, X_test, test_to_smiles, 
                                  metadata, centralised_model, federated_model)
    if test_trace:
        fig.add_trace(test_trace)
    
    # Update layout
    fig.update_layout(
        #title={
        #    'text': 'Chemical Space Visualisation: Training Data and Test Predictions',
        #    'y': 0.95,
        #    'x': 0.5,
        #    'xanchor': 'center',
        #    'font': {'size': 28, 'color': 'black'}
        #},
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        hovermode='closest',
        width=1400,
        height=1000,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font_size=24
        ),
        font=dict(family='Arial', size=24, color='black')
    )
    
    # Save plot
    os.makedirs('plots', exist_ok=True)
    fig.write_html('plots/chemical_space_umap.html')
    fig.write_image('plots/chemical_space_umap.png', scale=2)
    print("UMAP visualisation saved to plots/chemical_space_umap.html and plots/chemical_space_umap.png")

if __name__ == "__main__":
    create_umap_visualisation() 