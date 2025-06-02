"""Task definition for dermal permeability federated analytics."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set PLOT_DIR to always point to the correct federated_analytics/plots directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# Append plots directly to the federated_analytics path
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")
# Create the plots folder if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)


huskinDB_df = pd.read_csv('federated_analytics/data/HuskinDB_clean.csv')
skinpix_df = pd.read_csv('federated_analytics/data/SkinPiX_clean.csv')
usepa_df = pd.read_csv('federated_analytics/data/NCSU_USEPA_clean.csv')


def add_gaussian_noise_to_datapoints(data, epsilon):
    """Add Gaussian noise to original datapoints before computing histograms."""
    sigma = 1.0 / epsilon
    noise = np.random.normal(0, sigma, size=data.shape)
    noisy_data = data + noise
    return noisy_data

def plot_all_datasets_histogram(huskinDB_df, skinpix_df, usepa_df):
    """Plot all datasets' histograms in a single figure, minimal logic, no bins argument, no dataset dictionary."""
    # Collect all values for bin calculation
    all_values = []
    for df in [huskinDB_df, skinpix_df, usepa_df]:
        for layer in ['Epidermis', 'Dermis']:
            col = f'LogKp {layer} (cm/s)'
            if col in df.columns:
                all_values.append(df[col].dropna().values)
    all_values = np.concatenate(all_values)
    bins = np.linspace(-12, 0, 20)

    fig = go.Figure()
    # huskinDB
    if 'LogKp Epidermis (cm/s)' in huskinDB_df.columns:
        hist, _ = np.histogram(huskinDB_df['LogKp Epidermis (cm/s)'].dropna(), bins=bins)
        fig.add_bar(x=bins[:-1], y=hist, name='huskinDB Epidermis', marker_color='#0061FF', opacity=0.7)
    if 'LogKp Dermis (cm/s)' in huskinDB_df.columns:
        hist, _ = np.histogram(huskinDB_df['LogKp Dermis (cm/s)'].dropna(), bins=bins)
        fig.add_bar(x=bins[:-1], y=hist, name='huskinDB Dermis', marker_color='#8EB3BE', opacity=0.5)
    # SkinPiX
    if 'LogKp Epidermis (cm/s)' in skinpix_df.columns:
        hist, _ = np.histogram(skinpix_df['LogKp Epidermis (cm/s)'].dropna(), bins=bins)
        fig.add_bar(x=bins[:-1], y=hist, name='SkinPiX Epidermis', marker_color='#FF6F00', opacity=0.7)
    if 'LogKp Dermis (cm/s)' in skinpix_df.columns:
        hist, _ = np.histogram(skinpix_df['LogKp Dermis (cm/s)'].dropna(), bins=bins)
        fig.add_bar(x=bins[:-1], y=hist, name='SkinPiX Dermis', marker_color='#FFD180', opacity=0.5)
    # USEPA
    if 'LogKp Epidermis (cm/s)' in usepa_df.columns:
        hist, _ = np.histogram(usepa_df['LogKp Epidermis (cm/s)'].dropna(), bins=bins)
        fig.add_bar(x=bins[:-1], y=hist, name='NCSU & USEPA Epidermis', marker_color='#43A047', opacity=0.7)
    if 'LogKp Dermis (cm/s)' in usepa_df.columns:
        hist, _ = np.histogram(usepa_df['LogKp Dermis (cm/s)'].dropna(), bins=bins)
        fig.add_bar(x=bins[:-1], y=hist, name='NCSU &USEPA Dermis', marker_color='#B2DFDB', opacity=0.5)

    fig.add_vline(x=-3, line_width=5, line_dash="dash", line_color="black", opacity=0.7)
    fig.update_layout(
        title="Distribution of Dermal Permeability Data",
        title_x=0.5,
        font=dict(family="Onest, sans-serif", color="black", size=32),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=800,
        width=1400,
        showlegend=True,
        barmode='overlay',
        bargap=0,
        bargroupgap=0,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=1.3,
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(size=24)
        ),
        margin=dict(t=100, b=80, l=80, r=80)
    )
    fig.update_xaxes(
        title='LogKp (cm/s)',
        title_font=dict(size=28),
        tickfont=dict(size=24),
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=False,
        range=[-12, 0],
    )
    fig.update_yaxes(
        title='',
        title_font=dict(size=28),
        tickfont=dict(size=24),
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=False,
        range=[0, None],
        visible=False
    )
    fig.write_image(os.path.join(PLOT_DIR, 'all_datasets_histogram.png'), scale=3)
    

def plot_individual_histograms_subplots(huskinDB_df, skinpix_df, usepa_df):
    """Plot each dataset's histograms as subplots in a single figure (Epidermis and Dermis overlaid)."""
    # Collect all values for bin calculation
    all_values = []
    for df in [huskinDB_df, skinpix_df, usepa_df]:
        for layer in ['Epidermis', 'Dermis']:
            col = f'LogKp {layer} (cm/s)'
            if col in df.columns:
                all_values.append(df[col].dropna().values)
    all_values = np.concatenate(all_values)
    bins = np.linspace(-12, 0, 20)

    colors = {'Epidermis': '#0061FF', 'Dermis': '#8EB3BE'}
    datasets = [
        ("HuskinDB", huskinDB_df),
        ("SkinPiX", skinpix_df),
        ("NCSU & USEPA", usepa_df)
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["HuskinDB", "SkinPiX", "NCSU & US EPA"],
        horizontal_spacing=0.12
    )

    for idx, (_, df) in enumerate(datasets, 1):
        for layer in ['Epidermis', 'Dermis']:
            col = f'LogKp {layer} (cm/s)'
            if col in df.columns:
                hist, _ = np.histogram(df[col].dropna(), bins=bins)
                fig.add_trace(
                    go.Bar(
                        x=bins[:-1],
                        y=hist,
                        name=layer if idx == 1 else None,  # Show legend only for first subplot
                        marker_color=colors[layer],
                        opacity=0.85 if layer == 'Epidermis' else 0.65,
                        width=(bins[1] - bins[0]),
                        showlegend=(idx == 1)
                    ),
                    row=1, col=idx
                )
        fig.add_vline(x=-3, line_width=5, line_dash="dash", line_color="black", opacity=0.7, row=1, col=idx)

    fig.update_layout(
        font=dict(family="Onest, sans-serif", color="black", size=32),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=700,
        width=2000,
        showlegend=True,
        barmode='overlay',
        bargap=0,
        bargroupgap=0,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(size=36)
        ),
        margin=dict(t=120, b=80, l=80, r=80)
    )

    for i in range(1, 4):
        fig.update_xaxes(
            title_text='LogKp (cm/s)',
            title_font=dict(size=36),
            tickfont=dict(size=32),
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
            range=[-12, 0],
            dtick=2,
            row=1, col=i
        )
        fig.update_yaxes(
            visible=False,
            row=1, col=i
        )

    # Update subplot titles
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=40, color="black", family="Onest, sans-serif")
        ann['y'] = ann['y'] + 0.05

   
    fig.write_image(os.path.join(PLOT_DIR, 'individual_histograms.png'), scale=3)
  

def plot_federated_histogram_comparison(agg_hist, noisy_hist, bins, plot_dir=PLOT_DIR):
    """Plot side-by-side comparison of federated histograms: no noise vs. with noise."""
    colors = {'Epidermis': '#0061FF', 'Dermis': '#8EB3BE'}
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Default Federated Aggregation", "With Gaussian Noise on Datapoints (ε=1)"])
    for i, histograms in enumerate([agg_hist, noisy_hist], 1):
        for layer in ['Epidermis', 'Dermis']:
            fig.add_trace(
                go.Bar(
                    x=bins[:-1],
                    y=histograms[layer],
                    name=layer if i == 1 else None,
                    marker_color=colors[layer],
                    opacity=0.85 if layer == 'Epidermis' else 0.65,
                    width=(bins[1] - bins[0]),
                    showlegend=(i == 1)
                ),
                row=1, col=i
            )
        fig.add_vline(x=-3, line_width=5, line_dash="dash", line_color="black", opacity=0.7, row=1, col=i)

    fig.update_layout(
        font=dict(family="Onest, sans-serif", color="black", size=32),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=700,
        width=2000,
        showlegend=True,
        barmode='overlay',
        bargap=0,
        bargroupgap=0,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(size=36)
        ),
        margin=dict(t=120, b=80, l=80, r=80)
    )

    for i in range(1, 3):
        fig.update_xaxes(
            title_text='LogKp (cm/s)',
            title_font=dict(size=36),
            tickfont=dict(size=32),
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
            range=[-12, 0],
            dtick=2,
            row=1, col=i
        )
        fig.update_yaxes(
            visible=False,
            row=1, col=i
        )

    # Update subplot titles
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=40, color="black", family="Onest, sans-serif")
        ann['y'] = ann['y'] + 0.05

    # Save the plot
    fig.write_image(os.path.join(plot_dir, 'federated_histogram_comparison.png'), scale=3)


if __name__ == "__main__":
    plot_all_datasets_histogram(huskinDB_df, skinpix_df, usepa_df)
    plot_individual_histograms_subplots(huskinDB_df, skinpix_df, usepa_df)
   