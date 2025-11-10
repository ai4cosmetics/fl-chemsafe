"""Create PLOTLY visualization comparing local vs federated model performance."""

import plotly.graph_objects as go
import json
import pandas as pd
import numpy as np
from pathlib import Path


def load_existing_results():
    """Load existing performance results."""
    base_dir = Path(__file__).parent
    federated_path = base_dir / "results/federated_results.json"
    local_path = base_dir / "results/local_model_results.json"
    
    if not federated_path.exists() or not local_path.exists():
        print("No results found")
        return None
    
    with open(federated_path, 'r') as f:
        federated_results = json.load(f)
    
    with open(local_path, 'r') as f:
        local_results = json.load(f)
    
    combined_results = {**local_results, **{k: v for k, v in federated_results.items() if k == "Federated Model"}}

    rename_map = {
        "Client A (~80% mut)": "Organisation A (~80% mut)",
        "Client B (~20% mut)": "Organisation B (~20% mut)",
    }

    for old_name, new_name in rename_map.items():
        if old_name in combined_results:
            combined_results[new_name] = combined_results.pop(old_name)

    return combined_results


def create_performance_comparison_chart(results):
    """Create performance comparison chart."""
    
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1', 'fnr']
    metric_labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'FNR']
    colors = ['#0061FF', '#00B8D4', '#00C853', '#FFD600', '#FF6D00', '#D32F2F']
    
    model_order = ['Federated Model', 'Organisation A (~80% mut)', 'Organisation B (~20% mut)']
    data = []
    
    for model in model_order:
        if model in results:
            for metric, label in zip(metrics, metric_labels):
                data.append({
                    'Model': model,
                    'Metric': label,
                    'Value': results[model][metric],
                    'Color': colors[metrics.index(metric)]
                })
    
    df = pd.DataFrame(data)
    fig = go.Figure()
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        metric_data = df[df['Metric'] == label]
        
        fig.add_trace(go.Bar(
            name=label,
            y=metric_data['Model'],
            x=metric_data['Value'],
            marker_color=color,
            text=[f'{val:.3f}' for val in metric_data['Value']],
            textposition='auto',
            #textfont=dict(color='black', size=24),
            orientation='h'
        ))
    
    fig.update_layout(
        title=dict(
            text='',
            x=0.5,
            xanchor='center'
        ),
        yaxis_title='',
        xaxis_title='Performance Score',
        xaxis=dict(
            range=[0, 1.1],
            title=dict(font=dict(family='Arial', color='black', size=30)),
            tickfont=dict(family='Arial', color='black', size=30)
        ),
        yaxis=dict(
            title=dict(font=dict(family='Arial', color='black', size=30)),
            tickfont=dict(family='Arial', color='black', size=30)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,
            font=dict(family='Arial', color='black', size=38)
        ),
        font=dict(family='Arial', color='black', size=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_radar_chart(results):
    """Create a radar chart comparing all models."""
    
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1', 'fnr']
    metric_labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'FNR']
    colors = ['#9C27B0', '#FF9800', '#00ACC1']
    
    fig = go.Figure()
    model_order = ['Federated Model', 'Organisation A (~80% mut)', 'Organisation B (~20% mut)']
    model_names = [model for model in model_order if model in results]

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        values = [results[model_name][metric] for metric in metrics]
        values += [values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill='toself',
            name=model_name,
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.3
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 1],
                tickvals=[0, 0.5, 1.0],
                ticktext=['0', '0.5', '1.0'],
                tickangle=0,
                tickfont=dict(family='Arial', color='black', size=28)
            ),
            angularaxis=dict(
                tickfont=dict(family='Arial', color='black', size=32)
            )
        ),
        title=dict(
            text='',
            x=0.5,
            xanchor='center',
            font=dict(family='Arial', color='black', size=40)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(family='Arial', color='black', size=36),
            traceorder='normal'
        ),
        font=dict(family='Arial', color='black', size=28),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_confusion_matrix(model_name, metrics, test_labels):
    """Create confusion matrix for a single model."""
    total_positives = int(np.sum(test_labels == 1))
    total_negatives = int(np.sum(test_labels == 0))
    
    recall = metrics['recall']
    precision = metrics['precision']
    
    # Calculate confusion matrix values
    tp = int(recall * total_positives)
    fn = total_positives - tp
    
    if precision > 0:
        fp = int(tp / precision - tp)
    else:
        fp = total_negatives
    
    tn = total_negatives - fp
    
    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    
    # Create annotations
    annotations = []
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{labels[i][j]}: {confusion_matrix[i, j]}",
                    font=dict(family='Arial', color='white' if confusion_matrix[i, j] > 400 else 'black', size=30),
                    showarrow=False
                )
            )
    
    fig = go.Figure(go.Heatmap(
        z=confusion_matrix,
        x=['Predicted Non-mutagen', 'Predicted Mutagen'],
        y=['Actual Non-mutagen', 'Actual Mutagen'],
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title=dict(text=model_name, x=0.5, xanchor='center', font=dict(family='Arial', size=50)),
        xaxis=dict(side='bottom', tickfont=dict(family='Arial', size=30)),
        yaxis=dict(tickfont=dict(family='Arial', size=30), autorange='reversed'),
        annotations=annotations,
        font=dict(family='Arial', size=30),
        paper_bgcolor='rgba(0,0,0,0)',
        height=800,
        width=900
    )
    
    return fig


def main():
    """Load existing results and create visualization."""
    
    results = load_existing_results()
    if results is None:
        return
    
    plots_dir = Path(__file__).parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    fig1 = create_performance_comparison_chart(results)
    fig1.write_image(plots_dir / "performance_comparison.png", width=1200, height=1200, scale=2)
    
    fig2 = create_radar_chart(results)
    fig2.write_image(plots_dir / "performance_radar.png", width=1600, height=900, scale=2)
    
    # Create confusion matrices
    base_dir = Path(__file__).parent
    data = np.load(base_dir / "data/noniid_split.npz")
    test_labels = data['test_labels']
    
    for model_name, metrics in results.items():
        fig = create_confusion_matrix(model_name, metrics, test_labels)
        filename = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('~', '')
        fig.write_image(plots_dir / f"confusion_matrix_{filename}.png", width=900, height=800, scale=2)
    
    print(f"Saved to {plots_dir}/")


if __name__ == "__main__":
    main() 