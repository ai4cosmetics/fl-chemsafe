"""Create PLOTLY visualization comparing local vs federated model performance."""

import plotly.graph_objects as go
import json
import pandas as pd
from pathlib import Path


def load_existing_results():
    """Load existing performance results."""
    federated_path = Path("results/federated_results.json")
    local_path = Path("results/local_model_results.json")
    
    if not federated_path.exists() or not local_path.exists():
        print("No results found")
        return None
    
    with open(federated_path, 'r') as f:
        federated_results = json.load(f)
    
    with open(local_path, 'r') as f:
        local_results = json.load(f)
    
    return {**local_results, **{k: v for k, v in federated_results.items() if k == "Federated Model"}}


def create_performance_comparison_chart(results):
    """Create performance comparison chart."""
    
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['#0061FF', '#00B8D4', '#00C853', '#FFD600', '#FF6D00']
    
    model_order = ['Federated Model', 'Client B (~20% mut)', 'Client A (~80% mut)']
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
            title=dict(font=dict(color='black', size=28)),
            tickfont=dict(color='black', size=28)
        ),
        yaxis=dict(
            title=dict(font=dict(color='black', size=28)),
            tickfont=dict(color='black', size=28)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5,
            font=dict(color='black', size=28)
        ),
        font=dict(color='black', size=28),
    )
    
    return fig


def create_radar_chart(results):
    """Create a radar chart comparing all models."""
    
    metrics = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['#0061FF', '#00B8D4', '#FF6D00']
    
    fig = go.Figure()
    model_names = list(results.keys())
    
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
                tickfont=dict(color='black', size=16)
            ),
            angularaxis=dict(
                tickfont=dict(color='black', size=16)
            )
        ),
        title=dict(
            text='',
            x=0.5,
            xanchor='center',
            font=dict(color='black')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color='black', size=16)
        ),
        font=dict(color='black', size=16),
    )
    
    return fig


def main():
    """Load existing results and create visualization."""
    
    results = load_existing_results()
    if results is None:
        return
    
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    fig1 = create_performance_comparison_chart(results)
    fig1.write_image(plots_dir / "performance_comparison.png", width=1200, height=1100, scale=2)
    
    fig2 = create_radar_chart(results)
    fig2.write_image(plots_dir / "performance_radar.png", width=700, height=500, scale=2)


if __name__ == "__main__":
    main() 