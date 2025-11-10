"""
Model Performance Comparison Visualisation
Creates horizontal bar chart comparing federated vs baseline models.
"""

import pandas as pd
import plotly.graph_objects as go
import os

def create_model_comparison_plot():
    """Create horizontal bar chart comparing model performance metrics"""
    
    results_file = "results/complete_comparison.csv"
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    df = pd.read_csv(results_file)
    
    # Define metrics to plot (exclude MCC)
    metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['#0061FF', '#00B8D4', '#00C853', '#FFD600', '#FF6D00']
    
    # Define model order
    custom_order = ['Federated Learning', 'Centralised Learning', 'SkinDoctorCP Local', 'AI4Cosmetics Local']
    ordered_df = df.set_index('Model').loc[custom_order].reset_index()
    models = ordered_df['Model'].tolist()
    
    # Create figure
    fig = go.Figure()
    
    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            y=models,
            x=ordered_df[metric],
            name=metric,
            marker_color=colors[i],
            text=ordered_df[metric].round(3),
            textposition='auto',
            orientation='h',
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Federated Learning vs Baseline Models Comparison', #<br><sub>XGBoost Performance on Skin Sensitisation Classification (LLNA)</sub>
            'y': 0.94, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': dict(size=28)
        },
        xaxis_title='Performance Metric Score',
        font=dict(family='Onest, Arial, sans-serif', color="black", size=24),
        yaxis=dict(showgrid=False, zeroline=False, automargin=True, categoryorder='array', 
                  categoryarray=custom_order, title_font=dict(size=24), tickfont=dict(size=24)),
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor='rgba(128,128,128,0.2)', zeroline=False,
                  title_font=dict(size=24), tickfont=dict(size=24)),
        barmode='group', bargap=0.15, bargroupgap=0.1, height=800, width=1250,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=0.35, 
                   bgcolor='white', font=dict(size=24)),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=150, b=80, l=220, r=50)
    )
    
    # Save figure
    os.makedirs("plots", exist_ok=True)
    fig.write_image("plots/model_comparison.png", width=950, height=1000, scale=2)
    
    print("Performance comparison plot saved plots/model_comparison.png")
    return fig

if __name__ == "__main__":
    create_model_comparison_plot() 