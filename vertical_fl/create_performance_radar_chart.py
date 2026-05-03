"""Create radar chart for horizontal FL model performance."""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go


def create_radar_chart():
    """Create a radar chart comparing four horizontal FL models."""
    base_dir = Path(__file__).parent
    results_file = base_dir / "results/complete_comparison.csv"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)
    model_order = [
        "Federated Learning",
        "Centralised Learning",
        "SkinDoctorCP Local",
        "AI4Cosmetics Local",
    ]
    available_models = [m for m in model_order if m in df["Model"].values]
    if len(available_models) != 4:
        raise ValueError(f"Expected 4 models, found {available_models}")

    # Match vertical style/semantics: include FNR alongside other metrics.
    # complete_comparison.csv does not contain FNR, so derive it from Recall.
    metrics = ["AUC", "Accuracy", "Precision", "Recall", "F1", "FNR"]
    colors = ["#000000", "#FF9800", "#00ACC1", "#0061FF"]
    dash_styles = ["solid", "dash", "dot", "dashdot"]

    fig = go.Figure()
    for model_name, color, dash_style in zip(available_models, colors, dash_styles):
        row = df[df["Model"] == model_name].iloc[0]
        values = [
            float(row["AUC"]),
            float(row["Accuracy"]),
            float(row["Precision"]),
            float(row["Recall"]),
            float(row["F1"]),
            1.0 - float(row["Recall"]),
        ]
        values += [values[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill="none",
                name=model_name,
                line=dict(color=color, width=5, dash=dash_style),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.5, 1.0],
                ticktext=["0", "0.5", "1.0"],
                tickangle=0,
                tickfont=dict(family="Arial", color="black", size=40),
            ),
            angularaxis=dict(tickfont=dict(family="Arial", color="black", size=60)),
        ),
        title=dict(text="", x=0.5, xanchor="center"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(family="Arial", color="black", size=36),
            traceorder="normal",
        ),
        font=dict(family="Arial", color="black", size=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    output_dir = base_dir / "plots"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "performance_radar_horizontal.png"
    fig.write_image(output_path, width=1600, height=900, scale=2)
    print(f"Saved to {output_path}")
    return fig


if __name__ == "__main__":
    create_radar_chart()
