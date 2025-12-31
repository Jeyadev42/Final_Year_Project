# ============================================================
#  FULL UPGRADED VISUALIZATION PIPELINE (OPTION D)
# ============================================================

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

sns.set(style="whitegrid")


OUTPUT_PATH = "results/eval_pipeline_output.jsonl"
SUMMARY_PATH = "results/eval_summary.json"


# ============================================================
# 1. LOAD RESULTS
# ============================================================
def load_results():
    if not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(f"{OUTPUT_PATH} not found.")

    rows = []
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line.strip())
            m = rec["metrics"]

            rows.append({
                "id": rec["id"],
                "query": rec["query"],
                "relevance": m["relevance"],
                "grounding": m["grounding"]["score"],
                "completeness": m["completeness"]["score"],
                "contradiction": m["contradiction"]["score"],
                "reliability": m["reliability"]["score"],
                "overall": m["overall_score"]
            })

    return pd.DataFrame(rows)



# ============================================================
# 2. HELPER VISUALIZATION FUNCTIONS
# ============================================================

def plot_value_colored_bars(df, metric, title, filename):
    """
    Bars are colored based on metric value (not query index).
    """
    os.makedirs("visualizations", exist_ok=True)

    values = df[metric].values
    norm = (values - values.min()) / (values.max() - values.min() + 1e-9)
    colors = plt.cm.Blues(norm)

    plt.figure(figsize=(14, 5))
    plt.bar(df["id"], values, color=colors)
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel("Query ID")
    plt.ylabel(metric.capitalize())

    # Add threshold lines for interpretation
    if metric in ["relevance", "grounding", "reliability"]:
        plt.axhline(0.7, linestyle="--", color="red", alpha=0.5, label="Target threshold")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}")
    plt.close()



def plot_sorted(df, metric, filename):
    """
    Sorted ranking chart (best → worst).
    """
    sorted_df = df.sort_values(metric, ascending=False)
    values = sorted_df[metric].values
    norm = (values - values.min()) / (values.max() - values.min() + 1e-9)
    colors = plt.cm.Greens(norm)

    plt.figure(figsize=(12, 5))
    plt.bar(sorted_df["id"], values, color=colors)
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.title(f"Ranking of Queries by {metric.capitalize()}")
    plt.xlabel("Query ID")
    plt.ylabel(metric.capitalize())

    plt.tight_layout()
    plt.savefig(f"visualizations/{metric}_ranking.png")
    plt.close()



def plot_distribution(df, metric, filename):
    """
    Histogram + KDE for distribution.
    """
    plt.figure(figsize=(7, 4))
    sns.histplot(df[metric], bins=10, kde=True, color="blue", alpha=0.6)
    plt.title(f"Distribution of {metric.capitalize()}")
    plt.xlabel(metric.capitalize())
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}")
    plt.close()



def plot_trend(df, metric, filename):
    """
    Scatter + trendline.
    """
    plt.figure(figsize=(10, 4))
    sns.regplot(x=np.arange(len(df)), y=df[metric], scatter_kws={"s": 40})
    plt.ylim(0, 1)
    plt.title(f"Trend for {metric.capitalize()}")
    plt.xlabel("Query index")
    plt.ylabel(metric.capitalize())
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}")
    plt.close()



def plot_correlation(df):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.drop(columns=["id", "query"]).corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Metric Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("visualizations/correlation_heatmap.png")
    plt.close()



def radar_chart(df, query_id, filename):
    """
    Radar chart for a single query.
    """
    row = df[df["id"] == query_id].iloc[0]
    metrics = ["relevance", "grounding", "completeness", "reliability", "overall"]
    values = [row[m] for m in metrics]
    values += values[:1]

    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], metrics)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.3)

    plt.title(f"Radar Chart – {query_id}")
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}")
    plt.close()



# ============================================================
# 3. EXECUTIVE SUMMARY
# ============================================================

def save_summary(df):
    summary = {
        "top_5_queries": df.sort_values("overall", ascending=False).head(5)[["id", "overall"]].to_dict(orient="records"),
        "bottom_5_queries": df.sort_values("overall", ascending=True).head(5)[["id", "overall"]].to_dict(orient="records"),
        "contradiction_cases": df[df["contradiction"] > 0.5][["id", "contradiction"]].to_dict(orient="records"),
        "low_grounding_cases": df[df["grounding"] < 0.6][["id", "grounding"]].to_dict(orient="records"),
    }

    os.makedirs("visualizations", exist_ok=True)
    with open("visualizations/executive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)



# ============================================================
# 4. MAIN VISUALIZATION PIPELINE
# ============================================================

def visualize_all(df):
    os.makedirs("visualizations", exist_ok=True)

    metrics = ["relevance", "grounding", "completeness",
               "reliability", "contradiction", "overall"]

    # Generate charts for each metric
    for m in metrics:
        plot_value_colored_bars(df, m, f"{m.capitalize()} Score per Query", f"{m}_per_query.png")
        plot_sorted(df, m, f"{m}_ranking.png")
        plot_distribution(df, m, f"{m}_distribution.png")
        plot_trend(df, m, f"{m}_trend.png")

    # Radar charts for first 3 queries
    for q in df["id"].head(3):
        radar_chart(df, q, f"radar_{q}.png")

    plot_correlation(df)
    save_summary(df)

    print("\nVisualizations saved to ./visualizations/")


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":
    df = load_results()
    visualize_all(df)
    print("\nDone.")
