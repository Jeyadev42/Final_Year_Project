# ============================================================
# STREAMLIT EVALUATION DASHBOARD
# ============================================================

import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

sns.set(style="whitegrid")

OUTPUT_PATH = "results/eval_pipeline_output.jsonl"


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():
    if not os.path.exists(OUTPUT_PATH):
        st.error(f"File not found: {OUTPUT_PATH}")
        st.stop()

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
                "overall": m["overall_score"],
            })
    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def plot_value_colored_bars(df, metric):
    values = df[metric].values
    norm = (values - values.min()) / (values.max() - values.min() + 1e-9)
    colors = plt.cm.Blues(norm)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df["id"], values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric.capitalize()} Score per Query")
    ax.set_ylabel(metric.capitalize())
    ax.tick_params(axis="x", rotation=90)

    if metric in ["relevance", "grounding", "reliability"]:
        ax.axhline(0.7, linestyle="--", color="red", alpha=0.5, label="Target threshold")
        ax.legend()

    return fig


def plot_distribution(df, metric):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[metric], bins=10, kde=True, color="teal", ax=ax)
    ax.set_title(f"{metric.capitalize()} Distribution")
    ax.set_xlabel(metric.capitalize())
    return fig


def plot_ranking(df, metric):
    sorted_df = df.sort_values(metric, ascending=False)
    values = sorted_df[metric].values
    norm = (values - values.min()) / (values.max() - values.min() + 1e-9)
    colors = plt.cm.Greens(norm)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(sorted_df["id"], values, color=colors)
    ax.set_title(f"Ranking by {metric.capitalize()}")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=90)
    return fig


def plot_trend(df, metric):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.regplot(x=np.arange(len(df)), y=df[metric], scatter_kws={"s": 40}, ax=ax)
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"{metric.capitalize()} Trend")
    ax.set_ylim(0, 1)
    return fig


def radar_chart(df, query_id):
    row = df[df["id"] == query_id].iloc[0]
    metrics = ["relevance", "grounding", "completeness", "reliability", "overall"]
    values = [row[m] for m in metrics]
    values += values[:1]

    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], metrics)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.4)
    plt.title(f"Radar Chart – {query_id}")

    return fig



# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Evaluation Dashboard", layout="wide")

st.title("Evaluation Dashboard – Web-Augmented AI System")

df = load_results()

st.subheader("Overview Statistics")
st.dataframe(df)

col1, col2 = st.columns(2)
with col1:
    st.metric("Avg Overall Score", f"{df['overall'].mean():.3f}")
with col2:
    st.metric("Avg Relevance Score", f"{df['relevance'].mean():.3f}")


# ============================================================
# EXECUTIVE SUMMARY PANEL
# ============================================================

st.markdown("---")
st.header("Executive Summary")

summary = {
    "Top 5 High-Scoring Queries": df.sort_values("overall", ascending=False).head(5)[["id", "overall"]].to_dict("records"),
    "Bottom 5 Low-Scoring Queries": df.sort_values("overall", ascending=True).head(5)[["id", "overall"]].to_dict("records"),
    "Contradiction Cases": df[df["contradiction"] > 0.5][["id", "contradiction"]].to_dict("records"),
    "Low Grounding Cases": df[df["grounding"] < 0.6][["id", "grounding"]].to_dict("records"),
}

st.json(summary)


# ============================================================
# METRIC SELECTOR
# ============================================================

st.markdown("---")
st.header("Metric Visualization")

metric = st.selectbox(
    "Choose Metric:",
    ["relevance", "grounding", "completeness", "reliability", "contradiction", "overall"],
)

colA, colB = st.columns(2)

with colA:
    st.pyplot(plot_value_colored_bars(df, metric))
    st.pyplot(plot_trend(df, metric))

with colB:
    st.pyplot(plot_distribution(df, metric))
    st.pyplot(plot_ranking(df, metric))


# ============================================================
# RADAR CHART
# ============================================================

st.markdown("---")
st.header("Radar Plot for Query")

selected_query = st.selectbox("Select a Query ID:", df["id"].tolist())
st.pyplot(radar_chart(df, selected_query))


# ============================================================
# DOWNLOAD OPTIONS
# ============================================================

st.markdown("---")
st.header("Downloads")

st.download_button(
    label="Download Evaluation Table (CSV)",
    data=df.to_csv(index=False),
    file_name="evaluation_table.csv",
    mime="text/csv",
)

st.success("Dashboard Ready.")



