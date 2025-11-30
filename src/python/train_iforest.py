"""
Evaluation Module for Isolation Forest Model on QKD Operational Data
====================================================================

This script performs a full diagnostic analysis of the Isolation Forest model
trained over engineered QKD features.

It generates:
    - Time-block anomaly plots
    - QBER vs anomaly score comparison
    - SKR vs anomaly score comparison
    - Histogram of anomaly scores
    - Ranking of top anomalies
    - Correlation heatmap (matplotlib only, no seaborn)

Author:
    Miguel López Ferreiro
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from ..constants import constants


# ===============================================================
# CONFIG PATHS
# ===============================================================

SCORES_PATH = constants.MODEL_OUTPUT_DIR / "iforest_scores.csv"
OUTPUT_DIR  = constants.MODEL_OUTPUT_DIR / "evaluation"

PLOT_BLOCKS_PATH     = OUTPUT_DIR / "anomalies_by_blocks.png"
PLOT_QBER_PATH       = OUTPUT_DIR / "qber_vs_anomaly.png"
PLOT_SKR_PATH        = OUTPUT_DIR / "skr_vs_anomaly.png"
PLOT_HIST_PATH       = OUTPUT_DIR / "anomaly_histogram.png"
PLOT_HEATMAP_PATH    = OUTPUT_DIR / "feature_correlations.png"
TOP_ANOMALIES_PATH   = OUTPUT_DIR / "top_anomalies.csv"



# ===============================================================
# LOAD PROCESSED SCORES
# ===============================================================

def load_scores():
    """Loads the CSV produced by train_iforest.py."""

    if not SCORES_PATH.exists():
        raise FileNotFoundError(
            f"Isolation Forest score file not found at {SCORES_PATH}.\n"
            "Run train_iforest.py first."
        )

    print(f"[INFO] Loading scores from: {SCORES_PATH}")

    df = pd.read_csv(SCORES_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return df



# ===============================================================
# TIME-BLOCK PLOT
# ===============================================================

def plot_anomalies_by_time_blocks(df):

    print("[INFO] Plotting anomaly scores by temporal blocks...")

    df = df.copy()
    df["time_diff"] = df["timestamp"].diff().dt.days

    block_starts = df[df["time_diff"] > 5].index.tolist()
    block_starts = [0] + block_starts + [len(df)]

    plt.figure(figsize=(16, 6))

    for i in range(len(block_starts) - 1):
        start = block_starts[i]
        end = block_starts[i + 1]
        block = df.iloc[start:end]

        plt.plot(
            block["timestamp"],
            block["anomaly_score"],
            label=f"Block {i+1}",
            linewidth=1.2
        )

        anomalies = block[block["is_anomaly"] == 1]
        plt.scatter(
            anomalies["timestamp"],
            anomalies["anomaly_score"],
            color="red",
            s=16
        )

    plt.xlabel("Timestamp")
    plt.ylabel("Anomaly Score")
    plt.title("Isolation Forest — Anomaly Detection (Time Blocks)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_BLOCKS_PATH)
    plt.close()

    print(f"[INFO] Saved → {PLOT_BLOCKS_PATH}")



# ===============================================================
# QBER VS ANOMALY
# ===============================================================

def plot_qber_vs_anomaly(df):

    print("[INFO] Plotting QBER vs anomaly score...")

    plt.figure(figsize=(12, 5))
    plt.scatter(df["qber"], df["anomaly_score"], s=12, alpha=0.4)
    plt.xlabel("QBER")
    plt.ylabel("Anomaly Score")
    plt.title("QBER vs Anomaly Score")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(PLOT_QBER_PATH)
    plt.close()

    print(f"[INFO] Saved → {PLOT_QBER_PATH}")



# ===============================================================
# SKR VS ANOMALY
# ===============================================================

def plot_skr_vs_anomaly(df):

    print("[INFO] Plotting SKR vs anomaly score...")

    plt.figure(figsize=(12, 5))
    plt.scatter(df["skr"], df["anomaly_score"], s=12, alpha=0.4)
    plt.xlabel("SKR (bits/s)")
    plt.ylabel("Anomaly Score")
    plt.title("SKR vs Anomaly Score")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(PLOT_SKR_PATH)
    plt.close()

    print(f"[INFO] Saved → {PLOT_SKR_PATH}")



# ===============================================================
# HISTOGRAM OF ANOMALY SCORES
# ===============================================================

def plot_anomaly_histogram(df):

    print("[INFO] Plotting anomaly score histogram...")

    plt.figure(figsize=(10, 5))
    plt.hist(df["anomaly_score"], bins=40, color="skyblue", edgecolor="black")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Anomaly Scores")
    plt.tight_layout()

    plt.savefig(PLOT_HIST_PATH)
    plt.close()

    print(f"[INFO] Saved → {PLOT_HIST_PATH}")



# ===============================================================
# CORRELATION HEATMAP (WITHOUT SEABORN)
# ===============================================================

def plot_feature_correlation(df):

    print("[INFO] Computing feature correlation heatmap...")

    feature_cols = [c for c in df.columns if c.startswith("feat_")]

    corr = df[feature_cols].corr().values

    plt.figure(figsize=(14, 12))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.colorbar()

    plt.title("Correlation Heatmap of Engineered Features")
    plt.tight_layout()

    plt.savefig(PLOT_HEATMAP_PATH)
    plt.close()

    print(f"[INFO] Saved → {PLOT_HEATMAP_PATH}")



# ===============================================================
# TOP ANOMALIES TABLE
# ===============================================================

def export_top_anomalies(df, n=20):

    print(f"[INFO] Exporting top {n} anomalies...")

    df_sorted = df.sort_values("anomaly_score", ascending=False)
    df_top = df_sorted.head(n)

    df_top.to_csv(TOP_ANOMALIES_PATH, index=False)

    print(f"[INFO] Saved → {TOP_ANOMALIES_PATH}")



# ===============================================================
# MAIN
# ===============================================================

def main():

    print("\n[INFO] Starting evaluation of Isolation Forest...\n")

    df = load_scores()

    # Standard evaluation plots
    plot_anomalies_by_time_blocks(df)
    plot_qber_vs_anomaly(df)
    plot_skr_vs_anomaly(df)
    plot_anomaly_histogram(df)
    plot_feature_correlation(df)

    # Top anomalies
    export_top_anomalies(df, n=30)

    print("\n[INFO] Evaluation completed successfully.\n")



if __name__ == "__main__":
    main()
