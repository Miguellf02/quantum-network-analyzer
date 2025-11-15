"""
Preprocessing module for Quantum Key Distribution (QKD) datasets.

This script:
- Loads all raw CSVs from data/raw/
- Detects dataset type (QTI or Toshiba)
- Normalizes column names and formats
- Parses datetime columns
- Cleans numeric fields
- Removes outliers (optional)
- Saves a unified cleaned dataset in data/processed/

Author: Miguel López Ferreiro
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------- PATH CONFIG ------------------------------- #

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed/python_preprocessing")
PROCESSED_NAME = "QKD_PROCESSED.csv"


# ------------------------------- LOADING ---------------------------------- #

def load_all_raw():
    """Loads every CSV inside data/raw and returns list of (df, filename)."""
    files = list(RAW_DIR.glob("*.csv"))
    datasets = []

    if not files:
        raise FileNotFoundError("No CSV files found in data/raw/. Add datasets first.")

    for f in files:
        df = pd.read_csv(f)
        datasets.append((df, f.stem))
        print(f"[INFO] Loaded raw file: {f.name}")

    return datasets


# ------------------------------- STANDARDIZATION --------------------------- #

def standardize_column_names(df):
    """Standardizes column names (lowercase, underscores, trimmed)."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    return df


def detect_dataset_type(df):
    """Detects dataset based on its columns."""

    cols = set(df.columns)

    # QTI STRUCTURE (confirmed)
    if {"datetime", "secure_key_rate", "qber"} <= cols:
        return "QTI"

    # TOSHIBA STRUCTURE (confirmed)
    if {"time", "qber", "securekeyratebps"} <= cols:
        return "TOSHIBA"

    raise ValueError(f"Unknown dataset structure.\nColumns found: {cols}")


# ------------------------------- NORMALIZATION ----------------------------- #

def normalize_qti(df):
    print("[INFO] Formatting QTI dataset...")

    df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.rename(columns={
        "secure_key_rate": "skr",
        "channel_loss": "loss",
    })

    # Convert to numeric
    df["skr"] = pd.to_numeric(df["skr"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["qber"] = pd.to_numeric(df["qber"], errors="coerce")

    # QTI SKR is in kbps → convert to bps
    df["skr"] = df["skr"] * 1000  

    return df[["timestamp", "qber", "skr", "loss"]]



def normalize_toshiba(df):
    """
    Normalizes Toshiba dataset to the unified schema.

    Original columns :
    - Time
    - QBER
    - SecureKeyRate(bps)
    """

    print("[INFO] Formatting Toshiba dataset...")

    df = df.rename(columns={
        "time": "timestamp",
        "qber": "qber",
        "securekeyratebps": "skr"
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Toshiba datasets do not provide loss
    df["loss"] = 0.0

    return df[["timestamp", "qber", "skr", "loss"]]


def apply_dataset_specific_cleaning(df, dataset_type):
    """Routes dataframe to the correct normalization function."""
    if dataset_type == "QTI":
        return normalize_qti(df)
    elif dataset_type == "TOSHIBA":
        return normalize_toshiba(df)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# ------------------------------- MERGING ---------------------------------- #

def merge_datasets(dataset_list):
    """Merges all cleaned datasets into a single sorted DataFrame."""
    print("[INFO] Merging cleaned datasets...")
    
    df = pd.concat(dataset_list, ignore_index=True)
    df = df.sort_values("timestamp")
    df = df.dropna(subset=["timestamp"])

    return df



# ------------------------------- SAVE ------------------------------------- #

def save_processed(df):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / PROCESSED_NAME
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved processed dataset: {output_path}")


# ------------------------------- MAIN ------------------------------------- #

def main():
    print("[INFO] Starting preprocessing pipeline...")

    raw_files = load_all_raw()
    cleaned = []

    for df_raw, name in raw_files:
        df = standardize_column_names(df_raw)
        dtype = detect_dataset_type(df)

        print(f"[INFO] Detected dataset type: {dtype} ({name})")

        df_clean = apply_dataset_specific_cleaning(df, dtype)
        cleaned.append(df_clean)

    merged = merge_datasets(cleaned)

    save_processed(merged)

    print("[INFO] Preprocessing completed successfully.")
