"""
Preprocessing module for Quantum Key Distribution (QKD) datasets.

This script:
- Loads all raw CSVs from data/raw/
- Detects dataset type (QTI, Toshiba, or Unknown)
- Normalizes column names and formats
- Adds a 'source' column indicating the origin file
- Parses datetime columns
- Cleans numeric fields
- Saves a unified cleaned dataset in data/processed/

Author: Miguel López Ferreiro
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ..constants import constants


# ------------------------------
#  LOADING
# ------------------------------

def load_all_raw():
    """Loads only QKD CSVs from data/raw/qkd and returns list of (df, filename)."""
    
    qkd_path = constants.RAW_DIR / "qkd"
    files = list(qkd_path.glob("*.csv"))
    datasets = []

    if not files:
        raise FileNotFoundError(f"No CSV files found in {qkd_path}. Add QKD datasets first.")

    for f in files:
        df = pd.read_csv(f)
        datasets.append((df, f.stem))
        print(f"[INFO] Loaded QKD raw file: {f.name}")

    return datasets


# ------------------------------
#  STANDARDIZATION
# ------------------------------

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

    # QTI STRUCTURE
    if {"datetime", "secure_key_rate", "qber"} <= cols:
        return "QTI"

    # TOSHIBA STRUCTURE
    if {"time", "qber", "securekeyratebps"} <= cols:
        return "TOSHIBA"

    return "UNKNOWN"


# ------------------------------
#  NORMALIZATION FUNCTIONS
# ------------------------------

def normalize_qti(df, source_name):
    """Normalizes QTI dataset to unified schema."""
    
    print("[INFO] Formatting QTI dataset...")

    df["timestamp"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.rename(columns={
        "secure_key_rate": "skr",
        "channel_loss": "loss",
    })

    df["skr"] = pd.to_numeric(df["skr"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["qber"] = pd.to_numeric(df["qber"], errors="coerce")

    df["source"] = source_name.upper()

    return df[["timestamp", "qber", "skr", "loss", "source"]]


def normalize_toshiba(df, source_name):
    """Normalizes Toshiba dataset to unified schema."""
    
    print("[INFO] Formatting Toshiba dataset...")

    df = df.rename(columns={
        "time": "timestamp",
        "securekeyratebps": "skr"
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["qber"] = pd.to_numeric(df["qber"], errors="coerce")
    df["skr"] = pd.to_numeric(df["skr"], errors="coerce")

    df["loss"] = np.nan
    df["source"] = source_name.upper()

    return df[["timestamp", "qber", "skr", "loss", "source"]]


def normalize_unknown(df, source_name):
    """Attempts best-effort normalization for unknown datasets."""
    
    print(f"[WARNING] Unknown dataset structure → Using fallback normalization for {source_name}")

    # Try to detect timestamp-like column
    timestamp_col = None
    for c in df.columns:
        if "time" in c or "date" in c:
            timestamp_col = c
            break

    if timestamp_col is None:
        raise ValueError(f"Unknown dataset {source_name} does not contain any time-like column.")

    df["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Try to detect qber or skr-like values
    df["qber"] = df.filter(regex="qber|error", axis=1).iloc[:, 0] if df.filter(regex="qber|error", axis=1).shape[1] > 0 else np.nan
    df["skr"] = df.filter(regex="skr|key", axis=1).iloc[:, 0] if df.filter(regex="skr|key", axis=1).shape[1] > 0 else np.nan

    df["qber"] = pd.to_numeric(df["qber"], errors="coerce")
    df["skr"] = pd.to_numeric(df["skr"], errors="coerce")

    df["loss"] = np.nan
    df["source"] = "UNKNOWN_" + source_name.upper()

    return df[["timestamp", "qber", "skr", "loss", "source"]]


# ------------------------------
#  ROUTER
# ------------------------------

def apply_dataset_specific_cleaning(df, dataset_type, source_name):

    if dataset_type == "QTI":
        return normalize_qti(df, source_name)

    elif dataset_type == "TOSHIBA":
        return normalize_toshiba(df, source_name)

    else:
        return normalize_unknown(df, source_name)


# ------------------------------
#  MERGING
# ------------------------------

def merge_datasets(dataset_list):
    """Merges all cleaned datasets into a single sorted DataFrame."""
    print("[INFO] Merging cleaned datasets...")
    
    df = pd.concat(dataset_list, ignore_index=True)
    df = df.sort_values("timestamp")
    df = df.dropna(subset=["timestamp"])

    return df


#  SAVE
# En merge_datasets o antes de save_processed:
def resample_qkd_data(df):
    # Agrupamos por fuente para no mezclar tiempos de máquinas distintas
    resampled_list = []
    for source in df['source'].unique():
        source_df = df[df['source'] == source].copy()
        source_df = source_df.set_index('timestamp')
        
        # Resample a 1 minuto (o la frecuencia que decidas)
        # Usamos mean() para métricas y ffill() para el nombre de la fuente
        source_df = source_df.resample('1T').mean() 
        source_df['source'] = source
        
        # Opcional: Interpolar huecos pequeños para que el LSTM no vea NaNs
        source_df[['qber', 'skr']] = source_df[['qber', 'skr']].interpolate(method='linear', limit=2)
        
        resampled_list.append(source_df.reset_index())
    
    return pd.concat(resampled_list, ignore_index=True)

def save_processed(df):
    
    # 1. Crear el directorio usando la constante
    constants.PREPROCESSING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Definir la ruta de salida
    output_path = constants.PREPROCESSING_OUTPUT_DIR / constants.PROCESSED_FILE_NAME
    
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved processed dataset: {output_path}")

#  MAIN

def main():
    print("[INFO] Starting preprocessing pipeline...")

    raw_files = load_all_raw()
    cleaned = []

    for df_raw, name in raw_files:
        df = standardize_column_names(df_raw)
        dtype = detect_dataset_type(df)

        print(f"[INFO] Detected dataset type: {dtype} ({name})")

        df_clean = apply_dataset_specific_cleaning(df, dtype, name)
        cleaned.append(df_clean)

    merged = merge_datasets(cleaned)

    save_processed(merged)

    print("[INFO] Preprocessing completed successfully.")


# ---- ENTRY POINT ----

if __name__ == "__main__":
    main()
