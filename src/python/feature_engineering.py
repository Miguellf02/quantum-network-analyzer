"""
Feature Engineering Module for QKD Operational Data.

This script takes the unified, cleaned QKD dataset and transforms the
time-series metrics (QBER, SKR, Loss) into advanced features suitable
for Unsupervised Machine Learning anomaly detection models (e.g.,
Isolation Forest and Autoencoders).

The feature engineering focuses on:
1. Temporal (Time-based) components.
2. Lagged values (Time dependencies).
3. Rolling statistics (Stability and short-term trends).

Input: QKD_PROCESSED.csv
Output: QKD_FEATURES.csv

Uses constants defined in the 'constants' module for paths and file names.

Author: Miguel López Ferreiro
"""

import pandas as pd
import numpy as np # Necesario para el cálculo de Z-Score y otras operaciones
from pathlib import Path

# Importamos el módulo de constantes.
# Asumiendo la estructura de importación relativa (ej. desde src/python/feature_engineering.py a src/constants/constants.py)
from ..constants import constants 


# ------------------------------
# CONFIGURATION FOR FEATURE CREATION (Not defined in current constants.py)
# ------------------------------

# Define core metrics for feature creation (if not centralized in constants)
CORE_METRICS = ['qber', 'skr', 'loss']

# Configuration for rolling window statistics (e.g., 60 minutes)
ROLLING_WINDOW = 60 
LAG_WINDOWS = [1, 2, 3] 


# ------------------------------
# PATH CONFIG (Now derived from constants)
# ------------------------------

# Input path (from constants)
INPUT_PATH = constants.INPUT_PROCESSED_PATH

# Output path (from constants)
OUTPUT_PATH = constants.FEATURE_ENGINEERING_OUTPUT_DIR / constants.FEATURED_FILE_NAME


# ------------------------------
# LOADING
# ------------------------------

def load_processed_data(file_path):
    """Loads the cleaned QKD dataset and sets the timestamp index."""
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found at: {file_path}. Run preprocessing first.")
    
    print(f"[INFO] Loading processed data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Ensure 'timestamp' is in datetime format and set as index
    # Usamos constante para el nombre de columna si estuviera definida (ej. constants.TIMESTAMP_COL)
    df['timestamp'] = pd.to_datetime(df['timestamp']) 
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    return df

# ------------------------------
# 1. TEMPORAL FEATURES (No cambian, solo necesitan numpy si se usan sin/cos)
# ------------------------------

def create_temporal_features(df):
    """Extracts time components (hour, day of week) that might influence metrics."""
    print("[INFO] Creating temporal features...")
    
    df['feat_hour'] = df.index.hour
    df['feat_day_of_week'] = df.index.dayofweek
    df['feat_day_of_year'] = df.index.dayofyear
    
    return df

# ------------------------------
# 2. LAGGED FEATURES
# ------------------------------

def create_lagged_features(df, metrics, lag_windows):
    """Creates lagged features (previous values) for the specified metrics."""
    print(f"[INFO] Creating lagged features for {metrics} with lags: {lag_windows}")
    
    for metric in metrics:
        for lag in lag_windows:
            new_col = f'feat_{metric}_lag_{lag}'
            df[new_col] = df[metric].shift(lag)
            
    return df

# ------------------------------
# 3. ROLLING WINDOW STATISTICS (Stability)
# ------------------------------

def create_rolling_features(df, metrics, window):
    """
    Calculates rolling window statistics to capture short-term stability,
    which is essential for anomaly detection.
    """
    print(f"[INFO] Creating rolling statistics over window size: {window}")
    
    for metric in metrics:
        # Rolling Mean: Captures short-term trend/baseline
        df[f'feat_{metric}_roll_mean'] = df[metric].rolling(window=window).mean()
        
        # Rolling Std Dev: Captures short-term variability/stability (critical for QKD)
        df[f'feat_{metric}_roll_std'] = df[metric].rolling(window=window).std()
        
        # Rolling Min/Max
        df[f'feat_{metric}_roll_min'] = df[metric].rolling(window=window).min()
        df[f'feat_{metric}_roll_max'] = df[metric].rolling(window=window).max()
        
        # Rolling Median
        df[f'feat_{metric}_roll_median'] = df[metric].rolling(window=window).median()

        # Calculation of the short-term Z-Score
        df[f'feat_{metric}_short_zscore'] = (df[metric] - df[f'feat_{metric}_roll_mean']) / df[f'feat_{metric}_roll_std']
        
    return df

# ------------------------------
# MAIN PIPELINE
# ------------------------------

def main():
    print("[INFO] Starting Feature Engineering pipeline...")
    
    # 1. Load Data
    df = load_processed_data(INPUT_PATH)
    
    # 2. Create Temporal Features
    df = create_temporal_features(df)
    
    # 3. Create Lagged Features
    df = create_lagged_features(df, CORE_METRICS, LAG_WINDOWS)
    
    # 4. Create Rolling Statistics Features
    df = create_rolling_features(df, CORE_METRICS, ROLLING_WINDOW)

    # 5. Handle Missing Values
    initial_rows = len(df)
    df_featured = df.dropna()
    rows_dropped = initial_rows - len(df_featured)
    
    print(f"[INFO] Dropped {rows_dropped} rows due to NaN values (lags/rolling stats initialization).")
    
    # 6. Final Selection of Features and Standardization
    
    # Seleccionamos las métricas primarias para el contexto (qber, skr, loss, source)
    context_cols = CORE_METRICS + ['source'] 
    
    # Seleccionamos las nuevas características
    feature_cols = [col for col in df_featured.columns if col.startswith('feat_')]
    
    df_final = df_featured[context_cols + feature_cols]
    
    # 7. Save Processed Data
    # Usamos constante para el directorio de salida
    constants.FEATURE_ENGINEERING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH)
    
    print(f"[INFO] Saved final feature set with {len(feature_cols)} features: {OUTPUT_PATH}")
    print("[INFO] Feature Engineering completed successfully.")

# ---- ENTRY POINT ----

if __name__ == "__main__":
    main()