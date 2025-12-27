"""
Advanced Feature Engineering Module for QKD Operational Data.

This script takes the unified, cleaned QKD dataset and creates enriched,
multi-scale temporal features for anomaly detection using Isolation Forest
and Autoencoders.

Input: QKD_PROCESSED.csv
Output: QKD_FEATURES.csv

Author: Miguel López Ferreiro
"""

import pandas as pd
import numpy as np
from pathlib import Path
from ..constants import constants


# CONFIGURATION

CORE_METRICS = ['qber', 'skr', 'loss']

# EXTENDED LAG WINDOWS
LAG_WINDOWS = [1,2,3,6,12,24,48]  # samples, NOT minutes

# MULTI-SCALE ROLLING WINDOWS
ROLL_WINDOWS = [5, 10, 20]  # samples, NOT minutes

# path config
INPUT_PATH = constants.INPUT_PROCESSED_PATH
OUTPUT_PATH = constants.FEATURE_ENGINEERING_OUTPUT_DIR / constants.FEATURED_FILE_NAME



# LOADING

def load_processed_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found at: {file_path}. Run preprocessing first.")
    
    print(f"[INFO] Loading processed data from: {file_path}")
    df = pd.read_csv(file_path)

    # CAMBIO AQUÍ: Usamos format='ISO8601' para que sea flexible con los milisegundos
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
    
    # Eliminamos posibles NaNs en el timestamp que hayan fallado al parsear
    df = df.dropna(subset=['timestamp'])
    
    df = df.set_index('timestamp').sort_index()

    return df


# TEMPORAL FEATURES

def create_temporal_features(df):
    print("[INFO] Creating temporal features...")

    df['feat_hour'] = df.index.hour
    df['feat_day_of_week'] = df.index.dayofweek
    df['feat_day_of_year'] = df.index.dayofyear

    # Night indicator (QKD channels often degrade at night)
    df['feat_is_night'] = ((df.index.hour < 7) | (df.index.hour >= 22)).astype(int)

    # Segment of day (categorical → numeric)
    df['feat_time_segment'] = df.index.hour.map(
        lambda h: 0 if h < 6 else 1 if h < 12 else 2 if h < 18 else 3
    )

    return df


# LAGGED FEATURES

def create_lagged_features(df, metrics, lag_windows):
    print(f"[INFO] Creating lagged features with lags: {lag_windows}")

    for metric in metrics:
        for lag in lag_windows:
            df[f'feat_{metric}_lag_{lag}'] = df[metric].shift(lag)

    return df


# DERIVATIVES AND RATE FEATURES

def create_change_features(df, metrics):
    print("[INFO] Creating delta and pct-change features...")

    for metric in metrics:
        df[f'feat_{metric}_delta'] = df[metric].diff()
        df[f'feat_{metric}_pct'] = df[metric].pct_change()

    return df


# ROLLING FEATURES — MULTI WINDOW

def create_rolling_features(df, metrics, roll_windows):
    print(f"[INFO] Creating rolling stats for windows: {roll_windows}")

    for w in roll_windows:
        for metric in metrics:
            prefix = f'feat_{metric}_roll{w}'

            df[f'{prefix}_mean']   = df[metric].rolling(w).mean()
            df[f'{prefix}_std']    = df[metric].rolling(w).std()
            df[f'{prefix}_min']    = df[metric].rolling(w).min()
            df[f'{prefix}_max']    = df[metric].rolling(w).max()
            df[f'{prefix}_median'] = df[metric].rolling(w).median()

            # Local Z-score
            df[f'{prefix}_z'] = (
                (df[metric] - df[f'{prefix}_mean']) / df[f'{prefix}_std']
            )

    return df



# CROSS-METRIC INTERACTIONS

def create_interaction_features(df):
    print("[INFO] Creating interaction features (QBER-SKR-LOSS)...")

    # Physical relationships
    df['feat_qber_skr_ratio'] = df['qber'] / df['skr']
    df['feat_loss_skr_ratio'] = df['loss'] / df['skr']
    df['feat_qber_loss_product'] = df['qber'] * df['loss']

    # Stability proxies
    df['feat_skr_inv'] = 1 / df['skr'].replace(0, np.nan)
    df['feat_loss_gradient'] = df['loss'].diff()

    return df


# GLOBAL NORMALIZED METRICS

def create_global_zscores(df, metrics):
    print("[INFO] Creating global Z-scores...")

    for metric in metrics:
        mean = df[metric].mean()
        std = df[metric].std()
        df[f'feat_{metric}_z_global'] = (df[metric] - mean) / std

    return df


# MAIN PIPELINE

def main():
    print("[INFO] Starting Advanced Feature Engineering pipeline...")

    # 1. Load
    df = load_processed_data(INPUT_PATH)

    # 2. Temporal
    df = create_temporal_features(df)

    # 3. Lags
    df = create_lagged_features(df, CORE_METRICS, LAG_WINDOWS)

    # 4. Change features
    df = create_change_features(df, CORE_METRICS)

    # 5. Rolling multiscale
    df = create_rolling_features(df, CORE_METRICS, ROLL_WINDOWS)

    # 6. Cross-features
    df = create_interaction_features(df)

    # 7. Global Z-scores
    df = create_global_zscores(df, CORE_METRICS)

    # 8. Drop missing values
    initial_rows = len(df)
    df_clean = df
    dropped = initial_rows - len(df_clean)
    print(f"[INFO] Dropped {dropped} rows due to NaN (lags/rolling).")

    # 9. Select final columns
    context_cols = CORE_METRICS + ['source']
    feature_cols = [c for c in df_clean.columns if c.startswith('feat_')]

    df_final = df_clean[context_cols + feature_cols]

    # 10. Save
    constants.FEATURE_ENGINEERING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH)

    print(f"[INFO] Saved final feature set with {len(feature_cols)} features → {OUTPUT_PATH}")
    print("[INFO] Feature Engineering completed successfully.")


if __name__ == "__main__":
    main()
