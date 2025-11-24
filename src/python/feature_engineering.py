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

Author: Miguel López Ferreiro
"""

import pandas as pd
from pathlib import Path

# ------------------------------
# PATH CONFIG
# ------------------------------

BASE = Path(__file__).resolve().parents[2]

# Input directory (from the preprocessing script)
PROCESSED_DIR = BASE / "data" / "processed" / "python_preprocessing"
PROCESSED_NAME = "QKD_PROCESSED.csv"
INPUT_PATH = PROCESSED_DIR / PROCESSED_NAME

# Output directory for the feature-engineered dataset
FEATURED_DIR = BASE / "data" / "processed" / "feature_engineered"
FEATURED_NAME = "QKD_FEATURES.csv"
OUTPUT_PATH = FEATURED_DIR / FEATURED_NAME

# Configuration for rolling window statistics (e.g., 60 minutes)
ROLLING_WINDOW = 60 # Typically defined by the operational monitoring interval
LAG_WINDOWS = [1, 2, 3] # Lagging for 1, 2, and 3 previous time steps

# LOADING

def load_processed_data(file_path):
    """Loads the cleaned QKD dataset and sets the timestamp index."""
    if not file_path.exists():
        raise FileNotFoundError(f"Processed file not found at: {file_path}. Run preprocessing first.")
    
    print(f"[INFO] Loading processed data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Ensure 'timestamp' is in datetime format and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    return df

# 1. TEMPORAL FEATURES

def create_temporal_features(df):
    """Extracts time components (hour, day of week) that might influence metrics."""
    print("[INFO] Creating temporal features...")
    
    # Hour of the day (0-23): Useful for detecting daily cycles/traffic patterns
    df['feat_hour'] = df.index.hour
    
    # Day of the week (0=Monday, 6=Sunday): Useful for weekly cycles/maintenance
    df['feat_day_of_week'] = df.index.dayofweek
    
    # Day of the year (1-366): Less critical, but captures yearly trends
    df['feat_day_of_year'] = df.index.dayofyear
    
    # (Optional: If the data is periodic, use sin/cos transformations for hour/day_of_year)
    # df['feat_hour_sin'] = np.sin(2 * np.pi * df['feat_hour'] / 24)
    # df['feat_hour_cos'] = np.cos(2 * np.pi * df['feat_hour'] / 24)
    
    return df

# 2. LAGGED FEATURES

def create_lagged_features(df, metrics, lag_windows):
    """Creates lagged features (previous values) for the specified metrics."""
    print(f"[INFO] Creating lagged features for {metrics} with lags: {lag_windows}")
    
    for metric in metrics:
        for lag in lag_windows:
            new_col = f'feat_{metric}_lag_{lag}'
            # Shift the metric by 'lag' periods
            df[new_col] = df[metric].shift(lag)
            
    return df

# 3. ROLLING WINDOW STATISTICS (Stability)

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
        
        # Rolling Min/Max: Provides short-term bounds (useful for identifying spikes)
        df[f'feat_{metric}_roll_min'] = df[metric].rolling(window=window).min()
        df[f'feat_{metric}_roll_max'] = df[metric].rolling(window=window).max()
        
        # Rolling Median: Less sensitive to outliers, better for central tendency
        df[f'feat_{metric}_roll_median'] = df[metric].rolling(window=window).median()

        # Calculation of the short-term Z-Score (Deviation from the local mean)
        # This is a very powerful feature for anomaly detection
        df[f'feat_{metric}_short_zscore'] = (df[metric] - df[f'feat_{metric}_roll_mean']) / df[f'feat_{metric}_roll_std']
        
    return df

# MAIN PIPELINE

def main():
    print("[INFO] Starting Feature Engineering pipeline...")
    
    # 1. Load Data
    df = load_processed_data(INPUT_PATH)
    
    # Define core metrics for feature creation (QBER, SKR, Loss)
    core_metrics = ['qber', 'skr', 'loss']
    
    # 2. Create Temporal Features
    df = create_temporal_features(df)
    
    # 3. Create Lagged Features
    df = create_lagged_features(df, core_metrics, LAG_WINDOWS)
    
    # 4. Create Rolling Statistics Features
    df = create_rolling_features(df, core_metrics, ROLLING_WINDOW)

    # 5. Handle Missing Values
    # Features created with rolling windows or lags will have NaN values initially.
    # Since these NaNs typically represent the beginning of the time series,
    # and we are focusing on anomaly detection in steady state, we drop them.
    # Note: The dropna must occur *after* all features have been created.
    initial_rows = len(df)
    df_featured = df.dropna()
    rows_dropped = initial_rows - len(df_featured)
    
    print(f"[INFO] Dropped {rows_dropped} rows due to NaN values (lags/rolling stats initialization).")
    
    # 6. Final Selection of Features and Standardization
    # The ML models will be trained on the new features. We keep the original
    # metrics and source for context, but the final model input will use the new features.
    
    # Select only the features (columns starting with 'feat_') + original metrics/source
    feature_cols = [col for col in df_featured.columns if col.startswith('feat_')]
    context_cols = core_metrics + ['source'] # Keep original metrics and source for validation
    
    df_final = df_featured[context_cols + feature_cols]
    
    # 7. Save Processed Data
    FEATURED_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_PATH)
    
    print(f"[INFO] Saved final feature set with {len(feature_cols)} features: {OUTPUT_PATH}")
    print("[INFO] Feature Engineering completed successfully.")

# ---- ENTRY POINT ----

if __name__ == "__main__":
    main()