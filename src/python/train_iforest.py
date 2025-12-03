"""
train_iforest.py
------------------------------------
Entrenamiento del modelo Isolation Forest usando QKD_FEATURES.csv
Generado por feature_engineering.py
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Importar paths centralizados
from src.constants.constants import (
    INPUT_FEATURED_PATH,
    MODEL_OUTPUT_DIR
)

# ================================================================
# ---------------------- UTILIDADES ------------------------------
# ================================================================

def load_featured_dataset():
    """Carga el dataset con todas las features."""
    if not INPUT_FEATURED_PATH.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo de características:\n{INPUT_FEATURED_PATH}"
        )

    df = pd.read_csv(INPUT_FEATURED_PATH)
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def select_numeric_features(df):
    """
    Selecciona columnas numéricas, reemplaza inf/-inf por NaN
    y elimina filas no válidas.
    Devuelve:
        X_clean  = características válidas
        df_clean = dataframe original alineado
    """

    # 1. Selección numérica
    X = df.select_dtypes(include=[np.number])

    if X.empty:
        raise ValueError("No hay columnas numéricas en el dataset.")

    # 2. Reemplazar inf y -inf
    X = X.replace([np.inf, -np.inf], np.nan)

    # 3. Eliminar filas con NaN
    valid_idx = X.dropna().index
    X_clean = X.loc[valid_idx]
    df_clean = df.loc[valid_idx]

    print(f"[INFO] Columnas numéricas seleccionadas: {X_clean.shape[1]}")
    print(f"[INFO] Filas válidas tras limpieza: {X_clean.shape[0]} / {df.shape[0]}")

    return X_clean, df_clean


def standardize_features(X):
    """Estandarización del espacio vectorial."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def train_isolation_forest(
        X_scaled,
        contamination=0.01,
        n_estimators=600,
        max_samples="auto",
        random_state=42):

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state
    )

    model.fit(X_scaled)
    return model


def compute_anomaly_scores(model, X_scaled):
    """Convierte scores a formato intuitivo: mayor = más anómalo."""
    raw_scores = model.score_samples(X_scaled)
    return -raw_scores


def save_outputs(df_clean, scores, model, scaler):
    """Guarda CSV, modelo y scaler."""
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) CSV de salida
    df_out = df_clean.copy()
    df_out["anomaly_score"] = scores
    output_csv = MODEL_OUTPUT_DIR / "IFOREST_RESULTS.csv"
    df_out.to_csv(output_csv, index=False)
    print(f"[OK] Resultados guardados en: {output_csv}")

    # 2) Modelo
    model_path = MODEL_OUTPUT_DIR / "IFOREST_MODEL.joblib"
    joblib.dump(model, model_path)
    print(f"[OK] Modelo guardado en: {model_path}")

    # 3) Scaler
    scaler_path = MODEL_OUTPUT_DIR / "IFOREST_SCALER.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Scaler guardado en: {scaler_path}")


# ================================================================
# -------------------------- MAIN --------------------------------
# ================================================================

def main():
    print("\n========== Isolation Forest Training ==========\n")

    # 1. Cargar dataset
    df = load_featured_dataset()

    # 2. Seleccionar columnas numéricas + limpiar NAN/INF
    X_clean, df_clean = select_numeric_features(df)

    # 3. Estandarizar
    print("[INFO] Estandarizando características...")
    X_scaled, scaler = standardize_features(X_clean)

    # 4. Entrenar Isolation Forest
    print("[INFO] Entrenando Isolation Forest...")
    model = train_isolation_forest(X_scaled)

    # 5. Scores
    print("[INFO] Calculando anomaly scores...")
    scores = compute_anomaly_scores(model, X_scaled)

    # 6. Guardar salidas
    print("[INFO] Guardando resultados y modelo...")
    save_outputs(df_clean, scores, model, scaler)

    print("\n[OK] Entrenamiento Isolation Forest COMPLETADO.\n")


if __name__ == "__main__":
    main()
