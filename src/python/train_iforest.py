"""
train_iforest_per_source.py
------------------------------------------------
Entrenamiento de Isolation Forest por fuente
(QTI / TOSHIBA-2024-W25 / TOSHIBA-2025-W27)

Nota metodológica:
- Aprendizaje NO supervisado
- El modelo se entrena para aprender el comportamiento normal
- Las anomalías se definen como desviaciones estadísticas
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.constants.constants import (
    INPUT_FEATURED_PATH,
    MODEL_OUTPUT_DIR
)

# ================================================================
# CONFIGURACIÓN
# ================================================================

SOURCES = [
    "QTI",
    "TOSHIBA-2024-W25",
    "TOSHIBA-2025-W27"
]

# Contaminación conservadora por fuente
# En entornos reales este valor se ajustaría según SLA / política operativa
CONTAMINATION_BY_SOURCE = {
    "QTI": 0.01,
    "TOSHIBA-2024-W25": 0.01,
    "TOSHIBA-2025-W27": 0.01
}

N_ESTIMATORS = 600
RANDOM_STATE = 42

# Variables temporales discretas que deben entrar al modelo
TEMPORAL_FEATURES = [
    "feat_hour",
    "feat_dayofweek",
    "feat_is_night",
    "feat_time_segment"
]

# ================================================================
# UTILIDADES
# ================================================================

def normalize_source_column(df):
    """Normaliza la columna source para evitar errores de comparación."""
    df["source"] = (
        df["source"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return df


def prepare_numeric_features(df, nan_threshold=0.4):
    """
    Selecciona features numéricas y gestiona NaN de forma robusta.

    Pasos:
    1. Selección numérica
    2. Limpieza de inf
    3. Eliminación de columnas muy incompletas
    4. Eliminación de filas con NaN restantes
    """

    # 1. Selección numérica
    X = df.select_dtypes(include=[np.number]).copy()

    if X.empty:
        return None, None

    # 2. Limpieza de infinitos
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3. Eliminar columnas con demasiados NaN
    nan_ratio = X.isna().mean()
    cols_to_keep = nan_ratio[nan_ratio <= nan_threshold].index

    if len(cols_to_keep) == 0:
        return None, None

    X = X[cols_to_keep]

    # 4. Eliminar filas con NaN restantes
    valid_idx = X.dropna().index

    if len(valid_idx) == 0:
        return None, None

    return X.loc[valid_idx], df.loc[valid_idx]


def train_iforest(X_scaled, contamination):
    """
    Entrenamiento Isolation Forest.
    Se entrena sobre el propio dataset (one-class learning).
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=N_ESTIMATORS,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    model.fit(X_scaled)
    return model


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n========== Isolation Forest por fuente ==========\n")

    # 1. Cargar dataset
    df = pd.read_csv(INPUT_FEATURED_PATH)
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # 2. Normalizar source
    df = normalize_source_column(df)

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for source in SOURCES:

        source_norm = source.upper()
        print(f"\n========== SOURCE: {source_norm} ==========")

        # 3. Filtrar por source
        df_src = df[df["source"] == source_norm].copy()

        if df_src.empty:
            print(f"[WARN] No hay filas para source = {source_norm}")
            continue

        # 4. Preparar features numéricas
        X, df_clean = prepare_numeric_features(df_src)

        if X is None:
            print(f"[WARN] Todas las filas inválidas tras limpieza para {source_norm}")
            continue

        print(f"[INFO] Filas válidas: {X.shape[0]}")
        print(f"[INFO] Features usadas: {X.shape[1]}")

        # 5. Escalado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 6. Entrenamiento
        contamination = CONTAMINATION_BY_SOURCE.get(source_norm, 0.01)
        model = train_iforest(X_scaled, contamination)

        # 7. Scores y etiquetas
        anomaly_score = -model.score_samples(X_scaled)
        anomaly_label = model.predict(X_scaled)

        # 8. Guardar resultados
        df_out = df_clean.copy()
        df_out["anomaly_score"] = anomaly_score
        df_out["anomaly_label"] = anomaly_label

        csv_path = MODEL_OUTPUT_DIR / f"IFOREST_RESULTS_{source_norm}.csv"
        df_out.to_csv(csv_path, index=False)

        joblib.dump(
            model,
            MODEL_OUTPUT_DIR / f"IFOREST_MODEL_{source_norm}.joblib"
        )
        joblib.dump(
            scaler,
            MODEL_OUTPUT_DIR / f"IFOREST_SCALER_{source_norm}.joblib"
        )

        print(f"[OK] Resultados guardados para {source_norm}")

    print("\n[OK] Entrenamiento Isolation Forest COMPLETADO.\n")


if __name__ == "__main__":
    main()
