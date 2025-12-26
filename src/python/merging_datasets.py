"""
merging_datasets.py
------------------------------------------------
Fusión de resultados de detección de anomalías
(Isolation Forest + Autoencoder) por source.

Salida:
    - MERGED_ANOMALIES_<SOURCE>.csv

El dataset resultante permite:
- comparar modelos
- analizar consistencia
- preparar validación sintética (Scapy)
"""

import pandas as pd
import numpy as np

from src.constants.constants import MODEL_OUTPUT_DIR

# ================================================================
# CONFIGURACIÓN
# ================================================================

SOURCES = [
    "QTI",
    "TOSHIBA-2024-W25",
    "TOSHIBA-2025-W27"
]

AUTOENCODER_PERCENTILE = 99  # umbral de anomalía AE


# ================================================================
# UTILIDADES
# ================================================================

def load_results(source):
    """Carga los CSV de IF y Autoencoder para una source."""
    if_path = MODEL_OUTPUT_DIR / f"IFOREST_RESULTS_{source}.csv"
    ae_path = MODEL_OUTPUT_DIR / f"AUTOENCODER_RESULTS_{source}.csv"

    if not if_path.exists():
        raise FileNotFoundError(f"No existe {if_path}")
    if not ae_path.exists():
        raise FileNotFoundError(f"No existe {ae_path}")

    df_if = pd.read_csv(if_path)
    df_ae = pd.read_csv(ae_path)

    return df_if, df_ae


def compute_ae_threshold(scores, percentile):
    """Calcula el umbral de anomalía para el Autoencoder."""
    return np.percentile(scores, percentile)


# ================================================================
# MAIN
# ================================================================

def main():

    print("\n========== MERGING IF + AUTOENCODER ==========\n")

    for source in SOURCES:

        print(f"\n========== SOURCE: {source} ==========")

        # 1. Cargar resultados
        df_if, df_ae = load_results(source)

        # 2. Merge por timestamp + source
        df = pd.merge(
            df_if,
            df_ae[["timestamp", "source", "anomaly_score_autoencoder"]],
            on=["timestamp", "source"],
            how="inner"
        )

        # 3. Flags de anomalía
        df["anomaly_if"] = df["anomaly_label"] == -1

        ae_threshold = compute_ae_threshold(
            df["anomaly_score_autoencoder"],
            AUTOENCODER_PERCENTILE
        )

        df["anomaly_ae"] = df["anomaly_score_autoencoder"] >= ae_threshold

        # 4. Clasificación combinada
        def classify(row):
            if row["anomaly_if"] and row["anomaly_ae"]:
                return "BOTH"
            if row["anomaly_if"]:
                return "IF_ONLY"
            if row["anomaly_ae"]:
                return "AE_ONLY"
            return "NONE"

        df["anomaly_type"] = df.apply(classify, axis=1)

        # 5. Guardar resultado
        output_path = MODEL_OUTPUT_DIR / f"MERGED_ANOMALIES_{source}.csv"
        df.to_csv(output_path, index=False)

        # 6. Resumen
        print(f"[OK] Guardado: {output_path}")
        print("Resumen de anomalías:")
        print(df["anomaly_type"].value_counts())

    print("\n[OK] MERGING COMPLETADO.\n")


if __name__ == "__main__":
    main()
