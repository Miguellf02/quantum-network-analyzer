"""
analyze_anomalies.py
------------------------------------------------
Análisis avanzado de anomalías detectadas en infraestructuras QKD
a partir de la fusión de Isolation Forest y Autoencoder.

Este script:
- Analiza consistencia entre modelos
- Identifica eventos críticos
- Caracteriza tipologías de anomalías
- Extrae conclusiones operativas

Pensado para:
- Análisis final del TFG
- Soporte a conclusiones
- Preparación de validación sintética (Scapy)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.constants.constants import MODEL_OUTPUT_DIR

# ================================================================
# CONFIGURACIÓN
# ================================================================

SOURCES = [
    "QTI",
    "TOSHIBA-2024-W25",
    "TOSHIBA-2025-W27"
]

TOP_K_EVENTS = 10   # eventos más anómalos a inspeccionar
OUTPUT_DIR = MODEL_OUTPUT_DIR / "analysis_reports"


# ================================================================
# UTILIDADES
# ================================================================

def load_merged(source):
    path = MODEL_OUTPUT_DIR / f"MERGED_ANOMALIES_{source}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")
    return pd.read_csv(path, parse_dates=["timestamp"])


def describe_metric(df, metric):
    return {
        "mean": df[metric].mean(),
        "std": df[metric].std(),
        "min": df[metric].min(),
        "max": df[metric].max()
    }


# ================================================================
# ANÁLISIS PRINCIPAL
# ================================================================

def analyze_source(df, source):

    report = {}
    report["source"] = source
    report["total_samples"] = len(df)

    # ------------------------------------------------------------
    # 1. Distribución de anomalías
    # ------------------------------------------------------------
    anomaly_counts = df["anomaly_type"].value_counts().to_dict()
    report["anomaly_distribution"] = anomaly_counts

    # ------------------------------------------------------------
    # 2. Eventos críticos (IF + AE)
    # ------------------------------------------------------------
    critical = df[df["anomaly_type"] == "BOTH"]
    report["critical_events"] = len(critical)

    # ------------------------------------------------------------
    # 3. Estadísticas por tipo de anomalía
    # ------------------------------------------------------------
    stats_by_type = {}

    for a_type in ["NONE", "IF_ONLY", "AE_ONLY", "BOTH"]:
        subset = df[df["anomaly_type"] == a_type]
        if subset.empty:
            continue

        stats_by_type[a_type] = {
            "count": len(subset),
            "qber": describe_metric(subset, "qber"),
            "skr": describe_metric(subset, "skr"),
            "loss": describe_metric(subset, "loss"),
        }

    report["stats_by_anomaly_type"] = stats_by_type

    # ------------------------------------------------------------
    # 4. Top eventos más anómalos
    # ------------------------------------------------------------
    df["combined_score"] = (
        df["anomaly_score"] +
        df["anomaly_score_autoencoder"]
    )

    top_events = df.sort_values(
        "combined_score",
        ascending=False
    ).head(TOP_K_EVENTS)

    report["top_events"] = top_events[
        ["timestamp", "qber", "skr", "loss",
         "anomaly_type", "combined_score"]
    ]

    return report, top_events


# ================================================================
# MAIN
# ================================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n========== ANALYSIS OF QKD ANOMALIES ==========\n")

    global_summary = []

    for source in SOURCES:

        print(f"\n========== SOURCE: {source} ==========")

        df = load_merged(source)
        report, top_events = analyze_source(df, source)

        # --------------------------------------------------------
        # Guardar top eventos
        # --------------------------------------------------------
        top_events_path = OUTPUT_DIR / f"TOP_EVENTS_{source}.csv"
        top_events.to_csv(top_events_path, index=False)

        # --------------------------------------------------------
        # Guardar resumen textual
        # --------------------------------------------------------
        summary_path = OUTPUT_DIR / f"SUMMARY_{source}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Analysis report for source: {source}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {report['total_samples']}\n\n")
            f.write("Anomaly distribution:\n")
            for k, v in report["anomaly_distribution"].items():
                f.write(f"  {k}: {v}\n")

            f.write(f"\nCritical events (IF + AE): {report['critical_events']}\n\n")

            f.write("Statistics by anomaly type:\n")
            for a_type, stats in report["stats_by_anomaly_type"].items():
                f.write(f"\n[{a_type}]\n")
                f.write(f"Count: {stats['count']}\n")
                for metric in ["qber", "skr", "loss"]:
                    m = stats[metric]
                    f.write(
                        f"{metric.upper()}: "
                        f"mean={m['mean']:.4f}, "
                        f"std={m['std']:.4f}, "
                        f"min={m['min']:.4f}, "
                        f"max={m['max']:.4f}\n"
                    )

        print(f"[OK] Report generated for {source}")

        global_summary.append({
            "source": source,
            "total_samples": report["total_samples"],
            "critical_events": report["critical_events"],
            **report["anomaly_distribution"]
        })

    # ------------------------------------------------------------
    # Guardar resumen global
    # ------------------------------------------------------------
    summary_df = pd.DataFrame(global_summary)
    summary_df.to_csv(
        OUTPUT_DIR / "GLOBAL_SUMMARY.csv",
        index=False
    )

    print("\n[OK] GLOBAL ANALYSIS COMPLETED\n")


if __name__ == "__main__":
    main()
