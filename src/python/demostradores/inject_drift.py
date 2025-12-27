"""
inject_drift.py
------------------------------------------------
Inyección de degradación progresiva del canal (drift lento).

Escenario simulado:
- Envejecimiento / desalineación gradual del canal óptico
- Aumento progresivo del QBER
- Caída suave y sostenida de la SKR

Este escenario NO genera outliers abruptos.
Está diseñado para validar detección estructural (Autoencoder).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Rutas
RAW_PATH = Path("data/raw/qkd/Toshiba-2025-W27.csv")
OUTPUT_PATH = Path("data/raw/qkd/Toshiba-2025-W27-DRIFT.csv")

def inject():
    df = pd.read_csv(RAW_PATH)

    # Ventana de degradación progresiva
    start, end = 600, 900
    length = end - start

    print(f"[ATTACK] Inyectando degradación progresiva (drift) en {RAW_PATH.name}...")

    # Asegurar tipos
    df["SecureKeyRate(bps)"] = df["SecureKeyRate(bps)"].astype(float)
    df["QBER"] = df["QBER"].astype(float)

    # Factores de degradación progresiva
    skr_factors = np.linspace(1.0, 0.6, length)   # caída del 40% gradual
    qber_factors = np.linspace(1.0, 1.4, length)  # subida del 40% gradual

    # Aplicar drift
    df.loc[start:end-1, "SecureKeyRate(bps)"] *= skr_factors
    df.loc[start:end-1, "QBER"] *= qber_factors

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Archivo con drift generado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    inject()
