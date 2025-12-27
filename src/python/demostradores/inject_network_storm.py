import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/qkd/Toshiba-2025-W27.csv")
OUTPUT_PATH = Path("data/raw/qkd/Toshiba-2025-W27.csv")

def inject():
    df = pd.read_csv(RAW_PATH)
    # Ventana de la tormenta (ej: 8400 a 8600)
    start, end = 8400, 8600
    
    print(f"[ATTACK] Inyectando Tormenta de Red (Network Storm) en {RAW_PATH.name}...")
    
    df['SecureKeyRate(bps)'] = df['SecureKeyRate(bps)'].astype(float)
    df['QBER'] = df['QBER'].astype(float)

    # El efecto de un bucle es estocástico (aleatorio)
    # Creamos un ruido de alta frecuencia para la SKR y el QBER
    for i in range(start, end):
        # Caídas y subidas bruscas de la SKR (inestabilidad)
        df.loc[i, 'SecureKeyRate(bps)'] *= np.random.uniform(0.2, 0.9)
        # Picos de QBER por desincronización
        df.loc[i, 'QBER'] *= np.random.uniform(1.0, 2.5)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Escenario de tormenta generado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    inject()