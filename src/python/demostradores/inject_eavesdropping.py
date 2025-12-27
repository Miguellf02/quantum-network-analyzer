import pandas as pd
from pathlib import Path
import numpy as np

# Configuración de rutas (Rutas relativas desde la raíz del proyecto)
RAW_PATH = Path("data/raw/qkd/Toshiba-2025-W27.csv")
OUTPUT_PATH = Path("data/raw/qkd/Toshiba-2025-W27-EVE-ATTACK.csv")

def inject():
    if not RAW_PATH.exists():
        print(f"[ERROR] No existe el archivo base: {RAW_PATH}")
        return

    # Cargamos el dataset original de Toshiba
    df = pd.read_csv(RAW_PATH)
    
    # Definimos una ventana de ataque (ej. de la fila 2000 a la 2100)
    # Un ataque de intercepción suele ser más prolongado y sutil que un DoS
    start, end = 2000, 2100
    
    print(f"[ATTACK] Inyectando Eavesdropping (Intercepción) en {RAW_PATH.name}...")

    # Aseguramos tipos float para evitar Warnings de pandas
    df['QBER'] = df['QBER'].astype(float)
    df['SecureKeyRate(bps)'] = df['SecureKeyRate(bps)'].astype(float)

    # 1. EFECTO EN EL QBER: Subida progresiva
    # Simulamos que Eve empieza con una intercepción débil y va aumentando.
    # El QBER normal está en ~2.8%. Lo subiremos hasta un 8-10% (umbral crítico).
    noise = np.linspace(1.5, 3.5, num=(end - start + 1)) 
    df.loc[start:end, 'QBER'] = df.loc[start:end, 'QBER'] * noise

    # 2. EFECTO EN LA SKR: Caída proporcional a la corrección de errores
    # Al subir el QBER, el protocolo BB84 gasta más bits en corregir, bajando la tasa neta.
    # Simulamos una caída del 40% al 70% de la capacidad.
    drop_factor = np.linspace(0.6, 0.3, num=(end - start + 1))
    df.loc[start:end, 'SecureKeyRate(bps)'] = df.loc[start:end, 'SecureKeyRate(bps)'] * drop_factor

    # Guardamos el nuevo dataset de ataque
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Escenario de Intercepción generado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    inject()