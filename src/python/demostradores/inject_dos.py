import pandas as pd
from pathlib import Path
#inject_dos.py
# Configuración de rutas (ajusta a tus constantes)
RAW_PATH = Path("data/raw/qkd/Toshiba-2025-W27.csv")
OUTPUT_PATH = Path("data/raw/qkd/Toshiba-2025-W27-DOS-ATTACK.csv")

def inject():
    df = pd.read_csv(RAW_PATH)
    
    # Definimos el punto de ataque (ej. de la fila 500 a la 550)
    start, end = 500, 550
    
    print(f"[ATTACK] Inyectando DoS en {RAW_PATH.name}...")
    
    # Simulamos saturación del KMS: Caída del 95% en la tasa de clave
    # Usamos los nombres de columna originales de Toshiba antes del preprocesado
    df['SecureKeyRate(bps)'] = df['SecureKeyRate(bps)'].astype(float)
    df.loc[start:end, 'SecureKeyRate(bps)'] = df.loc[start:end, 'SecureKeyRate(bps)'] * 0.05  
    
    # El QBER en Toshiba suele ser estable, lo subimos un poco para simular jitter
    df.loc[start:end, 'QBER'] = df.loc[start:end, 'QBER'] * 1.5
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[SUCCESS] Archivo de ataque generado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    inject()