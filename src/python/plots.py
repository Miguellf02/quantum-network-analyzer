import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.constants import constants

# Configuración de Estilo UPM
plt.style.use('seaborn-v0_8-muted') 

def plot_advanced_operational_dashboard(source):
    path = constants.MODEL_OUTPUT_DIR / f"MERGED_ANOMALIES_{source}.csv"
    if not path.exists(): return
    
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp')
    
    # --- PLOT DOBLE EJE (LA JOYA DE LA CORONA) ---
    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    # Eje 1: SKR
    ax1.plot(df['timestamp'], df['skr'], color='tab:blue', alpha=0.4, label='SKR (bps)')
    ax1.set_ylabel('Secure Key Rate (bps)', color='tab:blue', fontsize=12)
    
    # Eje 2: QBER
    ax2 = ax1.twinx()
    ax2.plot(df['timestamp'], df['qber'], color='tab:red', alpha=0.3, label='QBER')
    ax2.set_ylabel('QBER (%)', color='tab:red', fontsize=12)

    # Añadir los puntos de anomalía coloreados
    colors = {"BOTH": "#e74c3c", "AE_ONLY": "#f1c40f", "IF_ONLY": "#f39c12"}
    for a_type, color in colors.items():
        subset = df[df["anomaly_type"] == a_type]
        ax1.scatter(subset["timestamp"], subset["skr"], color=color, label=f"Anomaly: {a_type}", s=30, zorder=5)

    plt.title(f"Análisis Operativo Multivariante - Fuente: {source}", fontsize=14)
    ax1.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.savefig(constants.PLOTS_DIR / f"DASHBOARD_{source}.png", dpi=300)
    plt.close()

    # --- PLOT 2: SCATTER FÍSICO (Para ver la correlación) ---
    plt.figure(figsize=(8, 6))
    for a_type, color in colors.items():
        sub = df[df["anomaly_type"] == a_type]
        plt.scatter(sub["qber"], sub["skr"], color=color, label=a_type, alpha=0.6)
    
    # Dibujamos los puntos normales en gris muy claro al fondo
    normal = df[df["anomaly_type"] == "NONE"]
    plt.scatter(normal["qber"], normal["skr"], color='lightgray', alpha=0.2, label='Normal', zorder=1)
    
    plt.xlabel("QBER")
    plt.ylabel("SKR (bps)")
    plt.title(f"Espacio de Fase QBER-SKR: {source}")
    plt.legend()
    plt.savefig(constants.PLOTS_DIR / f"SCATTER_{source}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Definimos las fuentes que queremos dibujar
     sources = [
        "QTI", 
        "TOSHIBA-2024-W25", 
        "TOSHIBA-2025-W27", 
        #"TOSHIBA-2025-W27-DOS-ATTACK", 
        #"TOSHIBA-2025-W27-DRIFT", 
        "TOSHIBA-2025-W27-EVE-ATTACK"
    ]
    
    # Creamos la carpeta si no existe
constants.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
print(f"\n[INFO] Generando visualizaciones en: {constants.PLOTS_DIR}")
    
for s in sources:
        print(f" -> Procesando: {s}")
        plot_advanced_operational_dashboard(s)
        
print("[OK] Proceso de visualización finalizado.")