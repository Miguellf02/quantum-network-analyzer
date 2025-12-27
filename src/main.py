"""
MAIN PIPELINE – QKD DATA ANALYZER

Orchestrates the entire QKD data processing workflow.
"""

import subprocess
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent / "constants"))
import constants
sys.path.pop()

PROJECT_BASE_DIR = constants.BASE_DIR
DEMOSTRADOR = "inject_dos.py"


# EXECUTION FUNCTIONS
import shutil
import os
from pathlib import Path

def clean_and_restore():
    """
    Limpia los resultados de ejecuciones anteriores y restaura los datasets originales.
    Garantiza un entorno de ejecución limpio para el pipeline.
    """
    print("\n [CLEANING] Iniciando limpieza y restauración del entorno... \n")
    
    # 1. Rutas de limpieza (según tu estructura en constants)
    processed_dir = Path("data/processed")
    plots_dir = Path("data/plots")
    models_dir = Path("data/models")
    # Borrar contenido de data/processed (excepto primer_analisis)
    if processed_dir.exists():
        for item in processed_dir.iterdir():
            if item.is_dir() and item.name != "primer_analisis":
                shutil.rmtree(item)
                print(f"[OK] Eliminado directorio: {item.name}")
            elif item.is_file():
                item.unlink()
                print(f"[OK] Eliminado archivo: {item.name}")

    # Borrar contenido de data/plots
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
        print("[OK] Carpeta de gráficas (plots) vaciada.")

    if models_dir.exists():
        shutil.rmtree(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        print("[OK] Carpeta de gráficas (plots) vaciada.")

    # 2. Restauración del archivo original de Toshiba 2025
    # Asumiendo que guardaste el original en data/raw/original/
    source_original = Path("data/raw/original/Toshiba-2025-W27.csv")
    target_raw = Path("data/raw/qkd/Toshiba-2025-W27.csv")

    if source_original.exists():
        # Aseguramos que el destino exista
        target_raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_original, target_raw)
        print(f"[OK] Dataset 2025 restaurado desde el backup original.")
    else:
        print(f"[WARNING] No se encontró el backup en {source_original}. Verifica la ruta.")

    print("\n [SUCCESS] Entorno restaurado. Listos para nueva ejecución. \n")

def run_anomaly_injection():
    """Ejecuta la creación de escenarios sintéticos como un proceso independiente."""
    print("\n [INFO] GENERATING SYNTHETIC ATTACK SCENARIOS \n")
    
    # Ruta al script de inyección
    injector_path = PROJECT_BASE_DIR / "src" / "python" / "demostradores" / DEMOSTRADOR
    
    if injector_path.exists():
        subprocess.run(
            [sys.executable, str(injector_path)],
            check=True,
            cwd=PROJECT_BASE_DIR 
        )
    else:
        print(f"[ERROR] No se encontró el script en: {injector_path}")


def run_r_script():
    """Executes the initial R exploratory analysis script."""
    print("\n EXECUTING R SCRIPT \n")
    subprocess.run([constants.RSCRIPT_EXE_PATH, str(constants.R_SCRIPT_PATH)], check=True)

def run_python_preprocessing():
    """Executes the Python script for data cleaning and unification using -m."""
    print("\n EXECUTING PYTHON PREPROCESSING \n")
    
    # Ejecutamos como módulo, referenciando la constante del módulo
    subprocess.run(
        [sys.executable, "-m", constants.PYTHON_MODULE_PREPROCESSING],
        check=True,
        cwd=PROJECT_BASE_DIR 
    )

def run_python_feature_engineering():
    """Executes the Python script for creating temporal and statistical features using -m."""
    print("\n EXECUTING PYTHON FEATURE ENGINEERING \n")
    
    # Ejecutamos como módulo, referenciando la constante del módulo
    subprocess.run(
        [sys.executable, "-m", constants.PYTHON_MODULE_FEATURE_ENG],
        check=True,
        cwd=PROJECT_BASE_DIR 
    )

def run_python_iforest():
    """Executes the Isolation Forest training script using -m."""
    print("\n EXECUTING ISOLATION FOREST TRAINING \n")
    
    subprocess.run(
        [sys.executable, "-m", constants.PYTHON_MODULE_IFOREST],
        check=True,
        cwd=PROJECT_BASE_DIR
    )


def run_python_autoencoder():
    """Executes the Autoencoder training script using -m."""
    print("\n EXECUTING AUTOENCODER TRAINING \n")
    
    subprocess.run(
        [sys.executable, "-m", constants.PYTHON_MODULE_AUTOENCODER],
        check=True,
        cwd=PROJECT_BASE_DIR
    )


def run_python_merging():
    """Executes the merging and comparison of IF and Autoencoder results."""
    print("\n EXECUTING MERGING OF ANOMALY RESULTS (IF + AUTOENCODER) \n")
    
    subprocess.run(
        [sys.executable, "-m", constants.PYTHON_MODULE_MERGING],
        check=True,
        cwd=PROJECT_BASE_DIR
    )

def run_python_anomaly_analysis():
    """Executes the anomaly analysis script using -m."""
    print("\n EXECUTING ANOMALY ANALYSIS \n")

    subprocess.run(
        [sys.executable, "-m", constants.PYTHON_MODULE_ANALYSIS],
        check=True,
        cwd=PROJECT_BASE_DIR
    )

def run_python_plots():
    """Genera las visualizaciones finales de los resultados."""
    print("\n EXECUTING GENERATION OF OPERATIONAL PLOTS \n")
    
    # Definimos el módulo en tus constantes como: src.python.analysis.plots
    subprocess.run(
        [sys.executable, "-m", constants.PYTHON_MODULE_PLOTS],
        check=True,
        cwd=PROJECT_BASE_DIR
    )
# MAIN PIPELINE EXECUTION

def main():
    clean_and_restore()

    run_anomaly_injection()

    # 1. Ejecutar análisis R inicial (opcional)
    # run_r_script()
    
    # 2. Ejecutar Preprocesamiento Python
    run_python_preprocessing()
    
    # 3. Ejecutar Ingeniería de Características
    run_python_feature_engineering()

    # 4. Entrenar Isolation Forest
    run_python_iforest()

    # 5. Entrenar Autoencoder
    run_python_autoencoder()
    
        # 6. Fusión y análisis comparativo IF + Autoencoder
    run_python_merging()

    run_python_anomaly_analysis()


    run_python_plots()
    print("\nTHE PIPELINE HAS FINISHED SATISFACTORILY\n")

if __name__ == "__main__":
    main()