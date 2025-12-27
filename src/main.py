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
    Limpia a fondo los resultados de ejecuciones pasadas y restaura el dataset 2025.
    """
    print("\n [CLEANING] Iniciando limpieza profunda del entorno... \n")
    
    # Definir rutas base usando la constante del proyecto
    # Asegúrate de que PROJECT_BASE_DIR esté bien definido en tu main
    base_path = Path(__file__).resolve().parents[1] 
    
    processed_dir = base_path / "data" / "processed"
    plots_dir = base_path / "data" / "plots"
    models_dir = processed_dir / "models"

    # 1. Limpieza de data/processed (excepto primer_analisis)
    if processed_dir.exists():
        for item in processed_dir.iterdir():
            if item.name == "primer_analisis":
                continue
            if item.is_dir():
                shutil.rmtree(item)
                print(f"[OK] Directorio eliminado: {item.name}")
            else:
                item.unlink()
                print(f"[OK] Archivo eliminado: {item.name}")

    # 2. Limpieza específica de carpetas de modelos (por si acaso quedaron huérfanas)
    sub_models = ["evaluation", "analysis_reports"]
    for sub in sub_models:
        target = models_dir / sub
        if target.exists():
            shutil.rmtree(target)
            print(f"[OK] Limpieza profunda: models/{sub} borrado.")

    # 3. Limpieza de data/plots
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("[OK] Carpeta de gráficas vaciada.")

    # 4. Restauración del Backup original
    # Usamos .resolve() para evitar problemas de rutas relativas en Windows
    source_original = (base_path / "data" / "raw" / "original" / "Toshiba-2025-W27.csv").resolve()
    target_raw = (base_path / "data" / "raw" / "qkd" / "Toshiba-2025-W27.csv").resolve()

    if source_original.exists():
        target_raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_original, target_raw)
        print(f"[OK] Dataset 2025 restaurado con éxito desde: {source_original.name}")
    else:
        print(f"[CRITICAL] Error: No se encuentra el backup en {source_original}")
        print("Asegúrate de que el archivo existe físicamente en esa carpeta.")

    print("\n [SUCCESS] Entorno listo para la nueva ejecución. \n")


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
    #run_r_script()
    
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