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

# EXECUTION FUNCTIONS


def run_anomaly_injection():
    """Ejecuta la creación de escenarios sintéticos como un proceso independiente."""
    print("\n [INFO] GENERATING SYNTHETIC ATTACK SCENARIOS \n")
    
    # Ruta al script de inyección
    injector_path = PROJECT_BASE_DIR / "src" / "python" / "demostradores" / "inject_drift.py"
    
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