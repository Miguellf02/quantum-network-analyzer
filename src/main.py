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

# MAIN PIPELINE EXECUTION

def main():
    
    # 1. Ejecutar análisis R inicial (opcional)
    # run_r_script()
    
    # 2. Ejecutar Preprocesamiento Python
    run_python_preprocessing()
    
    # 3. Ejecutar Ingeniería de Características
    run_python_feature_engineering()
    
    print("\nTHE PIPELINE HAS FINISHED SATISFACTORILY\n")

if __name__ == "__main__":
    main()