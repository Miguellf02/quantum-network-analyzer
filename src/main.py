"""
MAIN PIPELINE – QKD DATA ANALYZER
"""

import subprocess
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent 

# RUTA CORRECTA AL SCRIPT .R EN TU PROYECTO
R_SCRIPT = BASE_DIR / "src" / "R" / "Primer_Analisis.R"

# RUTA AL EJECUTABLE DE Rscript.exe
RSCRIPT_EXE = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" 

# SCRIPTS PYTHON
PY_SCRIPT_PREPROCESSING = BASE_DIR / "src" / "python"/ "preprocessing.py"
PY_SCRIPT_FEATURE_ENG = BASE_DIR / "src" / "python"/ "feature_engineering.py" # ¡NUEVA RUTA!

def run_r_script():
    print("\n EXECUTING R SCRIPT \n")
    # Nota: Usamos str() para asegurar la compatibilidad con subprocess si RSCRIPT_EXE lo necesita
    subprocess.run([RSCRIPT_EXE, str(R_SCRIPT)], check=True)

def run_python_preprocessing():
    print("\n EXECUTING PYTHON PREPROCESSING \n")
    subprocess.run(
        [sys.executable, str(PY_SCRIPT_PREPROCESSING)], # Usamos la nueva variable
        check=True,
        cwd=BASE_DIR 
    )

def run_python_feature_engineering():
    print("\n EXECUTING PYTHON FEATURE ENGINEERING \n")
    # Ejecutamos el script de feature engineering con el intérprete de Python
    subprocess.run(
        [sys.executable, str(PY_SCRIPT_FEATURE_ENG)],
        check=True,
        cwd=BASE_DIR 
    )
    
def main():
    # 1. Ejecutar análisis R inicial (opcional)
    # run_r_script() # Si ya no es necesario, puedes comentarlo
    
    # 2. Ejecutar Preprocesamiento Python (Limpieza/Unificación)
    run_python_preprocessing()
    
    # 3. Ejecutar Ingeniería de Características (Preparación para ML) - ¡NUEVA LLAMADA!
    run_python_feature_engineering()
    
    print("\nTHE PIPELINE HAS FINISHED SATISFACTORILY\n")

if __name__ == "__main__":
    main()