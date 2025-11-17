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

# SCRIPT PYTHON
PY_SCRIPT = BASE_DIR / "src" / "python"/ "preprocessing.py"

def run_r_script():
    print("\n EXECUTING R SCRIPT \n")
    subprocess.run([RSCRIPT_EXE, str(R_SCRIPT)], check=True)

def run_python_preprocessing():
    print("\n EXECUTING PYTHON PREPROCESSING \n")
    subprocess.run(
        [sys.executable, str(PY_SCRIPT)],
        check=True,
        cwd=BASE_DIR  
    )
def main():
    run_r_script()
    run_python_preprocessing()
    print("\nTHE PIPELINE HAS FINISHED SATISFACTORY\n")

if __name__ == "__main__":
    main()
