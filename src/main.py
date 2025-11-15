# src/main.py

import sys
from pathlib import Path

## We make sure that the 'python' directory is in the path
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from python.preprocessing import main as preprocessing_main

if __name__ == "__main__":
    preprocessing_main()
