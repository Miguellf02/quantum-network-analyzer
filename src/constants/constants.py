"""
CONSTANTS MODULE: Central repository for all project configurations, 
paths, column names, and model parameters.
"""

from pathlib import Path


# 1. PATH CONFIGURATION 
BASE_DIR = Path(__file__).resolve().parents[2] 

# Input/Output Directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Specific Processed Subdirectories
PREPROCESSING_OUTPUT_DIR = PROCESSED_DIR / "python_preprocessing"
FEATURE_ENGINEERING_OUTPUT_DIR = PROCESSED_DIR / "feature_engineered"
MODEL_OUTPUT_DIR = PROCESSED_DIR / "models" 

# File Names
RAW_FILE_PATTERN = "*.csv"
PROCESSED_FILE_NAME = "QKD_PROCESSED.csv"
FEATURED_FILE_NAME = "QKD_FEATURES.csv"

# Full Paths for Inputs/Outputs
INPUT_PROCESSED_PATH = PREPROCESSING_OUTPUT_DIR / PROCESSED_FILE_NAME
INPUT_FEATURED_PATH = FEATURE_ENGINEERING_OUTPUT_DIR / FEATURED_FILE_NAME

# 2. SCRIPT PATHS AND MODULES

# Absolute Path for R executable 
RSCRIPT_EXE_PATH = r"C:\Program Files\R\R-4.5.1\bin\Rscript.exe"

# Absolute Path for R script
R_SCRIPT_PATH = BASE_DIR / "src" / "R" / "Primer_Analisis.R"

# Python Modules for use with 'python -m' 
PYTHON_MODULE_PREPROCESSING = "src.python.preprocessing"
PYTHON_MODULE_FEATURE_ENG = "src.python.feature_engineering"
PYTHON_MODULE_IFOREST = "src.python.train_iforest"

# Absolute Paths for Python scripts 
PY_SCRIPT_PREPROCESSING_PATH = BASE_DIR / "src" / "python" / "preprocessing.py"
PY_SCRIPT_FEATURE_ENG_PATH = BASE_DIR / "src" / "python" / "feature_engineering.py"