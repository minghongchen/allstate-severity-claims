"""
    Paths
"""

from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RETRAIN_DATA_DIR = DATA_DIR / "retrain"

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"

# MLflow results directory
MLFLOW_RESULT_DIR = PROJECT_ROOT / "mlruns"
