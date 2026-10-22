"""Shared project paths and constants."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

KAGGLE_COMPETITION = "store-sales-time-series-forecasting"
KAGGLE_DATA_URL = (
    "https://www.kaggle.com/competitions/"
    "store-sales-time-series-forecasting/data"
)
