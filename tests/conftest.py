"""Shared pytest configuration and fixtures."""

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_sales_forecasting.datasets import DATASET_SPECS  # noqa: E402


def make_train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["2013-01-01", "2013-01-02", "2013-01-03"],
            "store_nbr": [1, 1, 2],
            "family": ["GROCERY", "GROCERY", "BEVERAGES"],
            "sales": [10.0, 20.0, 30.0],
            "onpromotion": [0, 1, 0],
        }
    )


def make_test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [10, 11],
            "date": ["2013-01-04", "2013-01-05"],
            "store_nbr": [1, 2],
            "family": ["GROCERY", "BEVERAGES"],
            "onpromotion": [0, 0],
        }
    )


def make_stores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["D", "B"],
            "cluster": [1, 2],
        }
    )


def make_oil_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02"],
            "dcoilwtico": [93.14, 92.57],
        }
    )


def make_holidays_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2013-01-01"],
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["New Year"],
            "transferred": [False],
        }
    )


def make_transactions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02"],
            "store_nbr": [1, 2],
            "transactions": [100, 200],
        }
    )


@pytest.fixture
def raw_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data/raw directory with small CSV samples."""
    frames = {
        "train": make_train_frame(),
        "test": make_test_frame(),
        "stores": make_stores_frame(),
        "oil": make_oil_frame(),
        "holidays_events": make_holidays_frame(),
        "transactions": make_transactions_frame(),
    }

    for name, frame in frames.items():
        frame.to_csv(tmp_path / DATASET_SPECS[name].filename, index=False)

    return tmp_path
