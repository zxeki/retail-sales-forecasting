"""Load the Kaggle competition CSV files from ``data/raw``."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

from .config import KAGGLE_COMPETITION, KAGGLE_DATA_URL, RAW_DATA_DIR
from .datasets import DATASET_NAMES, DATASET_SPECS, DatasetSpec


class DatasetNotFoundError(FileNotFoundError):
    """Raised when an expected competition CSV file is missing."""

    def __init__(self, name: str, expected_path: Path) -> None:
        self.name = name
        self.expected_path = expected_path
        super().__init__(build_missing_dataset_message(name, expected_path))


def build_missing_dataset_message(name: str, expected_path: Path) -> str:
    """Build an actionable error message for a missing dataset file."""
    data_dir = expected_path.parent
    return (
        f"Dataset '{name}' was not found.\n"
        f"Expected file: {expected_path}\n\n"
        "Download the competition files from:\n"
        f"  {KAGGLE_DATA_URL}\n\n"
        "Manual download:\n"
        "  1. Open the URL above and accept the competition rules.\n"
        "  2. Download and extract the ZIP archive.\n"
        "  3. Copy these files into the data/raw directory:\n"
        "       train.csv, test.csv, stores.csv, oil.csv,\n"
        "       holidays_events.csv, transactions.csv\n"
        f"     Target directory: {data_dir}\n\n"
        "Kaggle CLI (optional):\n"
        "  pip install kaggle\n"
        f"  kaggle competitions download -c {KAGGLE_COMPETITION} -p data/raw\n"
        "  PowerShell: Expand-Archive data/raw/"
        f"{KAGGLE_COMPETITION}.zip -DestinationPath data/raw\n"
        "  macOS/Linux: unzip data/raw/"
        f"{KAGGLE_COMPETITION}.zip -d data/raw"
    )


def _get_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        options = ", ".join(DATASET_NAMES)
        raise ValueError(
            f"Unknown dataset '{name}'. Choose one of: {options}."
        ) from exc


def _resolve_data_dir(data_dir: str | Path | None) -> Path:
    return Path(data_dir) if data_dir is not None else RAW_DATA_DIR


def dataset_path(name: str, data_dir: str | Path | None = None) -> Path:
    """Return the expected path of a competition CSV file."""
    spec = _get_spec(name)
    return _resolve_data_dir(data_dir) / spec.filename


def is_available(name: str, data_dir: str | Path | None = None) -> bool:
    """Return ``True`` when a competition CSV file exists."""
    return dataset_path(name, data_dir).is_file()


def available_datasets(data_dir: str | Path | None = None) -> list[str]:
    """List the known competition datasets that are present."""
    return [name for name in DATASET_NAMES if is_available(name, data_dir)]


def missing_datasets(data_dir: str | Path | None = None) -> list[str]:
    """List the known competition datasets that are missing."""
    return [name for name in DATASET_NAMES if not is_available(name, data_dir)]


def _parse_date_columns(
    dataframe: pd.DataFrame, spec: DatasetSpec
) -> pd.DataFrame:
    for column in spec.date_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_datetime(
                dataframe[column], errors="coerce"
            )
    return dataframe


def load_dataset(
    name: str,
    *,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load one competition CSV file and parse its date columns.

    Raises:
        DatasetNotFoundError: If the CSV file does not exist.
        ValueError: If ``name`` is not a known dataset.
    """
    spec = _get_spec(name)
    path = dataset_path(name, data_dir)

    if not path.is_file():
        raise DatasetNotFoundError(name, path)

    dataframe = pd.read_csv(path)
    return _parse_date_columns(dataframe, spec)


def load_datasets(
    names: Iterable[str] | None = None,
    *,
    data_dir: str | Path | None = None,
    skip_missing: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load multiple competition CSV files.

    Args:
        names: Dataset names to load. Defaults to all known datasets.
        data_dir: Directory containing the CSV files.
        skip_missing: When ``True``, silently skip missing files instead of
            raising :class:`DatasetNotFoundError`.
    """
    selected_names = list(names) if names is not None else list(DATASET_NAMES)
    loaded: dict[str, pd.DataFrame] = {}

    for name in selected_names:
        if skip_missing and not is_available(name, data_dir):
            continue
        loaded[name] = load_dataset(name, data_dir=data_dir)

    return loaded


def load_available_datasets(
    *,
    data_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load every competition CSV that is currently available."""
    return load_datasets(data_dir=data_dir, skip_missing=True)


def dataset_inventory(
    data_dir: str | Path | None = None,
) -> Mapping[str, bool]:
    """Return a mapping of dataset name to availability."""
    return {name: is_available(name, data_dir) for name in DATASET_NAMES}
