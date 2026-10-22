"""Download the Kaggle competition files into ``data/raw``."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
COMPETITION = "store-sales-time-series-forecasting"
DATASET_FILES = (
    "train.csv",
    "test.csv",
    "stores.csv",
    "oil.csv",
    "holidays_events.csv",
    "transactions.csv",
)


def copy_competition_files(
    source_dir: str | Path, target_dir: str | Path
) -> tuple[list[str], list[str]]:
    """Copy expected CSVs from a downloaded folder into ``target_dir``."""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []

    for filename in DATASET_FILES:
        source_file = source_path / filename
        if source_file.is_file():
            shutil.copy2(source_file, target_path / filename)
            copied.append(filename)
        else:
            missing.append(filename)

    return copied, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the Store Sales competition data with kagglehub."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Destination directory for the CSV files.",
    )
    args = parser.parse_args()

    print(f"Downloading competition: {COMPETITION}")
    downloaded_dir = Path(kagglehub.competition_download(COMPETITION))
    print(f"Kagglehub cache path: {downloaded_dir}")

    copied, missing = copy_competition_files(downloaded_dir, args.data_dir)
    print(f"Destination: {Path(args.data_dir).resolve()}")

    if copied:
        print("Copied: " + ", ".join(copied))
    if missing:
        print("Missing from downloaded folder: " + ", ".join(missing))
        print("Check that you accepted the competition rules on Kaggle.")
        return 1

    print("Dataset download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
