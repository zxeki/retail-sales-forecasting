"""Generate a Kaggle submission from the official test data.

Usage:
    python scripts/generate_submission.py [--output PATH] [--n-estimators N]

Trains the LightGBM model on the full training history, forecasts the official
test window, validates the format, and writes data/processed/submission.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_sales_forecasting.data_loader import (
    available_datasets,
    load_dataset,
)
from retail_sales_forecasting.submission import (
    DEFAULT_SUBMISSION_PATH,
    build_and_save_submission,
    validate_submission,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SUBMISSION_PATH,
        help="Where to write the submission CSV.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of LightGBM boosting rounds.",
    )
    args = parser.parse_args()

    required = {"train", "test"}
    missing = required - set(available_datasets())
    if missing:
        print(
            f"Missing required datasets: {', '.join(sorted(missing))}. "
            "Download the Kaggle dataset first (see README.md)."
        )
        return 1

    print("Training on full history and forecasting the test window...")
    result = build_and_save_submission(args.output, n_estimators=args.n_estimators)

    test = load_dataset("test")
    report = validate_submission(result.frame, test)

    print()
    print("Submission validation report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print()
    print(f"Saved submission to {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
