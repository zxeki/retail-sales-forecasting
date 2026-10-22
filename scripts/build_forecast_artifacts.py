"""Build and save the cached forecast artifacts used by the dashboard.

Usage:
    python scripts/build_forecast_artifacts.py

Trains the LightGBM model on the leakage-safe features, scores it against the
seasonal-naive baseline, and writes the predictions and metrics into
data/processed/. The Streamlit dashboard loads these files instead of
retraining on every visit.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from retail_sales_forecasting.data_loader import available_datasets
from retail_sales_forecasting.forecast import (
    FORECAST_ARTIFACT,
    METRICS_ARTIFACT,
    build_forecast_run,
    save_forecast_run,
)


def main() -> int:
    if "train" not in available_datasets():
        print(
            "train.csv was not found in data/raw/. Download the Kaggle dataset "
            "first (see README.md)."
        )
        return 1

    print("Running the forecasting pipeline...")
    run = build_forecast_run()
    save_forecast_run(run)

    print(f"Cutoff date:       {run.cutoff.date()}")
    print(f"Validation window: {run.date_min.date()} to {run.date_max.date()}")
    print(f"Validation rows:   {len(run.predictions):,}")
    print()
    print(run.metrics.to_string(index=False))
    print()
    print(f"Saved predictions: {FORECAST_ARTIFACT}")
    print(f"Saved metrics:     {METRICS_ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
