"""Kaggle submission generation and format validation.

Trains the selected model on the full training history, forecasts the official
test window, and writes a submission CSV that preserves the required columns
(`id`, `sales`) and the original test-row order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR
from .data_loader import load_available_datasets
from .features import build_feature_frame
from .models import DEFAULT_FEATURES, predict, train_lightgbm

SUBMISSION_COLUMNS = ("id", "sales")
DEFAULT_SUBMISSION_PATH = PROCESSED_DATA_DIR / "submission.csv"


@dataclass
class SubmissionResult:
    """A generated submission and its validation report."""

    frame: pd.DataFrame
    path: Path | None = None


def build_submission_frame(test: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Assemble the submission frame in the original test row order."""
    if len(test) != len(predictions):
        raise ValueError(
            "Prediction count does not match the number of test rows: "
            f"{len(predictions)} vs {len(test)}."
        )
    frame = pd.DataFrame(
        {"id": test["id"].to_numpy(), "sales": np.asarray(predictions, dtype=float)}
    )
    return frame.reset_index(drop=True)


def validate_submission(
    frame: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Validate submission shape, columns, ordering, and value sanity.

    Returns a report dict. Raises ``ValueError`` on a structural problem so the
    caller fails loudly instead of uploading a malformed file.
    """
    if list(frame.columns) != list(SUBMISSION_COLUMNS):
        raise ValueError(
            f"Submission columns must be {list(SUBMISSION_COLUMNS)}, "
            f"got {list(frame.columns)}."
        )
    if len(frame) != len(test):
        raise ValueError(
            f"Submission has {len(frame)} rows but test has {len(test)}."
        )
    if not frame["id"].equals(test["id"].reset_index(drop=True)):
        raise ValueError("Submission id column does not match test row order.")
    if frame["id"].duplicated().any():
        raise ValueError("Submission contains duplicate ids.")
    if frame["sales"].isna().any():
        raise ValueError("Submission contains missing sales values.")
    negative = int((frame["sales"] < 0).sum())

    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "id_order_matches_test": bool(
            frame["id"].equals(test["id"].reset_index(drop=True))
        ),
        "duplicate_ids": int(frame["id"].duplicated().sum()),
        "missing_sales": int(frame["sales"].isna().sum()),
        "negative_sales": negative,
        "min_sales": float(frame["sales"].min()),
        "max_sales": float(frame["sales"].max()),
        "mean_sales": float(frame["sales"].mean()),
    }


def _recursive_test_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model,
    columns: list[str],
    national_holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Forecast the test horizon one day at a time, feeding predictions back.

    Lags and rolling means for later test days depend on earlier test sales,
    which are unknown. This walks the test dates in order, predicts each date
    from the history accumulated so far, then appends those predictions as
    "sales" so the next date sees them. No future information is ever used.
    """
    working = train[["date", "store_nbr", "family", "sales", "onpromotion"]].copy()
    working["id"] = pd.NA
    test_sorted = test.sort_values(["date", "store_nbr", "family"])
    results = []

    for date in sorted(test["date"].unique()):
        day_test = test_sorted[test_sorted["date"] == date].copy()
        day_test["sales"] = np.nan
        combined = pd.concat([working, day_test], ignore_index=True, sort=False)
        featured = build_feature_frame(
            combined, country_holidays=national_holidays
        )
        day_features = featured[featured["date"] == date].copy()
        day_predictions = predict(model, day_features, columns)
        day_features = day_features.assign(prediction=day_predictions)

        results.append(day_features[["id", "prediction"]])

        # Feed the predictions back as observed sales for the next iteration.
        working = pd.concat(
            [
                working,
                day_features[["date", "store_nbr", "family", "onpromotion", "id"]].assign(
                    sales=day_predictions
                ),
            ],
            ignore_index=True,
            sort=False,
        )

    return pd.concat(results, ignore_index=True)


def generate_submission(
    train: pd.DataFrame,
    test: pd.DataFrame,
    holidays: pd.DataFrame,
    *,
    n_estimators: int = 300,
) -> SubmissionResult:
    """Train on the full history and forecast the official test window.

    The model is trained only on rows with known sales. Test predictions are
    produced recursively, one day at a time, so lag and rolling features for
    later test days use earlier predictions instead of missing values.
    """
    national_holidays = holidays[holidays["locale"] == "National"]

    training_frame = build_feature_frame(
        train, country_holidays=national_holidays
    )
    model, columns = train_lightgbm(
        training_frame, features=DEFAULT_FEATURES, n_estimators=n_estimators
    )

    predicted = _recursive_test_features(
        train, test, model, columns, national_holidays
    )

    # Reindex predictions back to the official test order by id.
    ordered = test[["id"]].merge(predicted, on="id", how="left")
    if ordered["prediction"].isna().any():
        raise ValueError("Some test ids did not receive a prediction.")

    frame = build_submission_frame(test, ordered["prediction"].to_numpy())
    validate_submission(frame, test)
    return SubmissionResult(frame=frame)


def save_submission(
    result: SubmissionResult, path: Path | None = None
) -> Path:
    """Write the submission frame to CSV."""
    target = Path(path) if path is not None else DEFAULT_SUBMISSION_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(target, index=False)
    result.path = target
    return target


def build_and_save_submission(
    path: Path | None = None, *, n_estimators: int = 300
) -> SubmissionResult:
    """Convenience wrapper: load data, generate, and save a submission."""
    datasets = load_available_datasets()
    result = generate_submission(
        datasets["train"],
        datasets["test"],
        datasets["holidays_events"],
        n_estimators=n_estimators,
    )
    save_submission(result, path)
    return result
