"""End-to-end forecasting pipeline and cached dashboard artifacts.

Ties together the chronological split, leakage-safe features, candidate models,
and evaluation into a single reproducible run. The dashboard consumes the
resulting tidy frames instead of retraining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROCESSED_DATA_DIR
from .data_loader import load_available_datasets
from .evaluation import (
    evaluate_forecast,
    seasonal_naive_forecast,
)
from .features import build_feature_frame
from .models import DEFAULT_FEATURES, predict, train_lightgbm
from .splits import chronological_split

FORECAST_ARTIFACT = PROCESSED_DATA_DIR / "forecast_validation.csv"
METRICS_ARTIFACT = PROCESSED_DATA_DIR / "forecast_metrics.json"
MODEL_NAME = "LightGBM"


@dataclass
class ForecastRun:
    """Outputs of a single forecasting run on the validation window."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    cutoff: pd.Timestamp
    model_name: str = MODEL_NAME

    @property
    def date_min(self) -> pd.Timestamp:
        return self.predictions["date"].min()

    @property
    def date_max(self) -> pd.Timestamp:
        return self.predictions["date"].max()


def _score(validation: pd.DataFrame, predictions: np.ndarray) -> dict:
    metrics = evaluate_forecast(validation["sales"], predictions)
    return {key: float(value) for key, value in metrics.items()}


def run_forecast(
    train: pd.DataFrame,
    holidays: pd.DataFrame,
    *,
    validation_days: int = 16,
) -> ForecastRun:
    """Train the selected model and score it against the seasonal-naive baseline.

    Returns validation rows with actual sales, model predictions, the baseline,
    and per-row absolute error, plus a metrics table for both approaches.
    """
    split = chronological_split(train, validation_days=validation_days)
    national_holidays = holidays[holidays["locale"] == "National"]
    featured = build_feature_frame(train, country_holidays=national_holidays)

    feature_train = featured[featured["date"] <= split.cutoff]
    feature_validation = featured[featured["date"] > split.cutoff]

    model, columns = train_lightgbm(feature_train, features=DEFAULT_FEATURES)
    model_predictions = predict(model, feature_validation, columns)

    target_dates = sorted(feature_validation["date"].unique())
    baseline = seasonal_naive_forecast(
        feature_train, target_dates, use_weekday_profile=False
    )

    scored = feature_validation[
        ["date", "store_nbr", "family", "sales"]
    ].copy()
    scored["prediction"] = model_predictions
    scored = scored.merge(
        baseline.rename(columns={"prediction": "baseline"}),
        on=["store_nbr", "family", "date"],
        how="left",
    )
    scored["baseline"] = scored["baseline"].fillna(0.0)
    scored["abs_error"] = (scored["sales"] - scored["prediction"]).abs()

    metrics = pd.DataFrame(
        [
            {"model": MODEL_NAME, **_score(feature_validation, model_predictions)},
            {
                "model": "Seasonal naive (last week)",
                **_score(feature_validation, scored["baseline"].to_numpy()),
            },
        ]
    ).sort_values("rmsle").reset_index(drop=True)

    return ForecastRun(
        predictions=scored.sort_values(["date", "store_nbr", "family"]).reset_index(
            drop=True
        ),
        metrics=metrics,
        cutoff=split.cutoff,
    )


def build_forecast_run(
    make_dirs: bool = True,
) -> ForecastRun:
    """Load the raw data and run the full forecasting pipeline."""
    if make_dirs:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_available_datasets()
    return run_forecast(datasets["train"], datasets["holidays_events"])


def save_forecast_run(run: ForecastRun) -> None:
    """Persist predictions and metrics so the dashboard can load them fast."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    run.predictions.to_csv(FORECAST_ARTIFACT, index=False)
    METRICS_ARTIFACT.write_text(
        json.dumps(
            {
                "model_name": run.model_name,
                "cutoff": run.cutoff.strftime("%Y-%m-%d"),
                "metrics": run.metrics.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_forecast_run() -> ForecastRun:
    """Load a saved forecast run, or build and save one if it is missing."""
    if FORECAST_ARTIFACT.exists() and METRICS_ARTIFACT.exists():
        predictions = pd.read_csv(FORECAST_ARTIFACT, parse_dates=["date"])
        payload = json.loads(METRICS_ARTIFACT.read_text(encoding="utf-8"))
        metrics = pd.DataFrame(payload["metrics"])
        return ForecastRun(
            predictions=predictions,
            metrics=metrics,
            cutoff=pd.Timestamp(payload["cutoff"]),
            model_name=payload.get("model_name", MODEL_NAME),
        )

    run = build_forecast_run()
    save_forecast_run(run)
    return run


def has_forecast_artifact() -> bool:
    """Whether a cached forecast run exists on disk."""
    return FORECAST_ARTIFACT.exists() and METRICS_ARTIFACT.exists()
