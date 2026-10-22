"""Data preparation helpers for the Streamlit dashboard.

These functions turn the raw datasets and the cached forecast run into small,
tidy frames and summaries so the app stays thin and the logic stays testable.
"""

from __future__ import annotations

import pandas as pd

from .forecast import ForecastRun


def filter_options(predictions: pd.DataFrame) -> dict:
    """Return the filter choices available in a forecast run."""
    return {
        "families": sorted(predictions["family"].unique().tolist()),
        "stores": sorted(int(store) for store in predictions["store_nbr"].unique()),
        "date_min": predictions["date"].min(),
        "date_max": predictions["date"].max(),
    }


def filter_predictions(
    predictions: pd.DataFrame,
    *,
    families: list[str] | None = None,
    stores: list[int] | None = None,
    date_range: tuple | None = None,
) -> pd.DataFrame:
    """Apply dashboard filters to the forecast predictions frame."""
    filtered = predictions.copy()
    if families:
        filtered = filtered[filtered["family"].isin(families)]
    if stores:
        filtered = filtered[filtered["store_nbr"].isin(stores)]
    if date_range is not None and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[
            (filtered["date"] >= pd.Timestamp(start))
            & (filtered["date"] <= pd.Timestamp(end))
        ]
    return filtered.reset_index(drop=True)


def daily_totals(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily actual, forecast, and baseline totals.

    Shows how the daily forecast tracks realized totals across the window.
    """
    daily = (
        predictions.groupby("date", as_index=False)
        .agg(actual=("sales", "sum"), forecast=("prediction", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    if "baseline" in predictions.columns:
        baseline = (
            predictions.groupby("date", as_index=False)["baseline"]
            .sum()
            .rename(columns={"baseline": "baseline"})
        )
        daily = daily.merge(baseline, on="date", how="left")
    return daily


def family_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate error and sales by family for the selected slice."""
    summary = (
        predictions.groupby("family", as_index=False)
        .agg(
            actual=("sales", "sum"),
            forecast=("prediction", "sum"),
            mean_abs_error=("abs_error", "mean"),
            rows=("abs_error", "size"),
        )
        .sort_values("mean_abs_error", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def store_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate error and sales by store for the selected slice."""
    return (
        predictions.groupby("store_nbr", as_index=False)
        .agg(
            actual=("sales", "sum"),
            forecast=("prediction", "sum"),
            mean_abs_error=("abs_error", "mean"),
            rows=("abs_error", "size"),
        )
        .sort_values("mean_abs_error", ascending=False)
        .reset_index(drop=True)
    )


def slice_metrics(predictions: pd.DataFrame) -> dict:
    """Compute RMSLE, MAE, and RMSE for a filtered slice (actual vs forecast)."""
    from .evaluation import (
        mean_absolute_error,
        root_mean_squared_error,
        root_mean_squared_log_error,
    )

    actual = predictions["sales"]
    forecast = predictions["prediction"]
    return {
        "rmsle": root_mean_squared_log_error(actual, forecast),
        "mae": mean_absolute_error(actual, forecast),
        "rmse": root_mean_squared_error(actual, forecast),
        "rows": int(len(predictions)),
    }


def metric_comparison(run: ForecastRun) -> pd.DataFrame:
    """Return the model-vs-baseline metrics table from a run."""
    return run.metrics.copy()
