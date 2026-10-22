"""Forecast evaluation metrics and a seasonal-naive baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_error(y_true, y_pred) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def root_mean_squared_log_error(y_true, y_pred) -> float:
    """Root mean squared logarithmic error (Kaggle competition metric).

    Negative predictions are clipped to zero because the log is undefined
    for negative values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_true < 0):
        raise ValueError("y_true contains negative values.")
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def evaluate_forecast(
    y_true, y_pred, *, prefix: str = ""
) -> dict[str, float]:
    """Compute MAE, RMSE, and RMSLE in one call."""
    return {
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}rmse": root_mean_squared_error(y_true, y_pred),
        f"{prefix}rmsle": root_mean_squared_log_error(y_true, y_pred),
    }


def seasonal_naive_forecast(
    train: pd.DataFrame,
    target_dates,
    *,
    date_column: str = "date",
    target_column: str = "sales",
    group_columns: tuple[str, ...] = ("store_nbr", "family"),
    season_length: int = 7,
    use_weekday_profile: bool = True,
) -> pd.DataFrame:
    """Seasonal-naive baseline for a weekly seasonal cycle.

    Two leakage-safe variants are supported:

    - ``use_weekday_profile=True`` (default): predict each future date with
      the mean historical sales for that group on the same day of week.
    - ``use_weekday_profile=False``: classic seasonal naive, repeating the
      last ``season_length`` daily values.

    Only data up to the last date in ``train`` is used.

    Returns a DataFrame with ``group_columns`` + ``date`` + ``prediction``.
    """
    target_dates = pd.DatetimeIndex(pd.to_datetime(target_dates)).normalize()
    frame = train.copy()
    frame[date_column] = pd.to_datetime(frame[date_column]).dt.normalize()

    rows = []

    if use_weekday_profile:
        frame["_dow"] = frame[date_column].dt.dayofweek
        profiles = (
            frame.groupby([*group_columns, "_dow"])[target_column]
            .mean()
            .to_dict()
        )
        group_means = frame.groupby(list(group_columns))[target_column].mean().to_dict()
        for keys, target_date in (
            (keys, date)
            for keys in group_means
            for date in target_dates
        ):
            dow = target_date.dayofweek
            value = profiles.get((*keys, dow))
            if value is None:
                value = group_means.get(keys, 0.0)
            rows.append((*keys, target_date, float(value)))
    else:
        daily = (
            frame.pivot_table(
                index=list(group_columns),
                columns=date_column,
                values=target_column,
                aggfunc="sum",
            )
            .sort_index(axis=1)
        )
        for keys, series in daily.iterrows():
            if not isinstance(keys, tuple):
                keys = (keys,)
            latest = series.dropna()
            if latest.empty:
                pattern = [0.0]
            else:
                pattern = list(latest.iloc[-season_length:].to_numpy())
                if not pattern:
                    pattern = [float(latest.mean())]
            for offset, target_date in enumerate(target_dates):
                value = pattern[offset % len(pattern)]
                rows.append((*keys, target_date, float(value)))

    result = pd.DataFrame(
        rows, columns=[*group_columns, date_column, "prediction"]
    )
    return result
