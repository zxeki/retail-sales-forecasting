"""Leakage-safe feature engineering for forecasting.

All features are computed from information available up to each row's date.
Lag and rolling features are shifted so a row never sees its own or future
sales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_COLUMNS = ("store_nbr", "family")


def add_calendar_features(
    data: pd.DataFrame,
    *,
    date_column: str = "date",
    country_holidays: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add calendar features derived from the date.

    Calendar features are safe because they are known in advance.
    """
    frame = data.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])

    frame["year"] = frame[date_column].dt.year
    frame["month"] = frame[date_column].dt.month
    frame["day"] = frame[date_column].dt.day
    frame["day_of_week"] = frame[date_column].dt.dayofweek
    frame["day_of_year"] = frame[date_column].dt.dayofyear
    frame["week_of_year"] = frame[date_column].dt.isocalendar().week.astype(int)
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)

    if country_holidays is not None and not country_holidays.empty:
        holidays = country_holidays.copy()
        holidays["date"] = pd.to_datetime(holidays["date"]).dt.normalize()
        holiday_dates = set(holidays["date"])
        frame["is_holiday"] = (
            frame[date_column].dt.normalize().isin(holiday_dates).astype(int)
        )
    else:
        frame["is_holiday"] = 0

    return frame


def add_promotion_features(
    data: pd.DataFrame,
    *,
    promotion_column: str = "onpromotion",
) -> pd.DataFrame:
    """Add a binary promotion flag from the promotion count.

    Only uses the current row's promotion value, which is known at
    prediction time, so it is leakage-free.
    """
    frame = data.copy()
    if promotion_column in frame.columns:
        frame["is_promoted"] = (frame[promotion_column] >= 1).astype(int)
    else:
        frame["is_promoted"] = 0
    return frame


def add_lag_features(
    data: pd.DataFrame,
    *,
    date_column: str = "date",
    target_column: str = "sales",
    group_columns: tuple[str, ...] = GROUP_COLUMNS,
    lags: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    """Add lagged sales per group.

    Each lag is shifted within its group, so a row only sees past sales.
    """
    frame = data.sort_values(list(group_columns) + [date_column]).copy()
    grouped = frame.groupby(list(group_columns))[target_column]
    for lag in lags:
        frame[f"lag_{lag}"] = grouped.shift(lag)
    return frame


def add_rolling_features(
    data: pd.DataFrame,
    *,
    date_column: str = "date",
    target_column: str = "sales",
    group_columns: tuple[str, ...] = GROUP_COLUMNS,
    windows: tuple[int, ...] = (7, 28),
) -> pd.DataFrame:
    """Add rolling-mean features of past sales per group.

    The series is shifted by one day before rolling so the current day's
    sales never contribute to its own features.
    """
    frame = data.sort_values(list(group_columns) + [date_column]).copy()
    grouped = frame.groupby(list(group_columns))[target_column]
    shifted = grouped.shift(1)
    for window in windows:
        frame[f"rolling_mean_{window}"] = (
            shifted.groupby([frame[col] for col in group_columns])
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(group_columns))), drop=True)
        )
    return frame


def build_feature_frame(
    data: pd.DataFrame,
    *,
    country_holidays: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply calendar, promotion, lag, and rolling features in one call."""
    frame = add_calendar_features(data, country_holidays=country_holidays)
    frame = add_promotion_features(frame)
    frame = add_lag_features(frame)
    frame = add_rolling_features(frame)
    return frame
