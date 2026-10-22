"""Reusable exploratory data analysis helpers.

These functions return tidy DataFrames so they can be tested and reused by
notebooks or later pipeline steps. They do not mutate their inputs.
"""

from __future__ import annotations

import pandas as pd

_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def daily_sales(train: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total sales by date."""
    daily = (
        train.groupby("date", as_index=False)["sales"].sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily


def sales_by_weekday(train: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean daily sales by day of week (0=Monday)."""
    daily = daily_sales(train)
    daily["weekday"] = daily["date"].dt.dayofweek
    grouped = (
        daily.groupby("weekday")["sales"].mean().reindex(range(7)).reset_index()
    )
    grouped["weekday_name"] = grouped["weekday"].map(
        dict(enumerate(_WEEKDAY_NAMES))
    )
    return grouped


def sales_by_month(train: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean daily sales by calendar month (1=January)."""
    daily = daily_sales(train)
    daily["month"] = daily["date"].dt.month
    grouped = (
        daily.groupby("month")["sales"].mean().reindex(range(1, 13)).reset_index()
    )
    grouped["month_name"] = grouped["month"].map(
        dict(enumerate(_MONTH_NAMES, start=1))
    )
    return grouped


def monthly_sales_trend(train: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total sales by year-month for a trend line."""
    trend = (
        train.assign(month=train["date"].dt.to_period("M"))
        .groupby("month", as_index=False)["sales"]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )
    trend["month"] = trend["month"].dt.to_timestamp()
    return trend


def yearly_sales_trend(train: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total sales by calendar year."""
    trend = (
        train.assign(year=train["date"].dt.year)
        .groupby("year", as_index=False)["sales"]
        .sum()
        .sort_values("year")
        .reset_index(drop=True)
    )
    return trend


def sales_by_store(
    train: pd.DataFrame, stores: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Aggregate total and daily-mean sales per store, optionally with metadata."""
    aggregated = (
        train.groupby("store_nbr")
        .agg(total_sales=("sales", "sum"), mean_daily_sales=("sales", "mean"))
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .reset_index(drop=True)
    )
    if stores is not None:
        aggregated = aggregated.merge(stores, on="store_nbr", how="left")
    return aggregated


def sales_by_family(train: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total and daily-mean sales per product family."""
    return (
        train.groupby("family")
        .agg(
            total_sales=("sales", "sum"),
            mean_daily_sales=("sales", "mean"),
            zero_sales_days=("sales", lambda s: int((s == 0).sum())),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .reset_index(drop=True)
    )


def promotion_summary(train: pd.DataFrame, threshold: int = 1) -> dict:
    """Summarise sales for promoted vs non-promoted rows.

    ``threshold`` is the minimum ``onpromotion`` value counted as promoted.
    """
    promoted_mask = train["onpromotion"] >= threshold
    promoted = train.loc[promoted_mask, "sales"]
    regular = train.loc[~promoted_mask, "sales"]
    total = len(train)

    return {
        "threshold": threshold,
        "promoted_rows": int(promoted_mask.sum()),
        "promoted_share": float(promoted_mask.sum() / total) if total else 0.0,
        "mean_sales_promoted": float(promoted.mean()) if len(promoted) else 0.0,
        "mean_sales_regular": float(regular.mean()) if len(regular) else 0.0,
        "median_sales_promoted": (
            float(promoted.median()) if len(promoted) else 0.0
        ),
        "median_sales_regular": (
            float(regular.median()) if len(regular) else 0.0
        ),
    }


def holidays_by_type(holidays: pd.DataFrame) -> pd.DataFrame:
    """Count holiday/event rows by type."""
    return (
        holidays.groupby("type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def transactions_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction totals and daily means per store."""
    transactions = transactions.assign(
        month=transactions["date"].dt.to_period("M")
    )
    daily = (
        transactions.groupby(["store_nbr", transactions["date"]])["transactions"]
        .sum()
        .reset_index()
    )
    return (
        daily.groupby("store_nbr")["transactions"]
        .agg(
            total_transactions="sum",
            mean_daily_transactions="mean",
        )
        .reset_index()
        .sort_values("total_transactions", ascending=False)
        .reset_index(drop=True)
    )


def oil_price_trend(oil: pd.DataFrame) -> pd.DataFrame:
    """Return the oil price series sorted by date."""
    return oil.sort_values("date").reset_index(drop=True)


def merge_oil_with_sales(
    train: pd.DataFrame, oil: pd.DataFrame
) -> pd.DataFrame:
    """Join daily total sales with the forward-filled oil price."""
    daily = daily_sales(train)
    oil_series = (
        oil.sort_values("date")[["date", "dcoilwtico"]]
        .drop_duplicates("date")
        .assign(dcoilwtico=lambda df: df["dcoilwtico"].ffill())
    )
    merged = daily.merge(oil_series, on="date", how="left")
    merged["dcoilwtico"] = merged["dcoilwtico"].ffill().bfill()
    return merged


def missing_value_report(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Report non-zero missing counts across several datasets."""
    rows = []
    for name, frame in datasets.items():
        counts = frame.isna().sum()
        for column, count in counts[counts > 0].items():
            rows.append(
                {
                    "dataset": name,
                    "column": column,
                    "missing_count": int(count),
                    "missing_share": float(count / len(frame)) if len(frame) else 0.0,
                }
            )
    return pd.DataFrame(rows, columns=[
        "dataset",
        "column",
        "missing_count",
        "missing_share",
    ])
