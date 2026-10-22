"""Tests for the exploratory analysis helpers."""

import pandas as pd
import pytest

from retail_sales_forecasting.analysis import (
    daily_sales,
    holidays_by_type,
    merge_oil_with_sales,
    missing_value_report,
    monthly_sales_trend,
    oil_price_trend,
    promotion_summary,
    sales_by_family,
    sales_by_month,
    sales_by_store,
    sales_by_weekday,
    transactions_summary,
    yearly_sales_trend,
)


@pytest.fixture
def train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2013-01-01",  # Tuesday
                    "2013-01-01",
                    "2013-01-02",  # Wednesday
                    "2013-01-08",  # Tuesday
                ]
            ),
            "store_nbr": [1, 2, 1, 1],
            "family": ["A", "A", "B", "A"],
            "sales": [10.0, 20.0, 5.0, 25.0],
            "onpromotion": [0, 3, 1, 0],
        }
    )


def test_daily_sales_sums_by_date(train_frame):
    daily = daily_sales(train_frame)

    assert list(daily.columns) == ["date", "sales"]
    assert daily.loc[daily["date"] == "2013-01-01", "sales"].iloc[0] == 30.0
    assert len(daily) == 3
    assert daily["date"].is_monotonic_increasing


def test_sales_by_weekday_has_seven_rows(train_frame):
    result = sales_by_weekday(train_frame)

    assert len(result) == 7
    assert list(result["weekday"]) == list(range(7))
    assert result.loc[result["weekday"] == 1, "weekday_name"].iloc[0] == "Tuesday"
    assert result.loc[result["weekday"] == 1, "sales"].iloc[0] == pytest.approx(
        (30.0 + 25.0) / 2
    )


def test_sales_by_month_has_twelve_rows(train_frame):
    result = sales_by_month(train_frame)

    assert len(result) == 12
    assert result.loc[result["month"] == 1, "month_name"].iloc[0] == "January"
    assert result.loc[result["month"] == 1, "sales"].iloc[0] == pytest.approx(
        (30.0 + 5.0 + 25.0) / 3
    )


def test_monthly_and_yearly_trends(train_frame):
    monthly = monthly_sales_trend(train_frame)
    yearly = yearly_sales_trend(train_frame)

    assert len(monthly) == 1
    assert monthly["sales"].iloc[0] == 60.0
    assert len(yearly) == 1
    assert yearly.loc[0, "year"] == 2013
    assert yearly["sales"].iloc[0] == 60.0


def test_sales_by_store_orders_by_total(train_frame):
    stores = pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["D", "D"],
            "cluster": [1, 2],
        }
    )
    result = sales_by_store(train_frame, stores)

    assert list(result["store_nbr"]) == [1, 2]
    assert "city" in result.columns
    assert result.loc[0, "total_sales"] == 40.0
    assert result.loc[1, "total_sales"] == 20.0


def test_sales_by_family_counts_zero_days(train_frame):
    result = sales_by_family(train_frame)

    family_a = result.loc[result["family"] == "A"].iloc[0]
    assert family_a["total_sales"] == 55.0
    assert family_a["zero_sales_days"] == 0


def test_promotion_summary_uses_threshold(train_frame):
    summary = promotion_summary(train_frame, threshold=1)

    assert summary["promoted_rows"] == 2
    assert summary["promoted_share"] == pytest.approx(0.5)
    assert summary["mean_sales_promoted"] == pytest.approx((20.0 + 5.0) / 2)
    assert summary["mean_sales_regular"] == pytest.approx((10.0 + 25.0) / 2)


def test_holidays_by_type_counts_rows():
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"]),
            "type": ["Holiday", "Holiday", "Event"],
        }
    )
    result = holidays_by_type(holidays)

    assert result.iloc[0]["type"] == "Holiday"
    assert result.iloc[0]["count"] == 2
    assert result.iloc[1]["type"] == "Event"
    assert result.iloc[1]["count"] == 1


def test_transactions_summary_aggregates_per_store():
    transactions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2013-01-01", "2013-01-02", "2013-01-01"]
            ),
            "store_nbr": [1, 1, 2],
            "transactions": [100, 200, 50],
        }
    )
    result = transactions_summary(transactions)

    store_1 = result.loc[result["store_nbr"] == 1].iloc[0]
    assert store_1["total_transactions"] == 300
    assert store_1["mean_daily_transactions"] == pytest.approx(150.0)


def test_oil_price_trend_sorts_by_date():
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-03", "2013-01-01", "2013-01-02"]),
            "dcoilwtico": [90.0, 93.0, 92.0],
        }
    )
    result = oil_price_trend(oil)

    assert result["date"].is_monotonic_increasing
    assert result["dcoilwtico"].iloc[0] == 93.0


def test_merge_oil_with_sales_forward_fills(train_frame):
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01"]),
            "dcoilwtico": [93.0],
        }
    )
    merged = merge_oil_with_sales(train_frame, oil)

    assert set(merged.columns) == {"date", "sales", "dcoilwtico"}
    assert merged["dcoilwtico"].notna().all()
    assert merged.loc[merged["date"] == "2013-01-02", "dcoilwtico"].iloc[0] == 93.0


def test_missing_value_report_lists_only_nonzero():
    datasets = {
        "a": pd.DataFrame({"x": [1, None, 3]}),
        "b": pd.DataFrame({"y": [1, 2, 3]}),
    }
    report = missing_value_report(datasets)

    assert len(report) == 1
    assert report.iloc[0]["dataset"] == "a"
    assert report.iloc[0]["column"] == "x"
    assert report.iloc[0]["missing_count"] == 1
    assert report.iloc[0]["missing_share"] == pytest.approx(1 / 3)


def test_missing_value_report_empty_when_complete():
    report = missing_value_report({"a": pd.DataFrame({"x": [1, 2, 3]})})
    assert report.empty
