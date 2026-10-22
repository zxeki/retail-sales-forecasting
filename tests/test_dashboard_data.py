"""Tests for the dashboard data helpers and forecast artifacts."""

import pandas as pd
import pytest

from retail_sales_forecasting.dashboard_data import (
    daily_totals,
    family_summary,
    filter_options,
    filter_predictions,
    metric_comparison,
    slice_metrics,
    store_summary,
)
from retail_sales_forecasting.forecast import ForecastRun


@pytest.fixture
def predictions() -> pd.DataFrame:
    dates = pd.date_range("2017-08-01", periods=3, freq="D")
    rows = []
    for store in (1, 2):
        for family in ("A", "B"):
            for offset, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "store_nbr": store,
                        "family": family,
                        "sales": 100.0 + offset + store,
                        "prediction": 90.0 + offset + store,
                        "baseline": 80.0 + offset + store,
                        "abs_error": 10.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def run(predictions) -> ForecastRun:
    metrics = pd.DataFrame(
        [
            {"model": "LightGBM", "mae": 10.0, "rmse": 20.0, "rmsle": 0.4},
            {"model": "Seasonal naive (last week)", "mae": 15.0, "rmse": 30.0, "rmsle": 0.6},
        ]
    )
    return ForecastRun(
        predictions=predictions,
        metrics=metrics,
        cutoff=pd.Timestamp("2017-07-31"),
    )


def test_filter_options_lists_choices(predictions):
    options = filter_options(predictions)

    assert options["families"] == ["A", "B"]
    assert options["stores"] == [1, 2]
    assert options["date_min"] == pd.Timestamp("2017-08-01")
    assert options["date_max"] == pd.Timestamp("2017-08-03")


def test_filter_predictions_by_family(predictions):
    filtered = filter_predictions(predictions, families=["A"])

    assert set(filtered["family"]) == {"A"}
    assert len(filtered) == 6


def test_filter_predictions_by_store(predictions):
    filtered = filter_predictions(predictions, stores=[2])

    assert set(filtered["store_nbr"]) == {2}


def test_filter_predictions_by_date_range(predictions):
    filtered = filter_predictions(
        predictions, date_range=("2017-08-02", "2017-08-03")
    )

    assert filtered["date"].min() == pd.Timestamp("2017-08-02")
    assert filtered["date"].max() == pd.Timestamp("2017-08-03")


def test_filter_predictions_combines_filters(predictions):
    filtered = filter_predictions(
        predictions, families=["A"], stores=[1], date_range=("2017-08-01", "2017-08-02")
    )

    assert len(filtered) == 2


def test_filter_predictions_does_not_mutate(predictions):
    original = predictions.copy()
    filter_predictions(predictions, families=["A"])
    pd.testing.assert_frame_equal(predictions, original)


def test_daily_totals_sums_by_date(predictions):
    daily = daily_totals(predictions)

    assert list(daily.columns) == ["date", "actual", "forecast", "baseline"]
    assert len(daily) == 3
    first = daily.iloc[0]
    assert first["actual"] == predictions[predictions["date"] == first["date"]]["sales"].sum()


def test_family_summary_sorted_by_error(predictions):
    summary = family_summary(predictions)

    assert "mean_abs_error" in summary.columns
    assert summary["mean_abs_error"].is_monotonic_decreasing
    assert set(summary["family"]) == {"A", "B"}


def test_store_summary_groups_by_store(predictions):
    summary = store_summary(predictions)

    assert set(summary["store_nbr"]) == {1, 2}
    assert len(summary) == 2


def test_slice_metrics_returns_expected_keys(predictions):
    metrics = slice_metrics(predictions)

    assert set(metrics) == {"rmsle", "mae", "rmse", "rows"}
    assert metrics["rows"] == len(predictions)
    assert metrics["mae"] == pytest.approx(10.0)


def test_metric_comparison_returns_copy(run):
    comparison = metric_comparison(run)
    comparison.loc[0, "mae"] = 0.0

    assert run.metrics.loc[0, "mae"] == 10.0
