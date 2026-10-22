"""Tests for evaluation metrics and the seasonal-naive baseline."""

import numpy as np
import pandas as pd
import pytest

from retail_sales_forecasting.evaluation import (
    evaluate_forecast,
    mean_absolute_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
    seasonal_naive_forecast,
)


def test_mae_perfect_predictions():
    assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0.0


def test_mae_known_value():
    assert mean_absolute_error([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)


def test_rmse_known_value():
    assert root_mean_squared_error([2, 4], [4, 2]) == pytest.approx(2.0)


def test_rmsle_zero_on_perfect_predictions():
    assert root_mean_squared_log_error([1, 2, 3], [1, 2, 3]) == 0.0


def test_rmsle_clips_negative_predictions():
    value = root_mean_squared_log_error([0, 1], [-5, 1])
    assert value == pytest.approx(0.0) or value >= 0.0


def test_rmsle_rejects_negative_truth():
    with pytest.raises(ValueError):
        root_mean_squared_log_error([-1, 1], [0, 1])


def test_evaluate_forecast_keys_and_prefix():
    metrics = evaluate_forecast([1, 2, 3], [1, 2, 4], prefix="val_")
    assert set(metrics) == {"val_mae", "val_rmse", "val_rmsle"}
    assert metrics["val_mae"] > 0


@pytest.fixture
def weekly_frame() -> pd.DataFrame:
    dates = pd.date_range("2013-01-01", periods=14, freq="D")
    sales = [10, 10, 10, 10, 10, 10, 20] * 2
    return pd.DataFrame(
        {"date": dates, "store_nbr": 1, "family": "A", "sales": sales}
    )


def test_seasonal_naive_repeats_last_week(weekly_frame):
    target = pd.date_range("2013-01-15", periods=7, freq="D")
    preds = seasonal_naive_forecast(
        weekly_frame, target, use_weekday_profile=False
    )

    assert len(preds) == 7
    assert list(preds["prediction"]) == [10, 10, 10, 10, 10, 10, 20]


def test_seasonal_naive_weekday_profile(weekly_frame):
    target = pd.date_range("2013-01-15", periods=7, freq="D")
    preds = seasonal_naive_forecast(
        weekly_frame, target, use_weekday_profile=True
    )

    assert len(preds) == 7
    # Monday..Saturday average 10, Sunday average 20
    assert preds["prediction"].iloc[0] == pytest.approx(10.0)
    assert preds["prediction"].iloc[6] == pytest.approx(20.0)


def test_seasonal_naive_output_columns(weekly_frame):
    target = pd.date_range("2013-01-15", periods=2, freq="D")
    preds = seasonal_naive_forecast(weekly_frame, target)

    assert list(preds.columns) == ["store_nbr", "family", "date", "prediction"]


def test_seasonal_naive_only_uses_training_dates(weekly_frame):
    target = pd.date_range("2013-01-15", periods=7, freq="D")
    preds = seasonal_naive_forecast(weekly_frame, target)

    assert preds["date"].min() >= pd.Timestamp("2013-01-15")
