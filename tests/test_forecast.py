"""Tests for the end-to-end forecasting pipeline."""

import numpy as np
import pandas as pd
import pytest

from retail_sales_forecasting.forecast import run_forecast


@pytest.fixture
def panel() -> pd.DataFrame:
    dates = pd.date_range("2013-01-01", periods=80, freq="D")
    rows = []
    rng = np.random.default_rng(1)
    for store in (1, 2):
        for family in ("A", "B"):
            level = 50.0 if family == "A" else 20.0
            for index, date in enumerate(dates):
                rows.append(
                    {
                        "date": date,
                        "store_nbr": store,
                        "family": family,
                        "sales": max(level + 5.0 * (index % 7) + rng.normal(0, 1), 0.0),
                        "onpromotion": index % 4,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def holidays() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-06", "2013-02-11"]),
            "locale": ["National", "National"],
            "type": ["Holiday", "Holiday"],
        }
    )


def test_run_forecast_returns_predictions_and_metrics(panel, holidays):
    pytest.importorskip("lightgbm")
    run = run_forecast(panel, holidays, validation_days=16)

    assert set(run.predictions.columns) >= {
        "date",
        "store_nbr",
        "family",
        "sales",
        "prediction",
        "baseline",
        "abs_error",
    }
    assert run.cutoff == pd.Timestamp("2013-01-01") + pd.Timedelta(63, unit="D")
    assert run.date_min > run.cutoff


def test_run_forecast_metrics_table(panel, holidays):
    pytest.importorskip("lightgbm")
    run = run_forecast(panel, holidays, validation_days=16)

    assert set(run.metrics["model"]) == {
        "LightGBM",
        "Seasonal naive (last week)",
    }
    assert set(["mae", "rmse", "rmsle"]).issubset(run.metrics.columns)
    assert run.metrics["rmsle"].is_monotonic_increasing


def test_run_forecast_predictions_are_non_negative(panel, holidays):
    pytest.importorskip("lightgbm")
    run = run_forecast(panel, holidays, validation_days=16)

    assert (run.predictions["prediction"] >= 0).all()


def test_run_forecast_abs_error_matches(panel, holidays):
    pytest.importorskip("lightgbm")
    run = run_forecast(panel, holidays, validation_days=16)

    expected = (run.predictions["sales"] - run.predictions["prediction"]).abs()
    assert np.allclose(run.predictions["abs_error"], expected)
