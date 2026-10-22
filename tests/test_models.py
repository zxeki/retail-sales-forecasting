"""Tests for candidate forecasting models and comparison helpers."""

import numpy as np
import pandas as pd
import pytest

from retail_sales_forecasting.models import (
    DEFAULT_FEATURES,
    LINEAR_FEATURES,
    compare_models,
    error_breakdown,
    predict,
    train_lightgbm,
    train_ridge,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    """A small two-store, two-family daily panel with a weekly pattern."""
    dates = pd.date_range("2013-01-01", periods=60, freq="D")
    rows = []
    rng = np.random.default_rng(0)
    stores = ([1] * 60) + ([2] * 60)
    families = (["A"] * 60) + (["B"] * 60)
    for store, family in zip(stores, families):
        for index, date in enumerate(dates):
            base = 10.0 + 5.0 * (index % 7) + (20.0 if store == 2 else 0.0)
            noise = float(rng.normal(0, 0.2))
            rows.append(
                {
                    "date": date,
                    "store_nbr": store,
                    "family": family,
                    "sales": max(base + noise, 0.0),
                    "onpromotion": index % 3,
                }
            )
    return pd.DataFrame(rows)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in DEFAULT_FEATURES if column in frame.columns]


@pytest.fixture
def with_calendar(panel):
    from retail_sales_forecasting.features import build_feature_frame

    return build_feature_frame(panel)


def test_train_lightgbm_returns_model_and_columns(with_calendar):
    pytest.importorskip("lightgbm")
    train = with_calendar[with_calendar["date"] < "2013-02-15"]

    model, columns = train_lightgbm(train, n_estimators=10, num_leaves=8)

    assert columns == _feature_columns(train)
    assert len(columns) > 0
    assert hasattr(model, "predict")


def test_train_ridge_uses_linear_features(with_calendar):
    pytest.importorskip("sklearn")
    train = with_calendar[with_calendar["date"] < "2013-02-15"]

    model, columns = train_ridge(train)

    assert columns == [column for column in LINEAR_FEATURES if column in train.columns]
    assert hasattr(model, "predict")


def test_predict_returns_non_negative_array(with_calendar):
    pytest.importorskip("sklearn")
    train = with_calendar[with_calendar["date"] < "2013-02-15"]
    validation = with_calendar[with_calendar["date"] >= "2013-02-15"]

    model, columns = train_ridge(train)
    predictions = predict(model, validation, columns)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(validation)
    assert np.all(predictions >= 0)


def test_predict_learns_weekly_signal(with_calendar):
    pytest.importorskip("sklearn")
    train = with_calendar[with_calendar["date"] < "2013-02-15"]
    validation = with_calendar[with_calendar["date"] >= "2013-02-15"].reset_index(drop=True)

    model, columns = train_ridge(train)
    predictions = predict(model, validation, columns)

    from retail_sales_forecasting.evaluation import mean_absolute_error

    naive = validation["lag_7"].fillna(validation["sales"].mean()).to_numpy()
    assert mean_absolute_error(validation["sales"], predictions) < mean_absolute_error(
        validation["sales"], naive
    )


def test_compare_models_ranks_by_rmsle(with_calendar):
    pytest.importorskip("sklearn")
    train = with_calendar[with_calendar["date"] < "2013-02-15"]
    validation = with_calendar[with_calendar["date"] >= "2013-02-15"]

    result = compare_models(train, validation, {"ridge": train_ridge})

    assert list(result["model"]) == ["ridge"]
    assert set(["mae", "rmse", "rmsle"]).issubset(result.columns)
    assert result["rmsle"].is_monotonic_increasing


def test_compare_models_with_feature_tuple(with_calendar):
    pytest.importorskip("sklearn")
    train = with_calendar[with_calendar["date"] < "2013-02-15"]
    validation = with_calendar[with_calendar["date"] >= "2013-02-15"]

    result = compare_models(
        train, validation, {"ridge": (train_ridge, LINEAR_FEATURES)}
    )

    assert len(result) == 1


def test_error_breakdown_groups_by_family(with_calendar):
    pytest.importorskip("sklearn")
    train = with_calendar[with_calendar["date"] < "2013-02-15"]
    validation = with_calendar[with_calendar["date"] >= "2013-02-15"]

    model, columns = train_ridge(train)
    predictions = predict(model, validation, columns)
    breakdown = error_breakdown(validation, predictions, by="family")

    assert "family" in breakdown.columns
    assert "mean_abs_error" in breakdown.columns
    assert breakdown["mean_abs_error"].is_monotonic_decreasing


def test_error_breakdown_by_store(with_calendar):
    pytest.importorskip("sklearn")
    train = with_calendar[with_calendar["date"] < "2013-02-15"]
    validation = with_calendar[with_calendar["date"] >= "2013-02-15"]

    model, columns = train_ridge(train)
    predictions = predict(model, validation, columns)
    breakdown = error_breakdown(validation, predictions, by="store_nbr")

    assert set(breakdown["store_nbr"]) == {1, 2}
