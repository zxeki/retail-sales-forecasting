"""Tests for leakage-safe feature engineering."""

import numpy as np
import pandas as pd
import pytest

from retail_sales_forecasting.features import (
    add_calendar_features,
    add_lag_features,
    add_promotion_features,
    add_rolling_features,
    build_feature_frame,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    dates = pd.date_range("2013-01-01", periods=30, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "A",
            "sales": np.arange(30, dtype=float),
            "onpromotion": [0] * 30,
        }
    )


def test_calendar_features_added(frame):
    result = add_calendar_features(frame)

    for column in [
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "is_holiday",
    ]:
        assert column in result.columns

    # 2013-01-05 is a Saturday
    row = result.loc[result["date"] == "2013-01-05"].iloc[0]
    assert row["day_of_week"] == 5
    assert row["is_weekend"] == 1
    assert row["is_holiday"] == 0


def test_calendar_features_with_holidays(frame):
    holidays = pd.DataFrame(
        {"date": pd.to_datetime(["2013-01-03"]), "locale": ["National"]}
    )
    result = add_calendar_features(frame, country_holidays=holidays)

    assert result.loc[result["date"] == "2013-01-03", "is_holiday"].iloc[0] == 1
    assert result["is_holiday"].sum() == 1


def test_promotion_feature(frame):
    frame = frame.copy()
    frame.loc[0, "onpromotion"] = 5
    result = add_promotion_features(frame)

    assert result["is_promoted"].iloc[0] == 1
    assert result["is_promoted"].sum() == 1


def test_lag_features_are_shifted(frame):
    result = add_lag_features(frame, lags=(7,))

    first = result.sort_values("date").iloc[0]
    assert pd.isna(first["lag_7"])
    eighth = result.sort_values("date").iloc[7]
    assert eighth["lag_7"] == 0.0  # value 7 days earlier


def test_lag_does_not_leak_current_row(frame):
    result = add_lag_features(frame, lags=(1,)).sort_values("date")

    assert result["lag_1"].iloc[0] is np.nan or pd.isna(result["lag_1"].iloc[0])
    for index in range(1, len(result)):
        assert result["lag_1"].iloc[index] == result["sales"].iloc[index - 1]


def test_rolling_features_exclude_current_day(frame):
    result = add_rolling_features(frame, windows=(7,)).sort_values("date")

    # first row has no history via shift(1); rolling min_periods=1 -> NaN
    assert pd.isna(result["rolling_mean_7"].iloc[0])
    # row 7: mean of sales[0:7] shifted -> mean of rows 0..6 = 3.0
    assert result["rolling_mean_7"].iloc[7] == pytest.approx(3.0)


def test_rolling_never_includes_future(frame):
    result = add_rolling_features(frame, windows=(3,)).sort_values("date")
    for index in range(len(result)):
        window = result["rolling_mean_3"].iloc[index]
        if not pd.isna(window):
            assert window <= result["sales"].iloc[index]


def test_lag_is_per_group():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2013-01-01", periods=4, freq="D").tolist() * 2,
            "store_nbr": [1, 1, 1, 1, 2, 2, 2, 2],
            "family": ["A"] * 4 + ["A"] * 4,
            "sales": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
        }
    )
    result = add_lag_features(frame, lags=(1,)).sort_values(["store_nbr", "date"])

    store_2 = result[result["store_nbr"] == 2].reset_index(drop=True)
    assert pd.isna(store_2["lag_1"].iloc[0])
    assert store_2["lag_1"].iloc[1] == 100.0


def test_build_feature_frame_combines_all(frame):
    result = build_feature_frame(frame)

    for column in ["lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28"]:
        assert column in result.columns
    assert "is_promoted" in result.columns
    assert "day_of_week" in result.columns


def test_features_do_not_modify_input(frame):
    original = frame.copy()
    build_feature_frame(frame)
    pd.testing.assert_frame_equal(frame, original)
