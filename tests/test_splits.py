"""Tests for chronological splitting."""

import pandas as pd
import pytest

from retail_sales_forecasting.splits import chronological_split


@pytest.fixture
def daily_frame() -> pd.DataFrame:
    dates = pd.date_range("2013-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "sales": range(10),
        }
    )


def test_split_by_days(daily_frame):
    split = chronological_split(daily_frame, validation_days=3)

    assert split.cutoff == pd.Timestamp("2013-01-07")
    assert split.train_dates == (pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-07"))
    assert split.validation_dates == (pd.Timestamp("2013-01-08"), pd.Timestamp("2013-01-10"))
    assert len(split.train) == 7
    assert len(split.validation) == 3


def test_split_is_chronological(daily_frame):
    split = chronological_split(daily_frame, validation_days=4)

    assert split.train["date"].max() < split.validation["date"].min()


def test_split_by_fraction(daily_frame):
    split = chronological_split(daily_frame, validation_fraction=0.2)

    assert len(split.validation) == 2
    assert len(split.train) == 8


def test_requires_exactly_one_parameter(daily_frame):
    with pytest.raises(ValueError):
        chronological_split(daily_frame)
    with pytest.raises(ValueError):
        chronological_split(daily_frame, validation_days=2, validation_fraction=0.2)


def test_invalid_days(daily_frame):
    with pytest.raises(ValueError):
        chronological_split(daily_frame, validation_days=0)
    with pytest.raises(ValueError):
        chronological_split(daily_frame, validation_days=10)


def test_invalid_fraction(daily_frame):
    with pytest.raises(ValueError):
        chronological_split(daily_frame, validation_fraction=1.5)
    with pytest.raises(ValueError):
        chronological_split(daily_frame, validation_fraction=0.0)


def test_missing_date_column():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"x": [1, 2]}), validation_days=1)


def test_too_few_dates():
    frame = pd.DataFrame({"date": pd.to_datetime(["2013-01-01"] * 3)})
    with pytest.raises(ValueError):
        chronological_split(frame, validation_days=1)


def test_split_does_not_mutate_input(daily_frame):
    original = daily_frame.copy()
    chronological_split(daily_frame, validation_days=3)
    pd.testing.assert_frame_equal(daily_frame, original)
