"""Tests for Kaggle submission generation and format validation."""

import numpy as np
import pandas as pd
import pytest

from retail_sales_forecasting.submission import (
    SUBMISSION_COLUMNS,
    build_submission_frame,
    generate_submission,
    save_submission,
    validate_submission,
)


@pytest.fixture
def panel() -> tuple:
    """Train (60 days) + test (6 days) panel for two stores and two families."""
    train_dates = pd.date_range("2013-01-01", periods=60, freq="D")
    test_dates = pd.date_range("2013-03-02", periods=6, freq="D")
    rng = np.random.default_rng(3)

    train_rows = []
    for store in (1, 2):
        for family in ("A", "B"):
            for index, date in enumerate(train_dates):
                train_rows.append(
                    {
                        "id": len(train_rows),
                        "date": date,
                        "store_nbr": store,
                        "family": family,
                        "sales": 30.0 + 5.0 * (index % 7) + rng.normal(0, 0.5),
                        "onpromotion": index % 3,
                    }
                )
    train = pd.DataFrame(train_rows)

    test_rows = []
    for offset, date in enumerate(test_dates):
        for store in (1, 2):
            for family in ("A", "B"):
                test_rows.append(
                    {
                        "id": 10_000 + len(test_rows),
                        "date": date,
                        "store_nbr": store,
                        "family": family,
                        "onpromotion": offset % 3,
                    }
                )
    test = pd.DataFrame(test_rows)

    holidays = pd.DataFrame(
        {"date": pd.to_datetime(["2013-01-06"]), "locale": ["National"]}
    )
    return train, test, holidays


def test_build_submission_frame_preserves_order():
    test = pd.DataFrame({"id": [3, 1, 2]})
    frame = build_submission_frame(test, np.array([0.3, 0.1, 0.2]))

    assert list(frame.columns) == list(SUBMISSION_COLUMNS)
    assert list(frame["id"]) == [3, 1, 2]


def test_build_submission_frame_length_mismatch():
    test = pd.DataFrame({"id": [1, 2, 3]})
    with pytest.raises(ValueError):
        build_submission_frame(test, np.array([1.0, 2.0]))


def test_validate_submission_accepts_valid_frame():
    test = pd.DataFrame({"id": [1, 2, 3]})
    frame = build_submission_frame(test, np.array([1.0, 2.0, 3.0]))

    report = validate_submission(frame, test)

    assert report["rows"] == 3
    assert report["duplicate_ids"] == 0
    assert report["missing_sales"] == 0
    assert report["id_order_matches_test"] is True


def test_validate_submission_rejects_wrong_columns():
    test = pd.DataFrame({"id": [1, 2]})
    bad = pd.DataFrame({"id": [1, 2], "value": [1.0, 2.0]})
    with pytest.raises(ValueError):
        validate_submission(bad, test)


def test_validate_submission_rejects_reordered_ids():
    test = pd.DataFrame({"id": [1, 2, 3]})
    bad = build_submission_frame(pd.DataFrame({"id": [3, 2, 1]}), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        validate_submission(bad, test)


def test_validate_submission_rejects_missing_sales():
    test = pd.DataFrame({"id": [1, 2]})
    bad = pd.DataFrame({"id": [1, 2], "sales": [1.0, np.nan]})
    with pytest.raises(ValueError):
        validate_submission(bad, test)


def test_generate_submission_matches_test_ids(panel):
    pytest.importorskip("lightgbm")
    train, test, holidays = panel
    result = generate_submission(train, test, holidays, n_estimators=20)

    assert list(result.frame.columns) == list(SUBMISSION_COLUMNS)
    assert result.frame["id"].tolist() == test["id"].tolist()
    assert len(result.frame) == len(test)


def test_generate_submission_recursive_fills_later_days(panel):
    pytest.importorskip("lightgbm")
    train, test, holidays = panel
    result = generate_submission(train, test, holidays, n_estimators=20)

    merged = result.frame.merge(test, on="id")
    # The last test day (beyond the 7-day lag horizon) must still be
    # forecast with non-trivial values, not collapsed to zero.
    last_day = merged["date"].max()
    last_total = merged[merged["date"] == last_day]["sales"].sum()
    assert last_total > 0

    daily = merged.groupby("date")["sales"].sum()
    assert (daily > 0).all()
    # No later day should be an order of magnitude below the first day.
    assert daily.min() > daily.max() / 20


def test_generate_submission_non_negative(panel):
    pytest.importorskip("lightgbm")
    train, test, holidays = panel
    result = generate_submission(train, test, holidays, n_estimators=20)

    assert (result.frame["sales"] >= 0).all()


def test_save_submission_writes_csv(panel, tmp_path):
    pytest.importorskip("lightgbm")
    train, test, holidays = panel
    result = generate_submission(train, test, holidays, n_estimators=20)

    target = tmp_path / "submission.csv"
    written = save_submission(result, target)

    assert written == target
    reloaded = pd.read_csv(target)
    assert list(reloaded.columns) == list(SUBMISSION_COLUMNS)
    assert len(reloaded) == len(test)
