"""Tests for basic dataset validation."""

import pandas as pd
import pytest

from retail_sales_forecasting.validation import (
    format_validation_report,
    validate_dataset,
)

from conftest import make_train_frame


def test_valid_train_dataset():
    result = validate_dataset(make_train_frame(), "train")

    assert result.dimensions == (3, 6)
    assert result.missing_columns == ()
    assert result.invalid_dates == {}
    assert result.missing_values == {}
    assert result.has_missing_values is False
    assert result.is_valid is True


def test_detects_missing_required_columns():
    train = make_train_frame().drop(columns=["sales"])

    result = validate_dataset(train, "train")

    assert result.missing_columns == ("sales",)
    assert result.is_valid is False


def test_detects_invalid_dates():
    train = make_train_frame()
    train.loc[1, "date"] = "not-a-date"

    result = validate_dataset(train, "train")

    assert result.invalid_dates == {"date": 1}
    assert result.is_valid is False


def test_reports_missing_values_without_failing_validation():
    train = make_train_frame()
    train.loc[0, "sales"] = None

    result = validate_dataset(train, "train")

    assert result.missing_values == {"sales": 1}
    assert result.has_missing_values is True
    assert result.is_valid is True


def test_detects_invalid_dates_in_test_dataset():
    test = pd.DataFrame(
        {
            "id": [1],
            "date": ["invalid"],
            "store_nbr": [1],
            "family": ["GROCERY"],
            "onpromotion": [0],
        }
    )

    result = validate_dataset(test, "test")

    assert result.invalid_dates == {"date": 1}
    assert result.is_valid is False


def test_stores_dataset_has_no_date_columns():
    stores = pd.DataFrame(
        {
            "store_nbr": [1],
            "city": ["Quito"],
            "state": ["Pichincha"],
            "type": ["D"],
            "cluster": [1],
        }
    )

    result = validate_dataset(stores, "stores")

    assert result.invalid_dates == {}
    assert result.is_valid is True


def test_format_validation_report_contains_summary():
    result = validate_dataset(make_train_frame(), "train")
    report = format_validation_report(result)

    assert "Dataset: train (train.csv)" in report
    assert "Dimensions: 3 rows x 6 columns" in report
    assert "Required columns: 6/6 present" in report
    assert "Missing values: none" in report
    assert "Status: OK" in report


def test_format_validation_report_lists_issues():
    train = make_train_frame().drop(columns=["family"])
    train.loc[0, "date"] = "not-a-date"

    result = validate_dataset(train, "train")
    report = format_validation_report(result)

    assert "Missing required columns: family" in report
    assert "Unparseable/missing dates: date (1)" in report
    assert "Status: ISSUES FOUND" in report


def test_as_dict_returns_plain_values():
    result = validate_dataset(make_train_frame(), "train")
    result_dict = result.as_dict()

    assert result_dict["dimensions"] == (3, 6)
    assert result_dict["missing_columns"] == ()
    assert result_dict["is_valid"] is True


def test_unknown_dataset_raises_value_error():
    with pytest.raises(ValueError, match="Unknown dataset"):
        validate_dataset(make_train_frame(), "unknown")
