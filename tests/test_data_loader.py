"""Tests for the competition data loader."""

import pandas as pd
import pytest

from retail_sales_forecasting.config import KAGGLE_DATA_URL
from retail_sales_forecasting.data_loader import (
    DatasetNotFoundError,
    available_datasets,
    dataset_inventory,
    dataset_path,
    is_available,
    load_available_datasets,
    load_dataset,
    load_datasets,
    missing_datasets,
)


def test_dataset_path_uses_expected_filename(raw_data_dir):
    assert dataset_path("train", raw_data_dir).name == "train.csv"


def test_available_datasets_lists_all_files(raw_data_dir):
    assert available_datasets(raw_data_dir) == [
        "train",
        "test",
        "stores",
        "oil",
        "holidays_events",
        "transactions",
    ]


def test_missing_datasets_is_empty_when_all_files_exist(raw_data_dir):
    assert missing_datasets(raw_data_dir) == []


def test_is_available(raw_data_dir, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert is_available("train", raw_data_dir) is True
    assert is_available("train", empty_dir) is False
    with pytest.raises(ValueError, match="Unknown dataset"):
        is_available("sample_submission", raw_data_dir)


def test_dataset_inventory(raw_data_dir):
    inventory = dataset_inventory(raw_data_dir)
    assert inventory["train"] is True
    assert len(inventory) == 6


def test_load_dataset_reads_csv_and_parses_dates(raw_data_dir):
    train = load_dataset("train", data_dir=raw_data_dir)

    assert train.shape == (3, 6)
    assert pd.api.types.is_datetime64_any_dtype(train["date"])
    assert train["date"].min() == pd.Timestamp("2013-01-01")
    assert train["date"].max() == pd.Timestamp("2013-01-03")
    assert list(train.columns) == [
        "id",
        "date",
        "store_nbr",
        "family",
        "sales",
        "onpromotion",
    ]


def test_load_dataset_parses_store_independent_dates(raw_data_dir):
    oil = load_dataset("oil", data_dir=raw_data_dir)
    transactions = load_dataset("transactions", data_dir=raw_data_dir)

    assert pd.api.types.is_datetime64_any_dtype(oil["date"])
    assert pd.api.types.is_datetime64_any_dtype(transactions["date"])


def test_load_datasets_loads_selected_names(raw_data_dir):
    loaded = load_datasets(["train", "stores"], data_dir=raw_data_dir)

    assert set(loaded) == {"train", "stores"}
    assert loaded["stores"].shape == (2, 5)


def test_load_datasets_skip_missing(raw_data_dir, tmp_path):
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    dataset_path("train", raw_data_dir).replace(partial_dir / "train.csv")

    loaded = load_datasets(
        ["train", "test"], data_dir=partial_dir, skip_missing=True
    )

    assert set(loaded) == {"train"}


def test_load_datasets_raises_for_missing_file(raw_data_dir, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(DatasetNotFoundError):
        load_datasets(["train", "test"], data_dir=empty_dir)


def test_load_available_datasets_returns_only_present_files(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert load_available_datasets(data_dir=empty_dir) == {}


def test_load_missing_dataset_has_actionable_message(tmp_path):
    empty_dir = tmp_path / "raw"
    empty_dir.mkdir()

    with pytest.raises(DatasetNotFoundError) as error:
        load_dataset("train", data_dir=empty_dir)

    message = str(error.value)
    assert "train.csv" in message
    assert "data/raw" in message
    assert KAGGLE_DATA_URL in message
    assert "train.csv, test.csv, stores.csv" in message


def test_unknown_dataset_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("unknown", data_dir=tmp_path)
