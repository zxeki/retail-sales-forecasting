"""Descriptions of the CSV files provided by the Kaggle competition."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Expected contents of one competition CSV file."""

    name: str
    filename: str
    required_columns: tuple[str, ...]
    date_columns: tuple[str, ...] = ()


DATASET_SPECS: dict[str, DatasetSpec] = {
    "train": DatasetSpec(
        name="train",
        filename="train.csv",
        required_columns=(
            "id",
            "date",
            "store_nbr",
            "family",
            "sales",
            "onpromotion",
        ),
        date_columns=("date",),
    ),
    "test": DatasetSpec(
        name="test",
        filename="test.csv",
        required_columns=(
            "id",
            "date",
            "store_nbr",
            "family",
            "onpromotion",
        ),
        date_columns=("date",),
    ),
    "stores": DatasetSpec(
        name="stores",
        filename="stores.csv",
        required_columns=("store_nbr", "city", "state", "type", "cluster"),
    ),
    "oil": DatasetSpec(
        name="oil",
        filename="oil.csv",
        required_columns=("date", "dcoilwtico"),
        date_columns=("date",),
    ),
    "holidays_events": DatasetSpec(
        name="holidays_events",
        filename="holidays_events.csv",
        required_columns=(
            "date",
            "type",
            "locale",
            "locale_name",
            "description",
            "transferred",
        ),
        date_columns=("date",),
    ),
    "transactions": DatasetSpec(
        name="transactions",
        filename="transactions.csv",
        required_columns=("date", "store_nbr", "transactions"),
        date_columns=("date",),
    ),
}

DATASET_NAMES = tuple(DATASET_SPECS)
