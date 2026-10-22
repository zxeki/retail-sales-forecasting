"""Basic validation for competition datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .datasets import DATASET_SPECS, DatasetSpec


@dataclass
class ValidationResult:
    """Summary of the checks performed on one dataset."""

    dataset: str
    filename: str
    n_rows: int
    n_columns: int
    required_columns: tuple[str, ...]
    missing_columns: tuple[str, ...]
    invalid_dates: dict[str, int]
    missing_values: dict[str, int]

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return ``(rows, columns)``."""
        return (self.n_rows, self.n_columns)

    @property
    def is_valid(self) -> bool:
        """A dataset is valid when required columns exist and dates parse."""
        return not self.missing_columns and not self.invalid_dates

    @property
    def has_missing_values(self) -> bool:
        """Return ``True`` when any cell is missing."""
        return bool(self.missing_values)

    def as_dict(self) -> dict[str, object]:
        """Return the result as a plain dictionary."""
        return {
            "dataset": self.dataset,
            "filename": self.filename,
            "dimensions": self.dimensions,
            "required_columns": self.required_columns,
            "missing_columns": self.missing_columns,
            "invalid_dates": self.invalid_dates,
            "missing_values": self.missing_values,
            "has_missing_values": self.has_missing_values,
            "is_valid": self.is_valid,
        }


def _get_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        options = ", ".join(DATASET_SPECS)
        raise ValueError(
            f"Unknown dataset '{name}'. Choose one of: {options}."
        ) from exc


def _invalid_date_counts(
    dataframe: pd.DataFrame, date_columns: tuple[str, ...]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in date_columns:
        if column not in dataframe.columns:
            continue
        parsed = pd.to_datetime(
            dataframe[column], errors="coerce", format="mixed"
        )
        invalid_count = int(parsed.isna().sum())
        if invalid_count:
            counts[column] = invalid_count
    return counts


def _missing_value_counts(dataframe: pd.DataFrame) -> dict[str, int]:
    counts = dataframe.isna().sum()
    return {
        str(column): int(count)
        for column, count in counts.items()
        if count > 0
    }


def validate_dataset(dataframe: pd.DataFrame, name: str) -> ValidationResult:
    """Validate a competition dataset without modifying it.

    Checks include required columns, dataset dimensions, date parsing, and
    missing values. Missing values are reported but do not make a dataset
    invalid; missing required columns or unparseable dates do.
    """
    spec = _get_spec(name)

    missing_columns = tuple(
        column
        for column in spec.required_columns
        if column not in dataframe.columns
    )

    return ValidationResult(
        dataset=name,
        filename=spec.filename,
        n_rows=int(dataframe.shape[0]),
        n_columns=int(dataframe.shape[1]),
        required_columns=spec.required_columns,
        missing_columns=missing_columns,
        invalid_dates=_invalid_date_counts(dataframe, spec.date_columns),
        missing_values=_missing_value_counts(dataframe),
    )


def format_validation_report(result: ValidationResult) -> str:
    """Format a validation result as a readable multi-line report."""
    rows, columns = result.dimensions
    present_columns = len(result.required_columns) - len(result.missing_columns)
    lines = [
        f"Dataset: {result.dataset} ({result.filename})",
        f"Dimensions: {rows:,} rows x {columns:,} columns",
        (
            "Required columns: "
            f"{present_columns}/{len(result.required_columns)} present"
        ),
    ]

    if result.missing_columns:
        lines.append(
            "Missing required columns: " + ", ".join(result.missing_columns)
        )
    else:
        lines.append("Missing required columns: none")

    if result.invalid_dates:
        details = ", ".join(
            f"{column} ({count:,})"
            for column, count in result.invalid_dates.items()
        )
        lines.append(f"Unparseable/missing dates: {details}")
    else:
        lines.append("Unparseable/missing dates: none")

    if result.missing_values:
        details = ", ".join(
            f"{column} ({count:,})"
            for column, count in result.missing_values.items()
        )
        lines.append(f"Missing values: {details}")
    else:
        lines.append("Missing values: none")

    lines.append("Status: OK" if result.is_valid else "Status: ISSUES FOUND")
    return "\n".join(lines)
