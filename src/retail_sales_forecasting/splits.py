"""Chronological train/validation splits.

Time-series data must be split by date, never randomly, so that validation
data always comes after training data (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    """A train/validation split by date."""

    train: pd.DataFrame
    validation: pd.DataFrame
    cutoff: pd.Timestamp

    @property
    def train_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.train["date"].min(), self.train["date"].max())

    @property
    def validation_dates(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return (self.validation["date"].min(), self.validation["date"].max())


def chronological_split(
    data: pd.DataFrame,
    *,
    date_column: str = "date",
    validation_days: int | None = None,
    validation_fraction: float | None = None,
) -> ChronologicalSplit:
    """Split ``data`` into train and validation by date.

    Exactly one of ``validation_days`` or ``validation_fraction`` must be set.
    ``validation_days`` takes the last N distinct dates as validation;
    ``validation_fraction`` uses the last fraction of distinct dates.
    """
    if (validation_days is None) == (validation_fraction is None):
        raise ValueError(
            "Set exactly one of validation_days or validation_fraction."
        )

    if date_column not in data.columns:
        raise ValueError(f"Missing date column: {date_column}")

    frame = data.copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    unique_dates = pd.Index(sorted(frame[date_column].unique()))

    if len(unique_dates) < 2:
        raise ValueError("Need at least two distinct dates to split.")

    if validation_days is not None:
        if validation_days < 1 or validation_days >= len(unique_dates):
            raise ValueError(
                "validation_days must be >= 1 and smaller than the number "
                "of distinct dates."
            )
        n_validation = validation_days
    else:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be strictly between 0 and 1."
            )
        n_validation = max(1, int(round(len(unique_dates) * validation_fraction)))
        n_validation = min(n_validation, len(unique_dates) - 1)

    split_index = len(unique_dates) - n_validation
    cutoff = pd.Timestamp(unique_dates[split_index - 1])

    train = frame[frame[date_column] <= cutoff].reset_index(drop=True)
    validation = frame[frame[date_column] > cutoff].reset_index(drop=True)

    return ChronologicalSplit(train=train, validation=validation, cutoff=cutoff)
