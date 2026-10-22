"""Candidate forecasting models and a comparison helper.

Models are trained on leakage-safe features (see :mod:`.features`) and scored
with the metrics in :mod:`.evaluation`. Everything returns tidy DataFrames so
results can be compared, tested, and reused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import evaluate_forecast

DEFAULT_FEATURES = (
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "is_promoted",
    "onpromotion",
    "store_nbr",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
)

# A smaller, bounded feature set used by the linear Ridge reference so that
# raw-scale columns (store_nbr, onpromotion counts) do not destabilize it.
LINEAR_FEATURES = (
    "month",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "is_promoted",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
)


def _numeric_matrix(frame: pd.DataFrame, features: tuple[str, ...]):
    """Select available feature columns as a float matrix."""
    columns = [column for column in features if column in frame.columns]
    if not columns:
        raise ValueError("None of the requested feature columns are present.")
    return frame[columns].astype(float).fillna(0.0), columns


def train_lightgbm(
    train: pd.DataFrame,
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    target: str = "sales",
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    num_leaves: int = 64,
    seed: int = 42,
):
    """Train a LightGBM regressor on the given features.

    The objective is L2 on ``log1p`` sales so that errors are judged on the
    same scale as the competition's RMSLE metric. Predictions are converted
    back to the original scale.
    """
    from lightgbm import LGBMRegressor

    matrix, columns = _numeric_matrix(train, features)
    target_values = np.log1p(train[target].astype(float).to_numpy())

    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(matrix, target_values)
    return model, columns


def train_ridge(
    train: pd.DataFrame,
    *,
    features: tuple[str, ...] = LINEAR_FEATURES,
    target: str = "sales",
):
    """Train a Ridge regression baseline on ``log1p`` sales.

    A simple, fast linear reference that needs no extra dependencies beyond
    scikit-learn.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    matrix, columns = _numeric_matrix(train, features)
    # log1p the heavy-tailed count features so the linear model is not
    # dominated by raw-scale volume, then standardize before Ridge.
    matrix = np.log1p(np.clip(matrix, 0, None))
    target_values = np.log1p(train[target].astype(float).to_numpy())

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(matrix, target_values)
    # Remember the feature transform so predict() applies it consistently.
    model._rsc_log_features = lambda m: np.log1p(np.clip(m, 0, None))
    return model, columns


def predict(model, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Predict sales in the original scale from a log1p-trained model."""
    matrix = frame[columns].astype(float).fillna(0.0)
    matrix = getattr(model, "_rsc_log_features", lambda m: m)(matrix)
    log_predictions = model.predict(matrix)
    return np.clip(np.expm1(log_predictions), 0, None)


def compare_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    candidates: dict,
    *,
    target: str = "sales",
) -> pd.DataFrame:
    """Train candidates and score them on the validation split.

    ``candidates`` maps a model name to a ``(trainer, feature_columns)`` tuple
    or just a trainer (in which case :data:`DEFAULT_FEATURES` is used).

    Returns a DataFrame indexed by model name with MAE, RMSE, and RMSLE,
    sorted by RMSLE ascending.
    """
    rows = {}
    for name, spec in candidates.items():
        if isinstance(spec, tuple):
            trainer, features = spec
        else:
            trainer, features = spec, DEFAULT_FEATURES

        model, columns = trainer(train, features=features, target=target)
        predictions = predict(model, validation, columns)
        metrics = evaluate_forecast(validation[target], predictions)
        rows[name] = metrics

    result = pd.DataFrame(rows).T
    return result.sort_values("rmsle").reset_index(names="model")


def error_breakdown(
    validation: pd.DataFrame,
    predictions: np.ndarray,
    *,
    by: str = "family",
    target: str = "sales",
) -> pd.DataFrame:
    """Compute mean absolute error grouped by a column such as family or store."""
    frame = validation.copy()
    frame["prediction"] = predictions
    frame["abs_error"] = (frame[target] - frame["prediction"]).abs()
    breakdown = (
        frame.groupby(by)["abs_error"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
        .rename(columns={"mean": "mean_abs_error", "count": "rows"})
    )
    return breakdown
