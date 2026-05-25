"""Metrics for counterfactual explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_dataframe(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    return frame


def _validate_original_and_counterfactuals(
    original: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    _as_dataframe(original, "original")
    _as_dataframe(counterfactuals, "counterfactuals")

    missing_original = [col for col in feature_columns if col not in original.columns]
    missing_counterfactual = [col for col in feature_columns if col not in counterfactuals.columns]

    if missing_original:
        raise ValueError(f"Original data is missing feature columns: {missing_original}")
    if missing_counterfactual:
        raise ValueError(f"Counterfactual data is missing feature columns: {missing_counterfactual}")
    if len(original) != len(counterfactuals):
        raise ValueError(
            "Original and counterfactual DataFrames must have the same number of rows. "
            f"Got {len(original)} and {len(counterfactuals)}."
        )


def changed_feature_matrix(
    original: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    feature_columns: list[str],
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """Return a boolean matrix indicating which features changed.

    Parameters
    ----------
    original:
        Original instances.
    counterfactuals:
        Counterfactual instances aligned row-by-row with ``original``.
    feature_columns:
        Features to compare.
    tolerance:
        Absolute tolerance for numeric comparisons.
    """
    _validate_original_and_counterfactuals(original, counterfactuals, feature_columns)

    changed = {}
    for col in feature_columns:
        left = original[col].to_numpy()
        right = counterfactuals[col].to_numpy()

        if np.issubdtype(original[col].dtype, np.number) and np.issubdtype(
            counterfactuals[col].dtype, np.number
        ):
            changed[col] = ~np.isclose(left, right, atol=tolerance, rtol=0.0, equal_nan=True)
        else:
            changed[col] = left != right

    return pd.DataFrame(changed, index=original.index)


def mean_l0_distance(
    original: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    feature_columns: list[str],
    tolerance: float = 1e-8,
) -> float:
    """Mean number of changed features per valid counterfactual."""
    changes = changed_feature_matrix(original, counterfactuals, feature_columns, tolerance)
    if len(changes) == 0:
        return float("nan")
    return float(changes.sum(axis=1).mean())


def mean_l1_distance(
    original: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    feature_columns: list[str],
) -> float:
    """Mean absolute L1 distance across feature columns."""
    _validate_original_and_counterfactuals(original, counterfactuals, feature_columns)
    if len(original) == 0:
        return float("nan")

    left = original[feature_columns].astype(float).to_numpy()
    right = counterfactuals[feature_columns].astype(float).to_numpy()
    return float(np.abs(left - right).sum(axis=1).mean())


def mean_changed_features(
    original: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    feature_columns: list[str],
    tolerance: float = 1e-8,
) -> float:
    """Alias for mean L0 distance, reported with a descriptive name."""
    return mean_l0_distance(original, counterfactuals, feature_columns, tolerance)


def protected_attribute_change_rate(
    original: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    protected_attributes: list[str],
    tolerance: float = 1e-8,
) -> float:
    """Fraction of valid counterfactuals that changed at least one protected attribute."""
    if not protected_attributes:
        return float("nan")

    changes = changed_feature_matrix(original, counterfactuals, protected_attributes, tolerance)
    if len(changes) == 0:
        return float("nan")
    return float(changes.any(axis=1).mean())


def validity_rate(
    requested_count: int,
    valid_count: int,
) -> float:
    """Fraction of query instances for which a valid counterfactual was obtained."""
    if requested_count < 0 or valid_count < 0:
        raise ValueError("requested_count and valid_count must be nonnegative.")
    if valid_count > requested_count:
        raise ValueError("valid_count cannot exceed requested_count.")
    if requested_count == 0:
        return float("nan")
    return float(valid_count / requested_count)


def summarize_counterfactuals(
    original: pd.DataFrame,
    counterfactuals: pd.DataFrame,
    feature_columns: list[str],
    protected_attributes: list[str],
    requested_count: int,
) -> dict[str, float]:
    """Compute standard counterfactual summary metrics."""
    valid_count = len(counterfactuals)

    return {
        "requested_count": float(requested_count),
        "valid_counterfactual_count": float(valid_count),
        "validity_rate": validity_rate(requested_count, valid_count),
        "mean_l0_distance": mean_l0_distance(original, counterfactuals, feature_columns),
        "mean_l1_distance": mean_l1_distance(original, counterfactuals, feature_columns),
        "mean_changed_features": mean_changed_features(original, counterfactuals, feature_columns),
        "protected_attribute_change_rate": protected_attribute_change_rate(
            original,
            counterfactuals,
            protected_attributes,
        ),
    }
