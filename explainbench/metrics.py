"""Evaluation metrics for model behavior and local explanations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


EPS = 1e-12


def _positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    """Return model scores for the positive class."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(float)


def model_performance(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute standard binary classification metrics."""
    y_pred = model.predict(X)
    y_score = _positive_scores(model, X)
    out = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, y_score))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def fairness_by_binary_group(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    group: Sequence[int],
    privileged_value: int = 0,
    unprivileged_value: int = 1,
) -> dict[str, float]:
    """Compute simple group fairness diagnostics for a binary protected attribute.

    Returned values are unprivileged minus privileged unless noted otherwise.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    group = np.asarray(group).astype(int)

    def rate(mask: np.ndarray, numerator: np.ndarray) -> float:
        denom = mask.sum()
        if denom == 0:
            return float("nan")
        return float(numerator[mask].mean())

    priv = group == privileged_value
    unpriv = group == unprivileged_value
    sel_priv = rate(priv, y_pred == 1)
    sel_unpriv = rate(unpriv, y_pred == 1)
    tpr_priv = rate(priv & (y_true == 1), y_pred == 1)
    tpr_unpriv = rate(unpriv & (y_true == 1), y_pred == 1)
    fpr_priv = rate(priv & (y_true == 0), y_pred == 1)
    fpr_unpriv = rate(unpriv & (y_true == 0), y_pred == 1)
    return {
        "selection_rate_privileged": sel_priv,
        "selection_rate_unprivileged": sel_unpriv,
        "demographic_parity_difference": sel_unpriv - sel_priv,
        "disparate_impact_ratio": sel_unpriv / (sel_priv + EPS),
        "equal_opportunity_difference": tpr_unpriv - tpr_priv,
        "false_positive_rate_difference": fpr_unpriv - fpr_priv,
        "equalized_odds_difference_absmax": max(abs(tpr_unpriv - tpr_priv), abs(fpr_unpriv - fpr_priv)),
    }


def sparsity(values: Sequence[float], threshold: float = 1e-8) -> int:
    """Number of non-negligible attribution values."""
    arr = np.asarray(values, dtype=float)
    return int(np.sum(np.abs(arr) > threshold))


def top_k_features(values: Sequence[float], feature_names: Sequence[str], k: int = 5) -> list[str]:
    """Return feature names with largest absolute attribution magnitudes."""
    arr = np.asarray(values, dtype=float)
    idx = np.argsort(np.abs(arr))[::-1][:k]
    return [feature_names[i] for i in idx]


def jaccard_top_k(a: Sequence[float], b: Sequence[float], k: int = 5) -> float:
    """Jaccard similarity between top-k absolute attribution sets."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    top_a = set(np.argsort(np.abs(a))[::-1][:k])
    top_b = set(np.argsort(np.abs(b))[::-1][:k])
    if not top_a and not top_b:
        return 1.0
    return len(top_a & top_b) / len(top_a | top_b)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for attribution vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < EPS:
        return float("nan")
    return float(np.dot(a, b) / denom)


def deletion_fidelity(
    model,
    X: pd.DataFrame,
    explanations: np.ndarray,
    background: pd.Series,
    k: int = 3,
) -> float:
    """Mean score drop after replacing top-k explanation features with background values.

    Larger positive values indicate that the top-ranked features are important to
    the model's score for the explained instances.
    """
    if len(X) == 0:
        return float("nan")
    base_scores = _positive_scores(model, X)
    X_removed = X.copy()
    for row_i, (_, row) in enumerate(X.iterrows()):
        top = np.argsort(np.abs(explanations[row_i]))[::-1][:k]
        cols = X.columns[top]
        X_removed.loc[row.name, cols] = background[cols].values
    removed_scores = _positive_scores(model, X_removed)
    return float(np.mean(base_scores - removed_scores))


def mean_group_attribution_gap(
    explanations: np.ndarray,
    X: pd.DataFrame,
    group_col: str,
    privileged_value: int = 0,
    unprivileged_value: int = 1,
) -> float:
    """Mean absolute attribution magnitude gap by group.

    Positive means the unprivileged group has larger average attribution magnitude.
    """
    if group_col not in X.columns:
        return float("nan")
    group = X[group_col].to_numpy()
    priv = explanations[group == privileged_value]
    unpriv = explanations[group == unprivileged_value]
    if len(priv) == 0 or len(unpriv) == 0:
        return float("nan")
    return float(np.mean(np.abs(unpriv)) - np.mean(np.abs(priv)))


def attribution_stability(
    original_explanations: np.ndarray,
    perturbed_explanations: np.ndarray,
    k: int = 3,
) -> dict[str, float]:
    """Compare original and perturbed explanation vectors.

    The benchmark reports two complementary stability diagnostics:

    1. Top-k Jaccard stability:
       Whether the same top-k absolute-attribution features remain important.

    2. Cosine stability:
       Whether the full attribution vectors point in a similar direction.

    Larger values indicate more stable explanations.
    """
    original = np.asarray(original_explanations, dtype=float)
    perturbed = np.asarray(perturbed_explanations, dtype=float)

    if original.shape != perturbed.shape:
        raise ValueError(
            f"Explanation shape mismatch: original={original.shape}, perturbed={perturbed.shape}"
        )

    if original.ndim != 2:
        raise ValueError(f"Expected 2D explanation arrays, got shape {original.shape}")

    if original.shape[0] == 0:
        return {
            f"stability_top{k}_jaccard_mean": float("nan"),
            f"stability_top{k}_jaccard_std": float("nan"),
            "stability_cosine_mean": float("nan"),
            "stability_cosine_std": float("nan"),
        }

    jaccards = np.array(
        [jaccard_top_k(a, b, k=k) for a, b in zip(original, perturbed)],
        dtype=float,
    )
    cosines = np.array(
        [cosine_similarity(a, b) for a, b in zip(original, perturbed)],
        dtype=float,
    )

    return {
        f"stability_top{k}_jaccard_mean": float(np.nanmean(jaccards)),
        f"stability_top{k}_jaccard_std": float(np.nanstd(jaccards)),
        "stability_cosine_mean": float(np.nanmean(cosines)),
        "stability_cosine_std": float(np.nanstd(cosines)),
    }
