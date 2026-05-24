"""Explainers used by the tabular benchmark runner.

Each explainer exposes:

    explain(X: pd.DataFrame) -> np.ndarray

and returns a dense attribution matrix with shape:

    (n_instances, n_features)

Rows align with X rows. Columns align with X.columns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def positive_scores(model, X: pd.DataFrame) -> np.ndarray:
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


@dataclass
class OcclusionExplainer:
    """Model-agnostic local feature importance via single-feature replacement.

    For each instance and feature, replace that feature with a background median
    value and measure the drop in positive-class score.
    """

    model: object
    background: pd.Series

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        base = positive_scores(self.model, X)
        values = np.zeros((len(X), X.shape[1]), dtype=float)
        for j, col in enumerate(X.columns):
            perturbed = X.copy()
            perturbed[col] = self.background[col]
            values[:, j] = base - positive_scores(self.model, perturbed)
        return values


@dataclass
class LinearCoefficientExplainer:
    """Local contribution approximation for fitted linear models."""

    model: object
    background: pd.Series

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self.model, "coef_"):
            raise TypeError("LinearCoefficientExplainer requires a fitted linear model with coef_.")
        coefs = np.ravel(self.model.coef_)
        centered = X.to_numpy(dtype=float) - self.background[X.columns].to_numpy(dtype=float)
        return centered * coefs


def _normalize_shap_values(raw_values, n_features: int) -> np.ndarray:
    """Convert SHAP outputs across versions/model types to a dense 2D array.

    SHAP can return:
    - a numpy array of shape (n_samples, n_features)
    - a numpy array of shape (n_samples, n_features, n_classes)
    - a list of class-specific arrays
    - an Explanation object with a .values attribute

    For binary classification, this function selects the positive-class values
    when class-specific values are present.
    """
    values = getattr(raw_values, "values", raw_values)

    if isinstance(values, list):
        if len(values) == 1:
            values = values[0]
        else:
            values = values[1]

    arr = np.asarray(values, dtype=float)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.ndim == 3:
        # Common binary classifier shape: (n_samples, n_features, n_classes)
        if arr.shape[1] == n_features:
            arr = arr[:, :, 1]
        # Less common shape: (n_classes, n_samples, n_features)
        elif arr.shape[2] == n_features:
            arr = arr[1, :, :]
        else:
            raise ValueError(f"Cannot normalize SHAP array with shape {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values after normalization, got shape {arr.shape}")

    if arr.shape[1] != n_features:
        raise ValueError(f"SHAP feature dimension mismatch: expected {n_features}, got {arr.shape[1]}")

    return arr


@dataclass
class SHAPTabularExplainer:
    """Benchmark-grade SHAP explainer returning dense attribution arrays.

    Parameters
    ----------
    model:
        A fitted sklearn estimator. For pipeline models, pass either the full
        pipeline or the already-extracted estimator depending on the data space.
    background:
        Background data in the same feature space expected by `model`.
    model_type:
        One of {"tree", "linear", "kernel"}.
    """

    model: object
    background: pd.DataFrame
    model_type: str = "tree"

    def __post_init__(self) -> None:
        import shap

        self.feature_names_ = list(self.background.columns)
        background = self.background.astype(float)

        if self.model_type == "tree":
            self.explainer_ = shap.TreeExplainer(self.model, background)
        elif self.model_type == "linear":
            self.explainer_ = shap.LinearExplainer(self.model, background)
        elif self.model_type == "kernel":
            self.explainer_ = shap.KernelExplainer(self.model.predict_proba, background)
        else:
            raise ValueError("model_type must be one of {'tree', 'linear', 'kernel'}")

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        X = X.astype(float)

        if self.model_type == "kernel":
            raw = self.explainer_.shap_values(X, silent=True)
        else:
            try:
                raw = self.explainer_.shap_values(X, check_additivity=False)
            except TypeError:
                raw = self.explainer_.shap_values(X)

        return _normalize_shap_values(raw, n_features=X.shape[1])


@dataclass
class LIMETabularExplainer:
    """Benchmark-grade LIME explainer returning dense attribution arrays.

    LIME explanations are generated one instance at a time. We set
    discretize_continuous=False so local_exp feature IDs map directly to
    original column indices.
    """

    model: object
    background: pd.DataFrame
    random_state: int = 42
    num_samples: int = 5000

    def __post_init__(self) -> None:
        import lime.lime_tabular

        self.feature_names_ = list(self.background.columns)
        self.explainer_ = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.background.to_numpy(dtype=float),
            feature_names=self.feature_names_,
            class_names=["0", "1"],
            mode="classification",
            discretize_continuous=False,
            random_state=self.random_state,
        )

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        X = X.astype(float)
        values = np.zeros((len(X), X.shape[1]), dtype=float)

        def predict_fn(array: np.ndarray) -> np.ndarray:
            frame = pd.DataFrame(array, columns=self.feature_names_)
            return self.model.predict_proba(frame)

        for row_i, (_, row) in enumerate(X.iterrows()):
            exp = self.explainer_.explain_instance(
                data_row=row.to_numpy(dtype=float),
                predict_fn=predict_fn,
                labels=(1,),
                num_features=X.shape[1],
                num_samples=self.num_samples,
            )
            for feature_idx, weight in exp.local_exp.get(1, []):
                values[row_i, int(feature_idx)] = float(weight)

        return values
