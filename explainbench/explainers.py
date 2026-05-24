"""Small, dependency-light explainers used by the benchmark runner."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def positive_scores(model, X: pd.DataFrame) -> np.ndarray:
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
