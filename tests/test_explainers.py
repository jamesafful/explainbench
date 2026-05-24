import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from explainbench.explainers import (
    LIMETabularExplainer,
    LinearCoefficientExplainer,
    OcclusionExplainer,
    SHAPTabularExplainer,
)


def _toy_data():
    X = pd.DataFrame(
        {
            "x1": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
            "x2": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def test_occlusion_explainer_returns_dense_matrix():
    X, y = _toy_data()
    model = LogisticRegression().fit(X, y)
    explainer = OcclusionExplainer(model=model, background=X.median(numeric_only=True))
    values = explainer.explain(X.head(3))
    assert values.shape == (3, 2)
    assert np.isfinite(values).all()


def test_linear_coefficient_explainer_returns_dense_matrix():
    X, y = _toy_data()
    model = LogisticRegression().fit(X, y)
    explainer = LinearCoefficientExplainer(model=model, background=X.median(numeric_only=True))
    values = explainer.explain(X.head(3))
    assert values.shape == (3, 2)
    assert np.isfinite(values).all()


def test_lime_explainer_returns_dense_matrix():
    X, y = _toy_data()
    model = LogisticRegression().fit(X, y)
    explainer = LIMETabularExplainer(
        model=model,
        background=X,
        random_state=42,
        num_samples=100,
    )
    values = explainer.explain(X.head(2))
    assert values.shape == (2, 2)
    assert np.isfinite(values).all()


def test_shap_linear_explainer_returns_dense_matrix():
    X, y = _toy_data()
    model = LogisticRegression().fit(X, y)
    explainer = SHAPTabularExplainer(model=model, background=X, model_type="linear")
    values = explainer.explain(X.head(2))
    assert values.shape == (2, 2)
    assert np.isfinite(values).all()
