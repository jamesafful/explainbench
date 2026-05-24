"""Benchmark runner for tabular local explanations."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .datasets import load_dataset
from .explainers import (
    LIMETabularExplainer,
    LinearCoefficientExplainer,
    OcclusionExplainer,
    SHAPTabularExplainer,
)
from .metrics import (
    deletion_fidelity,
    fairness_by_binary_group,
    mean_group_attribution_gap,
    model_performance,
    sparsity,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: str = "compas"
    test_size: float = 0.30
    random_state: int = 42
    n_explain: int = 200
    top_k: int = 3
    lime_num_samples: int = 5000


def _models(random_state: int):
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=random_state),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def _timed_explain(explainer, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    values = explainer.explain(X)
    elapsed = time.perf_counter() - start
    values = np.asarray(values, dtype=float)
    if values.shape != (len(X), X.shape[1]):
        raise ValueError(f"Explainer returned shape {values.shape}; expected {(len(X), X.shape[1])}")
    return values, float(elapsed)


def _scaled_frame(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )


def _explain_model(
    model_name: str,
    model,
    X_reference: pd.DataFrame,
    X_explain: pd.DataFrame,
    config: BenchmarkConfig,
) -> dict[str, tuple[np.ndarray, float]]:
    """Return {explainer_name: (attribution_matrix, runtime_seconds)}."""
    background = X_reference.median(numeric_only=True)
    explanations: dict[str, tuple[np.ndarray, float]] = {}

    explanations["occlusion"] = _timed_explain(
        OcclusionExplainer(model=model, background=background),
        X_explain,
    )

    explanations["lime"] = _timed_explain(
        LIMETabularExplainer(
            model=model,
            background=X_reference,
            random_state=config.random_state,
            num_samples=config.lime_num_samples,
        ),
        X_explain,
    )

    if model_name == "logistic_regression":
        # The fitted estimator is the final step of the pipeline; coefficient
        # and linear-SHAP contributions are computed in standardized space.
        scaler = model.named_steps["standardscaler"]
        clf = model.named_steps["logisticregression"]
        X_ref_scaled = _scaled_frame(scaler, X_reference)
        X_exp_scaled = _scaled_frame(scaler, X_explain)

        explanations["linear_coefficients"] = _timed_explain(
            LinearCoefficientExplainer(
                model=clf,
                background=X_ref_scaled.median(numeric_only=True),
            ),
            X_exp_scaled,
        )

        explanations["shap_linear"] = _timed_explain(
            SHAPTabularExplainer(
                model=clf,
                background=X_ref_scaled,
                model_type="linear",
            ),
            X_exp_scaled,
        )

    elif model_name == "random_forest":
        explanations["shap_tree"] = _timed_explain(
            SHAPTabularExplainer(
                model=model,
                background=X_reference,
                model_type="tree",
            ),
            X_explain,
        )

    return explanations


def run_benchmark(config: BenchmarkConfig = BenchmarkConfig()) -> pd.DataFrame:
    """Run a reproducible benchmark and return one row per model/explainer."""
    X, y, spec = load_dataset(config.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    X_explain = X_test.head(config.n_explain).copy()
    rows: list[dict[str, float | str | int]] = []

    for model_name, model in _models(config.random_state).items():
        model.fit(X_train, y_train)
        perf = model_performance(model, X_test, y_test)
        y_pred = model.predict(X_test)

        fairness_rows = {}
        for protected in spec.protected_attributes:
            if protected in X_test.columns:
                group_metrics = fairness_by_binary_group(y_test, y_pred, X_test[protected])
                fairness_rows.update({f"{protected}_{k}": v for k, v in group_metrics.items()})

        explanation_sets = _explain_model(model_name, model, X_train, X_explain, config)
        background = X_train.median(numeric_only=True)

        for explainer_name, (values, runtime_seconds) in explanation_sets.items():
            row = {
                "dataset": config.dataset,
                "model": model_name,
                "explainer": explainer_name,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_explain": len(X_explain),
                "random_state": config.random_state,
                **perf,
                **fairness_rows,
                "mean_sparsity": float(np.mean([sparsity(v) for v in values])),
                f"deletion_fidelity_top{config.top_k}": deletion_fidelity(
                    model, X_explain, values, background=background, k=config.top_k
                ),
                "explanation_runtime_seconds": float(runtime_seconds),
                "mean_runtime_per_instance_seconds": float(runtime_seconds / max(len(X_explain), 1)),
            }
            for protected in spec.protected_attributes:
                row[f"{protected}_mean_abs_attribution_gap"] = mean_group_attribution_gap(
                    values, X_explain, group_col=protected
                )
            rows.append(row)

    return pd.DataFrame(rows)


def save_benchmark(output: str | Path, config: BenchmarkConfig = BenchmarkConfig()) -> Path:
    """Run the benchmark and save CSV output."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = run_benchmark(config)
    df.to_csv(output, index=False)
    return output
