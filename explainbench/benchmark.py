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
    attribution_stability,
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
    n_stability: int = 50
    stability_noise_scale: float = 0.01
    stability_random_state: int = 123


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


def _is_binary_indicator(series: pd.Series) -> bool:
    values = set(series.dropna().unique().tolist())
    return values.issubset({0, 1, 0.0, 1.0})


def _make_stability_perturbation(
    X: pd.DataFrame,
    reference: pd.DataFrame,
    protected_attributes: set[str],
    noise_scale: float,
    random_state: int,
) -> pd.DataFrame:
    """Create small, reproducible perturbations for stability evaluation.

    For fairness-critical tabular data, we avoid perturbing protected attributes
    and binary indicator columns. On the current COMPAS dataset, this means the
    perturbation affects Number_of_Priors while leaving race, sex, age-bin,
    charge-type, and one-hot indicators unchanged.
    """
    rng = np.random.default_rng(random_state)
    X_perturbed = X.copy()

    for col in X.columns:
        if col in protected_attributes:
            continue
        if _is_binary_indicator(reference[col]):
            continue

        std = float(reference[col].std())
        if not np.isfinite(std) or std <= 0:
            std = 1.0

        noise = rng.normal(loc=0.0, scale=noise_scale * std, size=len(X))
        min_value = float(reference[col].min())
        max_value = float(reference[col].max())
        X_perturbed[col] = np.clip(X_perturbed[col].astype(float) + noise, min_value, max_value)

    return X_perturbed


def _evaluate_explainer(
    explainer,
    X_explain: pd.DataFrame,
    X_stability: pd.DataFrame,
    top_k: int,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Explain X and compute perturbation-based stability diagnostics."""
    values, runtime_seconds = _timed_explain(explainer, X_explain)

    n_stability = min(len(X_stability), len(X_explain))
    if n_stability == 0:
        stability = attribution_stability(values[:0], values[:0], k=top_k)
        stability["stability_n"] = 0
        stability["stability_runtime_seconds"] = 0.0
        return values, runtime_seconds, stability

    start = time.perf_counter()
    perturbed_values = explainer.explain(X_stability.head(n_stability))
    stability_runtime_seconds = time.perf_counter() - start

    perturbed_values = np.asarray(perturbed_values, dtype=float)
    expected_shape = (n_stability, X_explain.shape[1])
    if perturbed_values.shape != expected_shape:
        raise ValueError(f"Perturbed explanations shape {perturbed_values.shape}; expected {expected_shape}")

    stability = attribution_stability(values[:n_stability], perturbed_values, k=top_k)
    stability["stability_n"] = int(n_stability)
    stability["stability_runtime_seconds"] = float(stability_runtime_seconds)
    return values, runtime_seconds, stability


def _explain_model(
    model_name: str,
    model,
    X_reference: pd.DataFrame,
    X_explain: pd.DataFrame,
    X_stability: pd.DataFrame,
    config: BenchmarkConfig,
) -> dict[str, tuple[np.ndarray, float, dict[str, float]]]:
    """Return {explainer_name: (attribution_matrix, runtime_seconds, stability_metrics)}."""
    background = X_reference.median(numeric_only=True)
    explanations: dict[str, tuple[np.ndarray, float, dict[str, float]]] = {}

    explanations["occlusion"] = _evaluate_explainer(
        OcclusionExplainer(model=model, background=background),
        X_explain,
        X_stability,
        config.top_k,
    )

    explanations["lime"] = _evaluate_explainer(
        LIMETabularExplainer(
            model=model,
            background=X_reference,
            random_state=config.random_state,
            num_samples=config.lime_num_samples,
        ),
        X_explain,
        X_stability,
        config.top_k,
    )

    if model_name == "logistic_regression":
        # The fitted estimator is the final step of the pipeline; coefficient
        # and linear-SHAP contributions are computed in standardized space.
        scaler = model.named_steps["standardscaler"]
        clf = model.named_steps["logisticregression"]
        X_ref_scaled = _scaled_frame(scaler, X_reference)
        X_exp_scaled = _scaled_frame(scaler, X_explain)
        X_stab_scaled = _scaled_frame(scaler, X_stability)

        explanations["linear_coefficients"] = _evaluate_explainer(
            LinearCoefficientExplainer(
                model=clf,
                background=X_ref_scaled.median(numeric_only=True),
            ),
            X_exp_scaled,
            X_stab_scaled,
            config.top_k,
        )

        explanations["shap_linear"] = _evaluate_explainer(
            SHAPTabularExplainer(
                model=clf,
                background=X_ref_scaled,
                model_type="linear",
            ),
            X_exp_scaled,
            X_stab_scaled,
            config.top_k,
        )

    elif model_name == "random_forest":
        explanations["shap_tree"] = _evaluate_explainer(
            SHAPTabularExplainer(
                model=model,
                background=X_reference,
                model_type="tree",
            ),
            X_explain,
            X_stability,
            config.top_k,
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
    X_stability_base = X_explain.head(min(config.n_stability, len(X_explain))).copy()
    X_stability = _make_stability_perturbation(
        X_stability_base,
        reference=X_train,
        protected_attributes=set(spec.protected_attributes),
        noise_scale=config.stability_noise_scale,
        random_state=config.stability_random_state,
    )

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

        explanation_sets = _explain_model(model_name, model, X_train, X_explain, X_stability, config)
        background = X_train.median(numeric_only=True)

        for explainer_name, (values, runtime_seconds, stability_rows) in explanation_sets.items():
            row = {
                "dataset": config.dataset,
                "model": model_name,
                "explainer": explainer_name,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_explain": len(X_explain),
                "n_stability": len(X_stability),
                "random_state": config.random_state,
                "stability_random_state": config.stability_random_state,
                "stability_noise_scale": config.stability_noise_scale,
                **perf,
                **fairness_rows,
                "mean_sparsity": float(np.mean([sparsity(v) for v in values])),
                f"deletion_fidelity_top{config.top_k}": deletion_fidelity(
                    model, X_explain, values, background=background, k=config.top_k
                ),
                "explanation_runtime_seconds": float(runtime_seconds),
                "mean_runtime_per_instance_seconds": float(runtime_seconds / max(len(X_explain), 1)),
                **stability_rows,
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
