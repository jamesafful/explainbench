"""Run a COMPAS counterfactual benchmark with DiCE.

This benchmark is intentionally separate from the attribution benchmark because
DiCE generates counterfactual examples rather than feature-attribution vectors.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainbench.counterfactual_metrics import summarize_counterfactuals


TARGET_COLUMN = "Two_yr_Recidivism"
PROTECTED_ATTRIBUTES = ["African_American", "Female"]
DEFAULT_DATASET_PATH = "datasets/compas_clean.csv"
DEFAULT_OUTPUT_PATH = "results/compas_counterfactual_benchmark.csv"


def _load_compas(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r} in {path}.")

    y = data[TARGET_COLUMN].astype(int)
    X = data.drop(columns=[TARGET_COLUMN])
    return X, y


def _positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X)


def _model_performance(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_score = _positive_scores(model, X_test)

    try:
        roc_auc = roc_auc_score(y_test, y_score)
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc),
    }


def _is_binary_indicator(series: pd.Series) -> bool:
    values = set(series.dropna().unique().tolist())
    return values.issubset({0, 1, 0.0, 1.0})


def _actionable_features(X_train: pd.DataFrame, protected_attributes: list[str]) -> list[str]:
    """Return features allowed to vary in counterfactual search.

    We exclude protected attributes and binary indicator columns to avoid
    generating infeasible one-hot/protected-attribute edits.
    """
    actionable = []
    for col in X_train.columns:
        if col in protected_attributes:
            continue
        if _is_binary_indicator(X_train[col]):
            continue
        actionable.append(col)
    return actionable


def _permitted_range(X_train: pd.DataFrame, features_to_vary: list[str]) -> dict[str, list[float]]:
    return {
        col: [float(X_train[col].min()), float(X_train[col].max())]
        for col in features_to_vary
    }


def _build_models(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def _extract_first_counterfactual(dice_exp: Any, feature_columns: list[str]) -> pd.DataFrame | None:
    """Extract the first generated counterfactual from a DiCE explanation object."""
    cf_examples = getattr(dice_exp, "cf_examples_list", None)
    if not cf_examples:
        return None

    example = cf_examples[0]
    final_cfs_df = getattr(example, "final_cfs_df", None)

    if final_cfs_df is None or len(final_cfs_df) == 0:
        return None

    counterfactual = final_cfs_df.iloc[[0]].copy()

    missing = [col for col in feature_columns if col not in counterfactual.columns]
    if missing:
        raise ValueError(f"DiCE output is missing feature columns: {missing}")

    return counterfactual[feature_columns]


def _generate_dice_counterfactuals(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_query: pd.DataFrame,
    features_to_vary: list[str],
    total_cfs: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int, float]:
    """Generate one aligned DataFrame of valid DiCE counterfactuals."""
    try:
        import dice_ml
    except ImportError as exc:
        raise ImportError(
            "dice-ml is required for this benchmark. Install project requirements first."
        ) from exc

    train_data = X_train.copy()
    train_data[TARGET_COLUMN] = y_train.to_numpy()

    continuous_features = [
        col for col in X_train.columns if not _is_binary_indicator(X_train[col])
    ]

    dice_data = dice_ml.Data(
        dataframe=train_data,
        continuous_features=continuous_features,
        outcome_name=TARGET_COLUMN,
    )
    dice_model = dice_ml.Model(
        model=model,
        backend="sklearn",
        model_type="classifier",
    )
    dice_explainer = dice_ml.Dice(
        dice_data,
        dice_model,
        method="random",
    )

    permitted_range = _permitted_range(X_train, features_to_vary)

    valid_originals = []
    valid_counterfactuals = []
    failed_count = 0

    start = time.perf_counter()

    for i, (_, query_row) in enumerate(X_query.iterrows()):
        query_instance = query_row.to_frame().T

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dice_exp = dice_explainer.generate_counterfactuals(
                    query_instance,
                    total_CFs=total_cfs,
                    desired_class="opposite",
                    features_to_vary=features_to_vary,
                    permitted_range=permitted_range,
                    random_seed=random_seed + i,
                    verbose=False,
                )

            counterfactual = _extract_first_counterfactual(dice_exp, list(X_train.columns))

            if counterfactual is None:
                failed_count += 1
                continue

            original_prediction = int(model.predict(query_instance)[0])
            counterfactual_prediction = int(model.predict(counterfactual)[0])

            if counterfactual_prediction == original_prediction:
                failed_count += 1
                continue

            valid_originals.append(query_instance[list(X_train.columns)])
            valid_counterfactuals.append(counterfactual[list(X_train.columns)])

        except Exception:
            failed_count += 1
            continue

    runtime = time.perf_counter() - start

    if valid_originals:
        original_df = pd.concat(valid_originals, ignore_index=True)
        counterfactual_df = pd.concat(valid_counterfactuals, ignore_index=True)
    else:
        original_df = pd.DataFrame(columns=X_train.columns)
        counterfactual_df = pd.DataFrame(columns=X_train.columns)

    return original_df, counterfactual_df, len(X_query), failed_count, runtime


def run_compas_counterfactual_benchmark(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    random_state: int = 42,
    n_counterfactual: int = 50,
    total_cfs: int = 1,
) -> pd.DataFrame:
    """Run the COMPAS DiCE counterfactual benchmark."""
    X, y = _load_compas(dataset_path)

    missing_protected = [col for col in PROTECTED_ATTRIBUTES if col not in X.columns]
    if missing_protected:
        raise ValueError(f"Missing protected attributes in COMPAS data: {missing_protected}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )

    features_to_vary = _actionable_features(X_train, PROTECTED_ATTRIBUTES)
    if not features_to_vary:
        raise ValueError("No actionable non-protected, non-binary features are available.")

    X_query = X_test.head(n_counterfactual).copy()

    rows = []
    models = _build_models(random_state=random_state)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        performance = _model_performance(model, X_test, y_test)

        (
            valid_originals,
            valid_counterfactuals,
            requested_count,
            failed_count,
            runtime,
        ) = _generate_dice_counterfactuals(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_query=X_query,
            features_to_vary=features_to_vary,
            total_cfs=total_cfs,
            random_seed=random_state,
        )

        metrics = summarize_counterfactuals(
            original=valid_originals,
            counterfactuals=valid_counterfactuals,
            feature_columns=list(X.columns),
            protected_attributes=PROTECTED_ATTRIBUTES,
            requested_count=requested_count,
        )

        row = {
            "dataset": "compas",
            "model": model_name,
            "counterfactual_method": "dice_random",
            "random_state": random_state,
            "n_counterfactual": n_counterfactual,
            "total_cfs": total_cfs,
            "features_to_vary": "|".join(features_to_vary),
            "n_features_to_vary": len(features_to_vary),
            "protected_attributes": "|".join(PROTECTED_ATTRIBUTES),
            "failed_counterfactual_count": failed_count,
            "counterfactual_generation_runtime_seconds": runtime,
            "mean_runtime_per_query_seconds": runtime / requested_count if requested_count else float("nan"),
        }
        row.update(performance)
        row.update(metrics)
        rows.append(row)

    results = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COMPAS DiCE counterfactual benchmark.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help="Path to cleaned COMPAS CSV.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Train/test split and model random state.",
    )
    parser.add_argument(
        "--n-counterfactual",
        type=int,
        default=50,
        help="Number of test instances to query for counterfactuals.",
    )
    parser.add_argument(
        "--total-cfs",
        type=int,
        default=1,
        help="Number of counterfactuals requested from DiCE per query.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_compas_counterfactual_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        random_state=args.random_state,
        n_counterfactual=args.n_counterfactual,
        total_cfs=args.total_cfs,
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
