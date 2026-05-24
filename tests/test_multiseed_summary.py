import numpy as np
import pandas as pd

from experiments.run_compas_multiseed import summarize_multiseed


def test_summarize_multiseed_computes_mean_std_and_seed_count():
    df = pd.DataFrame(
        [
            {
                "dataset": "compas",
                "model": "logistic_regression",
                "explainer": "lime",
                "random_state": 0,
                "accuracy": 0.60,
                "deletion_fidelity_top3": 0.10,
            },
            {
                "dataset": "compas",
                "model": "logistic_regression",
                "explainer": "lime",
                "random_state": 1,
                "accuracy": 0.80,
                "deletion_fidelity_top3": 0.20,
            },
        ]
    )

    summary = summarize_multiseed(df, metrics=["accuracy", "deletion_fidelity_top3"])

    assert len(summary) == 1
    row = summary.iloc[0]

    assert row["dataset"] == "compas"
    assert row["model"] == "logistic_regression"
    assert row["explainer"] == "lime"
    assert row["n_seeds"] == 2

    assert np.isclose(row["accuracy_mean"], 0.70)
    assert np.isclose(row["accuracy_std"], np.std([0.60, 0.80], ddof=1))

    assert np.isclose(row["deletion_fidelity_top3_mean"], 0.15)
    assert np.isclose(row["deletion_fidelity_top3_std"], np.std([0.10, 0.20], ddof=1))


def test_summarize_multiseed_rejects_missing_random_state():
    df = pd.DataFrame(
        [
            {
                "dataset": "compas",
                "model": "random_forest",
                "explainer": "shap_tree",
                "accuracy": 0.70,
            }
        ]
    )

    try:
        summarize_multiseed(df, metrics=["accuracy"])
    except ValueError as exc:
        assert "random_state" in str(exc)
    else:
        raise AssertionError("Expected ValueError when random_state is missing.")
