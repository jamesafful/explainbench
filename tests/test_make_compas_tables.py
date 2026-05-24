import pandas as pd

from experiments.make_compas_tables import (
    generate_compas_tables,
    make_explanation_quality_table,
    make_model_performance_table,
)


def _toy_summary():
    return pd.DataFrame(
        [
            {
                "dataset": "compas",
                "model": "logistic_regression",
                "explainer": "lime",
                "n_seeds": 5,
                "accuracy_mean": 0.68,
                "accuracy_std": 0.01,
                "f1_mean": 0.62,
                "f1_std": 0.02,
                "roc_auc_mean": 0.74,
                "roc_auc_std": 0.01,
                "mean_sparsity_mean": 10.9,
                "mean_sparsity_std": 0.1,
                "deletion_fidelity_top3_mean": 0.15,
                "deletion_fidelity_top3_std": 0.01,
            },
            {
                "dataset": "compas",
                "model": "logistic_regression",
                "explainer": "shap_linear",
                "n_seeds": 5,
                "accuracy_mean": 0.68,
                "accuracy_std": 0.01,
                "f1_mean": 0.62,
                "f1_std": 0.02,
                "roc_auc_mean": 0.74,
                "roc_auc_std": 0.01,
                "mean_sparsity_mean": 9.0,
                "mean_sparsity_std": 0.0,
                "deletion_fidelity_top3_mean": 0.11,
                "deletion_fidelity_top3_std": 0.01,
            },
        ]
    )


def test_model_performance_table_deduplicates_model_rows():
    table = make_model_performance_table(_toy_summary())

    assert len(table) == 1
    assert table.iloc[0]["dataset"] == "compas"
    assert table.iloc[0]["model"] == "logistic_regression"
    assert table.iloc[0]["accuracy"] == "0.680 ± 0.010"
    assert table.iloc[0]["f1"] == "0.620 ± 0.020"
    assert table.iloc[0]["roc_auc"] == "0.740 ± 0.010"


def test_explanation_quality_table_keeps_explainer_rows():
    table = make_explanation_quality_table(_toy_summary())

    assert len(table) == 2
    assert set(table["explainer"]) == {"lime", "shap_linear"}
    assert "deletion_fidelity_top3" in table.columns
    assert table.loc[table["explainer"] == "lime", "deletion_fidelity_top3"].iloc[0] == "0.150 ± 0.010"


def test_generate_compas_tables_writes_expected_files(tmp_path):
    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "tables"

    full = _toy_summary()
    for prefix in [
        "explanation_runtime_seconds",
        "mean_runtime_per_instance_seconds",
        "stability_runtime_seconds",
        "stability_top3_jaccard_mean",
        "stability_cosine_mean",
        "African_American_demographic_parity_difference",
        "African_American_equal_opportunity_difference",
        "African_American_false_positive_rate_difference",
        "African_American_mean_abs_attribution_gap",
        "Female_demographic_parity_difference",
        "Female_equal_opportunity_difference",
        "Female_false_positive_rate_difference",
        "Female_mean_abs_attribution_gap",
    ]:
        full[f"{prefix}_mean"] = 0.1
        full[f"{prefix}_std"] = 0.01

    full.to_csv(summary_path, index=False)

    paths = generate_compas_tables(summary_path, output_dir)

    expected = {
        "compas_model_performance",
        "compas_explanation_quality",
        "compas_runtime",
        "compas_stability",
        "compas_fairness",
    }
    assert set(paths) == expected

    for path in paths.values():
        assert path.exists()
        assert pd.read_csv(path).shape[0] > 0
