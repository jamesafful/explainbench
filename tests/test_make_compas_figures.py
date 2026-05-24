from pathlib import Path

import pandas as pd

from experiments.make_compas_figures import generate_compas_figures


def _toy_summary() -> pd.DataFrame:
    rows = []
    for model, explainer in [
        ("logistic_regression", "lime"),
        ("logistic_regression", "shap_linear"),
        ("random_forest", "lime"),
        ("random_forest", "shap_tree"),
    ]:
        rows.append(
            {
                "dataset": "compas",
                "model": model,
                "explainer": explainer,
                "n_seeds": 5,
                "accuracy_mean": 0.68,
                "accuracy_std": 0.01,
                "roc_auc_mean": 0.74,
                "roc_auc_std": 0.01,
                "deletion_fidelity_top3_mean": 0.12,
                "deletion_fidelity_top3_std": 0.02,
                "explanation_runtime_seconds_mean": 1.0,
                "explanation_runtime_seconds_std": 0.1,
                "stability_top3_jaccard_mean_mean": 0.95,
                "stability_top3_jaccard_mean_std": 0.02,
                "African_American_mean_abs_attribution_gap_mean": 0.01,
                "African_American_mean_abs_attribution_gap_std": 0.002,
                "Female_mean_abs_attribution_gap_mean": 0.005,
                "Female_mean_abs_attribution_gap_std": 0.001,
            }
        )
    return pd.DataFrame(rows)


def test_generate_compas_figures_writes_expected_pngs(tmp_path):
    summary_path = tmp_path / "summary.csv"
    output_dir = tmp_path / "figures"
    _toy_summary().to_csv(summary_path, index=False)

    paths = generate_compas_figures(summary_path, output_dir)

    expected = {
        "compas_runtime_vs_fidelity",
        "compas_stability_by_explainer",
        "compas_attribution_gap",
        "compas_model_performance",
    }
    assert set(paths) == expected

    for path in paths.values():
        path = Path(path)
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0
