from pathlib import Path

import pandas as pd

from experiments.make_compas_counterfactual_outputs import (
    generate_compas_counterfactual_outputs,
    make_counterfactual_table,
)


def _toy_counterfactual_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "compas",
                "model": "logistic_regression",
                "counterfactual_method": "dice_random",
                "random_state": 42,
                "n_counterfactual": 10,
                "total_cfs": 1,
                "features_to_vary": "Number_of_Priors",
                "n_features_to_vary": 1,
                "protected_attributes": "African_American|Female",
                "failed_counterfactual_count": 2,
                "counterfactual_generation_runtime_seconds": 1.25,
                "mean_runtime_per_query_seconds": 0.125,
                "accuracy": 0.68,
                "f1": 0.62,
                "roc_auc": 0.74,
                "requested_count": 10.0,
                "valid_counterfactual_count": 8.0,
                "validity_rate": 0.8,
                "mean_l0_distance": 1.0,
                "mean_l1_distance": 12.5,
                "mean_changed_features": 1.0,
                "protected_attribute_change_rate": 0.0,
            },
            {
                "dataset": "compas",
                "model": "random_forest",
                "counterfactual_method": "dice_random",
                "random_state": 42,
                "n_counterfactual": 10,
                "total_cfs": 1,
                "features_to_vary": "Number_of_Priors",
                "n_features_to_vary": 1,
                "protected_attributes": "African_American|Female",
                "failed_counterfactual_count": 1,
                "counterfactual_generation_runtime_seconds": 3.0,
                "mean_runtime_per_query_seconds": 0.3,
                "accuracy": 0.69,
                "f1": 0.63,
                "roc_auc": 0.73,
                "requested_count": 10.0,
                "valid_counterfactual_count": 9.0,
                "validity_rate": 0.9,
                "mean_l0_distance": 1.0,
                "mean_l1_distance": 10.0,
                "mean_changed_features": 1.0,
                "protected_attribute_change_rate": 0.0,
            },
        ]
    )


def test_make_counterfactual_table_formats_expected_columns():
    table = make_counterfactual_table(_toy_counterfactual_results())

    assert len(table) == 2
    assert "valid_counterfactuals" in table.columns
    assert "validity_rate_raw" in table.columns
    assert table.loc[table["model"] == "logistic_regression", "valid_counterfactuals"].iloc[0] == "8/10"
    assert table.loc[table["model"] == "random_forest", "validity_rate"].iloc[0] == "0.900"


def test_generate_compas_counterfactual_outputs_writes_expected_files(tmp_path):
    input_path = tmp_path / "compas_counterfactual_benchmark.csv"
    table_dir = tmp_path / "tables"
    figure_dir = tmp_path / "figures"
    _toy_counterfactual_results().to_csv(input_path, index=False)

    paths = generate_compas_counterfactual_outputs(
        input_path=input_path,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )

    expected = {
        "compas_counterfactuals_table",
        "compas_counterfactual_validity",
        "compas_counterfactual_distance",
        "compas_counterfactual_runtime",
    }
    assert set(paths) == expected

    table_path = Path(paths["compas_counterfactuals_table"])
    assert table_path.exists()
    assert pd.read_csv(table_path).shape[0] == 2

    for key, path in paths.items():
        path = Path(path)
        assert path.exists()
        assert path.stat().st_size > 0
        if key != "compas_counterfactuals_table":
            assert path.suffix == ".png"
