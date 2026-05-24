"""Generate paper-ready COMPAS tables from multi-seed benchmark summaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT = "results/compas_multiseed_summary.csv"
DEFAULT_OUTPUT_DIR = "paper/tables"


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _format_mean_std(row: pd.Series, mean_col: str, std_col: str, digits: int = 3) -> str:
    mean = row[mean_col]
    std = row[std_col]

    if pd.isna(mean):
        return "NA"
    if pd.isna(std):
        return f"{mean:.{digits}f} ± NA"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def make_model_performance_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create one row per model with model-performance metrics.

    Model performance does not depend on explainer, so this table deduplicates
    model-level rows from the model/explainer summary.
    """
    required = [
        "dataset",
        "model",
        "n_seeds",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]
    _require_columns(summary, required)

    rows = []
    for _, row in summary.sort_values(["dataset", "model", "explainer"]).iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "n_seeds": int(row["n_seeds"]),
                "accuracy": _format_mean_std(row, "accuracy_mean", "accuracy_std"),
                "f1": _format_mean_std(row, "f1_mean", "f1_std"),
                "roc_auc": _format_mean_std(row, "roc_auc_mean", "roc_auc_std"),
                "accuracy_mean": row["accuracy_mean"],
                "accuracy_std": row["accuracy_std"],
                "f1_mean": row["f1_mean"],
                "f1_std": row["f1_std"],
                "roc_auc_mean": row["roc_auc_mean"],
                "roc_auc_std": row["roc_auc_std"],
            }
        )

    table = pd.DataFrame(rows)
    return table.drop_duplicates(subset=["dataset", "model"]).reset_index(drop=True)


def make_explanation_quality_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create paper-ready explanation-quality table."""
    required = [
        "dataset",
        "model",
        "explainer",
        "n_seeds",
        "mean_sparsity_mean",
        "mean_sparsity_std",
        "deletion_fidelity_top3_mean",
        "deletion_fidelity_top3_std",
    ]
    _require_columns(summary, required)

    rows = []
    for _, row in summary.sort_values(["dataset", "model", "explainer"]).iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "explainer": row["explainer"],
                "n_seeds": int(row["n_seeds"]),
                "mean_sparsity": _format_mean_std(row, "mean_sparsity_mean", "mean_sparsity_std"),
                "deletion_fidelity_top3": _format_mean_std(
                    row,
                    "deletion_fidelity_top3_mean",
                    "deletion_fidelity_top3_std",
                ),
                "mean_sparsity_mean": row["mean_sparsity_mean"],
                "mean_sparsity_std": row["mean_sparsity_std"],
                "deletion_fidelity_top3_mean": row["deletion_fidelity_top3_mean"],
                "deletion_fidelity_top3_std": row["deletion_fidelity_top3_std"],
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def make_runtime_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create paper-ready runtime table."""
    required = [
        "dataset",
        "model",
        "explainer",
        "n_seeds",
        "explanation_runtime_seconds_mean",
        "explanation_runtime_seconds_std",
        "mean_runtime_per_instance_seconds_mean",
        "mean_runtime_per_instance_seconds_std",
        "stability_runtime_seconds_mean",
        "stability_runtime_seconds_std",
    ]
    _require_columns(summary, required)

    rows = []
    for _, row in summary.sort_values(["dataset", "model", "explainer"]).iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "explainer": row["explainer"],
                "n_seeds": int(row["n_seeds"]),
                "explanation_runtime_seconds": _format_mean_std(
                    row,
                    "explanation_runtime_seconds_mean",
                    "explanation_runtime_seconds_std",
                ),
                "mean_runtime_per_instance_seconds": _format_mean_std(
                    row,
                    "mean_runtime_per_instance_seconds_mean",
                    "mean_runtime_per_instance_seconds_std",
                    digits=6,
                ),
                "stability_runtime_seconds": _format_mean_std(
                    row,
                    "stability_runtime_seconds_mean",
                    "stability_runtime_seconds_std",
                ),
                "explanation_runtime_seconds_mean": row["explanation_runtime_seconds_mean"],
                "explanation_runtime_seconds_std": row["explanation_runtime_seconds_std"],
                "mean_runtime_per_instance_seconds_mean": row[
                    "mean_runtime_per_instance_seconds_mean"
                ],
                "mean_runtime_per_instance_seconds_std": row[
                    "mean_runtime_per_instance_seconds_std"
                ],
                "stability_runtime_seconds_mean": row["stability_runtime_seconds_mean"],
                "stability_runtime_seconds_std": row["stability_runtime_seconds_std"],
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def make_stability_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create paper-ready stability table."""
    required = [
        "dataset",
        "model",
        "explainer",
        "n_seeds",
        "stability_top3_jaccard_mean_mean",
        "stability_top3_jaccard_mean_std",
        "stability_cosine_mean_mean",
        "stability_cosine_mean_std",
    ]
    _require_columns(summary, required)

    rows = []
    for _, row in summary.sort_values(["dataset", "model", "explainer"]).iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "explainer": row["explainer"],
                "n_seeds": int(row["n_seeds"]),
                "top3_jaccard_stability": _format_mean_std(
                    row,
                    "stability_top3_jaccard_mean_mean",
                    "stability_top3_jaccard_mean_std",
                ),
                "cosine_stability": _format_mean_std(
                    row,
                    "stability_cosine_mean_mean",
                    "stability_cosine_mean_std",
                ),
                "stability_top3_jaccard_mean_mean": row[
                    "stability_top3_jaccard_mean_mean"
                ],
                "stability_top3_jaccard_mean_std": row[
                    "stability_top3_jaccard_mean_std"
                ],
                "stability_cosine_mean_mean": row["stability_cosine_mean_mean"],
                "stability_cosine_mean_std": row["stability_cosine_mean_std"],
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def make_fairness_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create paper-ready fairness and attribution-gap table."""
    required = [
        "dataset",
        "model",
        "explainer",
        "n_seeds",
        "African_American_demographic_parity_difference_mean",
        "African_American_demographic_parity_difference_std",
        "African_American_equal_opportunity_difference_mean",
        "African_American_equal_opportunity_difference_std",
        "African_American_false_positive_rate_difference_mean",
        "African_American_false_positive_rate_difference_std",
        "African_American_mean_abs_attribution_gap_mean",
        "African_American_mean_abs_attribution_gap_std",
        "Female_demographic_parity_difference_mean",
        "Female_demographic_parity_difference_std",
        "Female_equal_opportunity_difference_mean",
        "Female_equal_opportunity_difference_std",
        "Female_false_positive_rate_difference_mean",
        "Female_false_positive_rate_difference_std",
        "Female_mean_abs_attribution_gap_mean",
        "Female_mean_abs_attribution_gap_std",
    ]
    _require_columns(summary, required)

    rows = []
    for _, row in summary.sort_values(["dataset", "model", "explainer"]).iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "explainer": row["explainer"],
                "n_seeds": int(row["n_seeds"]),
                "african_american_demographic_parity_difference": _format_mean_std(
                    row,
                    "African_American_demographic_parity_difference_mean",
                    "African_American_demographic_parity_difference_std",
                ),
                "african_american_equal_opportunity_difference": _format_mean_std(
                    row,
                    "African_American_equal_opportunity_difference_mean",
                    "African_American_equal_opportunity_difference_std",
                ),
                "african_american_false_positive_rate_difference": _format_mean_std(
                    row,
                    "African_American_false_positive_rate_difference_mean",
                    "African_American_false_positive_rate_difference_std",
                ),
                "african_american_mean_abs_attribution_gap": _format_mean_std(
                    row,
                    "African_American_mean_abs_attribution_gap_mean",
                    "African_American_mean_abs_attribution_gap_std",
                ),
                "female_demographic_parity_difference": _format_mean_std(
                    row,
                    "Female_demographic_parity_difference_mean",
                    "Female_demographic_parity_difference_std",
                ),
                "female_equal_opportunity_difference": _format_mean_std(
                    row,
                    "Female_equal_opportunity_difference_mean",
                    "Female_equal_opportunity_difference_std",
                ),
                "female_false_positive_rate_difference": _format_mean_std(
                    row,
                    "Female_false_positive_rate_difference_mean",
                    "Female_false_positive_rate_difference_std",
                ),
                "female_mean_abs_attribution_gap": _format_mean_std(
                    row,
                    "Female_mean_abs_attribution_gap_mean",
                    "Female_mean_abs_attribution_gap_std",
                ),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def generate_compas_tables(input_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Generate all COMPAS paper tables and return output paths."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    summary = pd.read_csv(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "compas_model_performance": make_model_performance_table(summary),
        "compas_explanation_quality": make_explanation_quality_table(summary),
        "compas_runtime": make_runtime_table(summary),
        "compas_stability": make_stability_table(summary),
        "compas_fairness": make_fairness_table(summary),
    }

    paths = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready COMPAS result tables.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to compas_multiseed_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where paper tables will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_compas_tables(args.input, args.output_dir)
    for name, path in paths.items():
        print(f"Wrote {name}: {path.resolve()}")


if __name__ == "__main__":
    main()
