"""Generate paper-ready COMPAS figures from multi-seed benchmark summaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT = "results/compas_multiseed_summary.csv"
DEFAULT_OUTPUT_DIR = "paper/figures"


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _label(row: pd.Series) -> str:
    return f"{row['model']}\n{row['explainer']}"


def _prepare_summary(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    df = pd.read_csv(path)
    _require_columns(df, ["dataset", "model", "explainer", "n_seeds"])
    return df.sort_values(["dataset", "model", "explainer"]).reset_index(drop=True)


def make_runtime_vs_fidelity_figure(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot explanation runtime against deletion fidelity.

    This figure shows the quality-runtime tradeoff across explainers.
    """
    required = [
        "dataset",
        "model",
        "explainer",
        "deletion_fidelity_top3_mean",
        "deletion_fidelity_top3_std",
        "explanation_runtime_seconds_mean",
        "explanation_runtime_seconds_std",
    ]
    _require_columns(summary, required)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in summary.iterrows():
        ax.errorbar(
            row["explanation_runtime_seconds_mean"],
            row["deletion_fidelity_top3_mean"],
            xerr=row["explanation_runtime_seconds_std"],
            yerr=row["deletion_fidelity_top3_std"],
            fmt="o",
            capsize=3,
        )
        ax.annotate(
            _label(row),
            (
                row["explanation_runtime_seconds_mean"],
                row["deletion_fidelity_top3_mean"],
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Explanation runtime, seconds, log scale")
    ax.set_ylabel("Deletion fidelity, top-3")
    ax.set_title("COMPAS explanation quality-runtime tradeoff")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def make_stability_by_explainer_figure(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot top-3 Jaccard stability by model/explainer."""
    required = [
        "dataset",
        "model",
        "explainer",
        "stability_top3_jaccard_mean_mean",
        "stability_top3_jaccard_mean_std",
    ]
    _require_columns(summary, required)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [_label(row) for _, row in summary.iterrows()]
    values = summary["stability_top3_jaccard_mean_mean"]
    errors = summary["stability_top3_jaccard_mean_std"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(summary)), values, yerr=errors, capsize=3)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-3 Jaccard stability")
    ax.set_title("COMPAS perturbation stability by explainer")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def make_attribution_gap_figure(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot protected-group mean absolute attribution gaps."""
    required = [
        "dataset",
        "model",
        "explainer",
        "African_American_mean_abs_attribution_gap_mean",
        "African_American_mean_abs_attribution_gap_std",
        "Female_mean_abs_attribution_gap_mean",
        "Female_mean_abs_attribution_gap_std",
    ]
    _require_columns(summary, required)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [_label(row) for _, row in summary.iterrows()]
    x = list(range(len(summary)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        [i - width / 2 for i in x],
        summary["African_American_mean_abs_attribution_gap_mean"],
        width,
        yerr=summary["African_American_mean_abs_attribution_gap_std"],
        capsize=3,
        label="African_American",
    )
    ax.bar(
        [i + width / 2 for i in x],
        summary["Female_mean_abs_attribution_gap_mean"],
        width,
        yerr=summary["Female_mean_abs_attribution_gap_std"],
        capsize=3,
        label="Female",
    )
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean absolute attribution gap")
    ax.set_title("COMPAS protected-group attribution gaps")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def make_model_performance_figure(summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot model-level accuracy and ROC-AUC.

    Model metrics are repeated for each explainer row, so the figure
    deduplicates by dataset/model.
    """
    required = [
        "dataset",
        "model",
        "accuracy_mean",
        "accuracy_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]
    _require_columns(summary, required)

    model_df = summary.drop_duplicates(subset=["dataset", "model"]).sort_values(["dataset", "model"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = model_df["model"].tolist()
    x = list(range(len(model_df)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        [i - width / 2 for i in x],
        model_df["accuracy_mean"],
        width,
        yerr=model_df["accuracy_std"],
        capsize=3,
        label="Accuracy",
    )
    ax.bar(
        [i + width / 2 for i in x],
        model_df["roc_auc_mean"],
        width,
        yerr=model_df["roc_auc_std"],
        capsize=3,
        label="ROC-AUC",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("COMPAS model performance")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_compas_figures(input_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Generate all COMPAS paper figures and return output paths."""
    summary = _prepare_summary(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "compas_runtime_vs_fidelity": make_runtime_vs_fidelity_figure(
            summary,
            output_dir / "compas_runtime_vs_fidelity.png",
        ),
        "compas_stability_by_explainer": make_stability_by_explainer_figure(
            summary,
            output_dir / "compas_stability_by_explainer.png",
        ),
        "compas_attribution_gap": make_attribution_gap_figure(
            summary,
            output_dir / "compas_attribution_gap.png",
        ),
        "compas_model_performance": make_model_performance_figure(
            summary,
            output_dir / "compas_model_performance.png",
        ),
    }
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-ready COMPAS figures.")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to compas_multiseed_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where paper figures will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_compas_figures(args.input, args.output_dir)
    for name, path in paths.items():
        print(f"Wrote {name}: {path.resolve()}")


if __name__ == "__main__":
    main()
