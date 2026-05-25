"""Generate paper-ready COMPAS counterfactual tables and figures."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_INPUT = "results/compas_counterfactual_benchmark.csv"
DEFAULT_TABLE_DIR = "paper/tables"
DEFAULT_FIGURE_DIR = "paper/figures"


REQUIRED_COLUMNS = [
    "dataset",
    "model",
    "counterfactual_method",
    "random_state",
    "n_counterfactual",
    "total_cfs",
    "features_to_vary",
    "n_features_to_vary",
    "protected_attributes",
    "failed_counterfactual_count",
    "counterfactual_generation_runtime_seconds",
    "mean_runtime_per_query_seconds",
    "accuracy",
    "f1",
    "roc_auc",
    "requested_count",
    "valid_counterfactual_count",
    "validity_rate",
    "mean_l0_distance",
    "mean_l1_distance",
    "mean_changed_features",
    "protected_attribute_change_rate",
]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _read_counterfactual_results(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Counterfactual result file not found: {input_path}")

    df = pd.read_csv(input_path)
    _require_columns(df, REQUIRED_COLUMNS)
    return df.sort_values(["dataset", "model", "counterfactual_method"]).reset_index(drop=True)


def _fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _model_label(model_name: str) -> str:
    return str(model_name).replace("_", " ")


def make_counterfactual_table(results: pd.DataFrame) -> pd.DataFrame:
    """Create a paper-ready counterfactual summary table."""
    _require_columns(results, REQUIRED_COLUMNS)

    rows = []
    for _, row in results.sort_values(["dataset", "model", "counterfactual_method"]).iterrows():
        requested = int(row["requested_count"])
        valid = int(row["valid_counterfactual_count"])
        failed = int(row["failed_counterfactual_count"])

        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "counterfactual_method": row["counterfactual_method"],
                "n_counterfactual": int(row["n_counterfactual"]),
                "valid_counterfactuals": f"{valid}/{requested}",
                "failed_counterfactual_count": failed,
                "validity_rate": _fmt(row["validity_rate"]),
                "mean_l0_distance": _fmt(row["mean_l0_distance"]),
                "mean_l1_distance": _fmt(row["mean_l1_distance"]),
                "mean_changed_features": _fmt(row["mean_changed_features"]),
                "protected_attribute_change_rate": _fmt(
                    row["protected_attribute_change_rate"]
                ),
                "runtime_seconds": _fmt(
                    row["counterfactual_generation_runtime_seconds"]
                ),
                "mean_runtime_per_query_seconds": _fmt(
                    row["mean_runtime_per_query_seconds"]
                ),
                "features_to_vary": row["features_to_vary"],
                "protected_attributes": row["protected_attributes"],
                "validity_rate_raw": row["validity_rate"],
                "mean_l0_distance_raw": row["mean_l0_distance"],
                "mean_l1_distance_raw": row["mean_l1_distance"],
                "protected_attribute_change_rate_raw": row[
                    "protected_attribute_change_rate"
                ],
                "runtime_seconds_raw": row[
                    "counterfactual_generation_runtime_seconds"
                ],
            }
        )

    return pd.DataFrame(rows)


def make_validity_figure(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot counterfactual validity rate by model."""
    _require_columns(
        results,
        [
            "model",
            "validity_rate",
            "valid_counterfactual_count",
            "requested_count",
        ],
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [_model_label(model) for model in results["model"]]
    x = range(len(results))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, results["validity_rate"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validity rate")
    ax.set_title("COMPAS DiCE counterfactual validity")
    ax.grid(True, axis="y", alpha=0.3)

    for i, (_, row) in enumerate(results.iterrows()):
        ax.text(
            i,
            row["validity_rate"] + 0.03,
            f"{int(row['valid_counterfactual_count'])}/{int(row['requested_count'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def make_distance_figure(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot counterfactual L1 distance by model."""
    _require_columns(results, ["model", "mean_l0_distance", "mean_l1_distance"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [_model_label(model) for model in results["model"]]
    x = range(len(results))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, results["mean_l1_distance"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean L1 distance")
    ax.set_title("COMPAS DiCE counterfactual distance")
    ax.grid(True, axis="y", alpha=0.3)

    for i, (_, row) in enumerate(results.iterrows()):
        ax.text(
            i,
            row["mean_l1_distance"],
            f"L0={row['mean_l0_distance']:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def make_runtime_figure(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Plot counterfactual generation runtime by model."""
    _require_columns(results, ["model", "counterfactual_generation_runtime_seconds"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [_model_label(model) for model in results["model"]]
    x = range(len(results))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, results["counterfactual_generation_runtime_seconds"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Runtime, seconds")
    ax.set_title("COMPAS DiCE counterfactual generation runtime")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def generate_compas_counterfactual_outputs(
    input_path: str | Path = DEFAULT_INPUT,
    table_dir: str | Path = DEFAULT_TABLE_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> dict[str, Path]:
    """Generate paper-ready counterfactual table and figures."""
    results = _read_counterfactual_results(input_path)

    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    table = make_counterfactual_table(results)
    table_path = table_dir / "compas_counterfactuals.csv"
    table.to_csv(table_path, index=False)

    paths = {
        "compas_counterfactuals_table": table_path,
        "compas_counterfactual_validity": make_validity_figure(
            results,
            figure_dir / "compas_counterfactual_validity.png",
        ),
        "compas_counterfactual_distance": make_distance_figure(
            results,
            figure_dir / "compas_counterfactual_distance.png",
        ),
        "compas_counterfactual_runtime": make_runtime_figure(
            results,
            figure_dir / "compas_counterfactual_runtime.png",
        ),
    }
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready COMPAS counterfactual outputs."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to compas_counterfactual_benchmark.csv.",
    )
    parser.add_argument(
        "--table-dir",
        default=DEFAULT_TABLE_DIR,
        help="Directory where counterfactual tables will be written.",
    )
    parser.add_argument(
        "--figure-dir",
        default=DEFAULT_FIGURE_DIR,
        help="Directory where counterfactual figures will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_compas_counterfactual_outputs(
        input_path=args.input,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
    )
    for name, path in paths.items():
        print(f"Wrote {name}: {path.resolve()}")


if __name__ == "__main__":
    main()
