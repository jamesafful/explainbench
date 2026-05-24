"""Run the COMPAS benchmark across multiple random seeds.

This script produces two files:

1. A seed-level benchmark CSV with one row per seed/model/explainer.
2. A summary CSV with mean/std metrics grouped by dataset/model/explainer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainbench.benchmark import BenchmarkConfig, run_benchmark


DEFAULT_SUMMARY_METRICS = [
    "accuracy",
    "f1",
    "roc_auc",
    "mean_sparsity",
    "deletion_fidelity_top3",
    "explanation_runtime_seconds",
    "mean_runtime_per_instance_seconds",
    "stability_top3_jaccard_mean",
    "stability_top3_jaccard_std",
    "stability_cosine_mean",
    "stability_cosine_std",
    "stability_runtime_seconds",
    "African_American_demographic_parity_difference",
    "African_American_disparate_impact_ratio",
    "African_American_equal_opportunity_difference",
    "African_American_false_positive_rate_difference",
    "African_American_equalized_odds_difference_absmax",
    "African_American_mean_abs_attribution_gap",
    "Female_demographic_parity_difference",
    "Female_disparate_impact_ratio",
    "Female_equal_opportunity_difference",
    "Female_false_positive_rate_difference",
    "Female_equalized_odds_difference_absmax",
    "Female_mean_abs_attribution_gap",
]


def summarize_multiseed(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize seed-level benchmark results by dataset/model/explainer.

    Parameters
    ----------
    df:
        Seed-level benchmark dataframe returned by repeated calls to run_benchmark.
    metrics:
        Numeric metric columns to summarize. If omitted, uses available columns
        from DEFAULT_SUMMARY_METRICS.

    Returns
    -------
    pd.DataFrame
        One row per dataset/model/explainer, with metric_mean and metric_std
        columns plus n_seeds.
    """
    group_cols = ["dataset", "model", "explainer"]

    missing_group_cols = [col for col in group_cols if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Missing required grouping columns: {missing_group_cols}")

    if "random_state" not in df.columns:
        raise ValueError("Input dataframe must include a random_state column.")

    if metrics is None:
        metrics = [metric for metric in DEFAULT_SUMMARY_METRICS if metric in df.columns]
    else:
        metrics = [metric for metric in metrics if metric in df.columns]

    if not metrics:
        raise ValueError("No requested summary metrics are present in the dataframe.")

    grouped = df.groupby(group_cols, dropna=False)

    mean_df = grouped[metrics].mean().reset_index()
    mean_df = mean_df.rename(columns={metric: f"{metric}_mean" for metric in metrics})

    std_df = grouped[metrics].std(ddof=1).reset_index()
    std_df = std_df.rename(columns={metric: f"{metric}_std" for metric in metrics})

    n_seed_df = grouped["random_state"].nunique().reset_index(name="n_seeds")

    summary = mean_df.merge(std_df, on=group_cols).merge(n_seed_df, on=group_cols)

    # Keep identifying columns first, then n_seeds, then metrics.
    ordered_cols = group_cols + ["n_seeds"] + [
        col for col in summary.columns if col not in group_cols + ["n_seeds"]
    ]
    return summary[ordered_cols].sort_values(group_cols).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COMPAS benchmark across multiple random seeds.")
    parser.add_argument(
        "--output",
        default="results/compas_multiseed_benchmark.csv",
        help="Path to write seed-level benchmark results.",
    )
    parser.add_argument(
        "--summary-output",
        default="results/compas_multiseed_summary.csv",
        help="Path to write grouped mean/std summary results.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Random seeds used for train/test split and model training.",
    )
    parser.add_argument(
        "--n-explain",
        type=int,
        default=200,
        help="Number of held-out test instances to explain per seed.",
    )
    parser.add_argument(
        "--n-stability",
        type=int,
        default=50,
        help="Number of explained instances used for stability diagnostics per seed.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k used for deletion fidelity and top-k stability.",
    )
    parser.add_argument(
        "--lime-num-samples",
        type=int,
        default=5000,
        help="Number of perturbed samples used by LIME per explained instance.",
    )
    parser.add_argument(
        "--stability-noise-scale",
        type=float,
        default=0.01,
        help="Noise scale for continuous-feature perturbations.",
    )
    parser.add_argument(
        "--stability-random-state",
        type=int,
        default=123,
        help="Base random seed for stability perturbations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = []
    for seed in args.seeds:
        config = BenchmarkConfig(
            dataset="compas",
            random_state=seed,
            n_explain=args.n_explain,
            n_stability=args.n_stability,
            top_k=args.top_k,
            lime_num_samples=args.lime_num_samples,
            stability_noise_scale=args.stability_noise_scale,
            stability_random_state=args.stability_random_state + seed,
        )
        df_seed = run_benchmark(config)
        df_seed["benchmark_seed_index"] = seed
        rows.append(df_seed)
        print(f"Completed seed {seed}: {len(df_seed)} rows")

    benchmark_df = pd.concat(rows, ignore_index=True)
    summary_df = summarize_multiseed(benchmark_df)

    output = Path(args.output)
    summary_output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    benchmark_df.to_csv(output, index=False)
    summary_df.to_csv(summary_output, index=False)

    print(f"Wrote seed-level benchmark results to {output.resolve()}")
    print(f"Wrote summary results to {summary_output.resolve()}")


if __name__ == "__main__":
    main()
