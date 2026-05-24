"""Run the COMPAS ExplainBench benchmark and save a CSV file."""
from __future__ import annotations

import argparse
from pathlib import Path

from explainbench.benchmark import BenchmarkConfig, save_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the COMPAS ExplainBench benchmark.")
    parser.add_argument(
        "--output",
        default="results/compas_initial_benchmark.csv",
        help="Path to write the benchmark CSV.",
    )
    parser.add_argument(
        "--n-explain",
        type=int,
        default=200,
        help="Number of held-out test instances to explain.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for train/test split and model training.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top attribution features used for deletion fidelity.",
    )
    parser.add_argument(
        "--lime-num-samples",
        type=int,
        default=5000,
        help="Number of perturbed samples used by LIME per explained instance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        dataset="compas",
        random_state=args.random_state,
        n_explain=args.n_explain,
        top_k=args.top_k,
        lime_num_samples=args.lime_num_samples,
    )
    output = save_benchmark(Path(args.output), config=config)
    print(f"Wrote benchmark results to {output.resolve()}")


if __name__ == "__main__":
    main()
