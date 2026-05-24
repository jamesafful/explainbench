"""Run the COMPAS ExplainBench benchmark and save a CSV file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        "--n-stability",
        type=int,
        default=50,
        help="Number of explained instances used for perturbation-based stability diagnostics.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for train/test split and model training.",
    )
    parser.add_argument(
        "--stability-random-state",
        type=int,
        default=123,
        help="Random seed used for perturbation-based stability diagnostics.",
    )
    parser.add_argument(
        "--stability-noise-scale",
        type=float,
        default=0.01,
        help="Noise scale for continuous-feature perturbations, expressed as a fraction of feature standard deviation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top attribution features used for deletion fidelity and top-k stability.",
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
        n_stability=args.n_stability,
        top_k=args.top_k,
        lime_num_samples=args.lime_num_samples,
        stability_noise_scale=args.stability_noise_scale,
        stability_random_state=args.stability_random_state,
    )
    output = save_benchmark(Path(args.output), config=config)
    print(f"Wrote benchmark results to {output.resolve()}")


if __name__ == "__main__":
    main()
