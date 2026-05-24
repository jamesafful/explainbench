"""Run the initial COMPAS benchmark used to harden ExplainBench.

Usage:
    python experiments/run_compas_benchmark.py --output results/compas_initial_benchmark.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explainbench.benchmark import BenchmarkConfig, save_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/compas_initial_benchmark.csv")
    parser.add_argument("--n-explain", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    out = save_benchmark(
        args.output,
        BenchmarkConfig(
            dataset="compas",
            n_explain=args.n_explain,
            random_state=args.seed,
            top_k=args.top_k,
        ),
    )
    print(f"Wrote benchmark results to {Path(out).resolve()}")


if __name__ == "__main__":
    main()
