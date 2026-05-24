# Reproducibility Guide

This document describes how to reproduce the current COMPAS benchmark results, tables, and figures.

## Environment

The recommended environment is defined in:

```text
environment.yml
```

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate explainbench
python -m pip install -e . --no-deps
```

If the environment already exists:

```bash
conda activate explainbench
python -m pip install -e . --no-deps
```

## Run tests

```bash
python -m pytest -q
```

Expected current result:

```text
17 passed
```

Warning counts may vary across dependency versions.

## Run the one-seed COMPAS benchmark

```bash
python experiments/run_compas_benchmark.py \
  --output results/compas_initial_benchmark.csv \
  --n-explain 200 \
  --n-stability 50 \
  --lime-num-samples 5000
```

This generates:

```text
results/compas_initial_benchmark.csv
```

## Run the multi-seed COMPAS benchmark

```bash
python experiments/run_compas_multiseed.py \
  --output results/compas_multiseed_benchmark.csv \
  --summary-output results/compas_multiseed_summary.csv \
  --seeds 0 1 2 3 4 \
  --n-explain 200 \
  --n-stability 50 \
  --lime-num-samples 5000
```

This generates:

```text
results/compas_multiseed_benchmark.csv
results/compas_multiseed_summary.csv
```

Expected row counts:

```text
results/compas_multiseed_benchmark.csv: 35 rows
results/compas_multiseed_summary.csv: 7 rows
```

Reason:

```text
5 seeds x 7 model/explainer combinations = 35 seed-level rows
```

## Generate paper tables

```bash
python experiments/make_compas_tables.py \
  --input results/compas_multiseed_summary.csv \
  --output-dir paper/tables
```

This generates:

```text
paper/tables/compas_model_performance.csv
paper/tables/compas_explanation_quality.csv
paper/tables/compas_runtime.csv
paper/tables/compas_stability.csv
paper/tables/compas_fairness.csv
```

## Generate paper figures

```bash
python experiments/make_compas_figures.py \
  --input results/compas_multiseed_summary.csv \
  --output-dir paper/figures
```

This generates:

```text
paper/figures/compas_runtime_vs_fidelity.png
paper/figures/compas_stability_by_explainer.png
paper/figures/compas_attribution_gap.png
paper/figures/compas_model_performance.png
```

## Full reproduction sequence

From a clean checkout with the environment installed:

```bash
python -m pytest -q

python experiments/run_compas_benchmark.py \
  --output results/compas_initial_benchmark.csv \
  --n-explain 200 \
  --n-stability 50 \
  --lime-num-samples 5000

python experiments/run_compas_multiseed.py \
  --output results/compas_multiseed_benchmark.csv \
  --summary-output results/compas_multiseed_summary.csv \
  --seeds 0 1 2 3 4 \
  --n-explain 200 \
  --n-stability 50 \
  --lime-num-samples 5000

python experiments/make_compas_tables.py \
  --input results/compas_multiseed_summary.csv \
  --output-dir paper/tables

python experiments/make_compas_figures.py \
  --input results/compas_multiseed_summary.csv \
  --output-dir paper/figures
```

## Notes on runtime

LIME is the slowest explainer in the current benchmark, especially for random forests.

Runtime values are hardware-dependent. The committed runtime values should be treated as empirical diagnostics from the benchmark run, not universal constants.

## Notes on warnings

The current dependency stack may emit deprecation warnings from third-party packages such as SciPy, matplotlib, pyparsing, SHAP, or NumPy.

Warnings should be monitored, but the current benchmark validation criterion is that tests pass and result-generation scripts complete successfully.
