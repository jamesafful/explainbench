# ExplainBench Artifact Guide

This guide is intended for reviewers, users, and future maintainers who want to reproduce the current ExplainBench artifact.

## Current artifact scope

The current reproducible artifact focuses on the COMPAS benchmark.

Implemented and reproducible:

- COMPAS attribution benchmark
- COMPAS multi-seed benchmark summaries
- COMPAS paper-ready attribution tables
- COMPAS paper-ready attribution figures
- COMPAS DiCE counterfactual benchmark
- COMPAS paper-ready counterfactual table
- COMPAS paper-ready counterfactual figures
- benchmark protocol documentation
- metric documentation
- dataset documentation
- reproducibility documentation
- counterfactual benchmark documentation
- GitHub Actions test workflow

Not yet implemented as reproducible benchmark outputs:

- Adult Income benchmark
- LendingClub benchmark

These should not be claimed as completed empirical benchmarks until scripts, results, tables, figures, and documentation are added.

## Repository layout

```text
datasets/       Cleaned benchmark data currently used by the repository
docs/           Benchmark, metric, dataset, and reproducibility documentation
experiments/    Executable benchmark and output-generation scripts
explainbench/   Python package code
paper/          Paper-ready tables and figures generated from result CSVs
results/        Committed benchmark outputs
tests/          Automated tests
```

## Installation

Recommended environment:

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

## Validate the artifact

Run:

```bash
python -m pytest -q
```

The expected current result is:

```text
26 passed
```

Warning counts may vary across environments.

## Reproduce COMPAS attribution benchmark outputs

Run the one-seed attribution benchmark:

```bash
python experiments/run_compas_benchmark.py \
  --output results/compas_initial_benchmark.csv \
  --n-explain 200 \
  --n-stability 50 \
  --lime-num-samples 5000
```

Run the multi-seed attribution benchmark:

```bash
python experiments/run_compas_multiseed.py \
  --output results/compas_multiseed_benchmark.csv \
  --summary-output results/compas_multiseed_summary.csv \
  --seeds 0 1 2 3 4 \
  --n-explain 200 \
  --n-stability 50 \
  --lime-num-samples 5000
```

Generate attribution paper tables:

```bash
python experiments/make_compas_tables.py \
  --input results/compas_multiseed_summary.csv \
  --output-dir paper/tables
```

Generate attribution paper figures:

```bash
python experiments/make_compas_figures.py \
  --input results/compas_multiseed_summary.csv \
  --output-dir paper/figures
```

## Reproduce COMPAS counterfactual benchmark outputs

Run the DiCE counterfactual benchmark:

```bash
python experiments/run_compas_counterfactual_benchmark.py \
  --output results/compas_counterfactual_benchmark.csv \
  --n-counterfactual 50 \
  --total-cfs 1
```

Generate counterfactual paper outputs:

```bash
python experiments/make_compas_counterfactual_outputs.py \
  --input results/compas_counterfactual_benchmark.csv \
  --table-dir paper/tables \
  --figure-dir paper/figures
```

## Expected output files

Attribution results:

```text
results/compas_initial_benchmark.csv
results/compas_multiseed_benchmark.csv
results/compas_multiseed_summary.csv
paper/tables/compas_model_performance.csv
paper/tables/compas_explanation_quality.csv
paper/tables/compas_runtime.csv
paper/tables/compas_stability.csv
paper/tables/compas_fairness.csv
paper/figures/compas_runtime_vs_fidelity.png
paper/figures/compas_stability_by_explainer.png
paper/figures/compas_attribution_gap.png
paper/figures/compas_model_performance.png
```

Counterfactual results:

```text
results/compas_counterfactual_benchmark.csv
paper/tables/compas_counterfactuals.csv
paper/figures/compas_counterfactual_validity.png
paper/figures/compas_counterfactual_distance.png
paper/figures/compas_counterfactual_runtime.png
```

## Artifact claims

The artifact currently supports the following claims:

1. ExplainBench can run a reproducible COMPAS attribution benchmark.
2. ExplainBench can evaluate attribution explainers using model performance, sparsity, deletion fidelity, perturbation stability, protected-group attribution gaps, and runtime.
3. ExplainBench can run a separate COMPAS DiCE counterfactual benchmark.
4. ExplainBench can evaluate counterfactuals using validity, feature-change distance, protected-attribute change rate, and runtime.
5. ExplainBench can generate paper-ready COMPAS tables and figures from committed benchmark CSVs.
6. ExplainBench has automated tests and a GitHub Actions workflow.

The artifact does not yet support completed empirical claims about Adult Income or LendingClub.

## Notes for reviewers

Counterfactual results are intentionally reported separately from attribution results.

SHAP, LIME, occlusion, and linear coefficients produce attribution vectors. DiCE produces counterfactual feature profiles. These explanation families are therefore evaluated with different metrics.
