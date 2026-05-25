# COMPAS Counterfactual Benchmark

This document describes the current COMPAS counterfactual benchmark implemented in ExplainBench.

## Scope

The current counterfactual benchmark uses DiCE on the cleaned COMPAS dataset.

Implemented files:

```text
explainbench/counterfactual_metrics.py
experiments/run_compas_counterfactual_benchmark.py
experiments/make_compas_counterfactual_outputs.py
results/compas_counterfactual_benchmark.csv
```

Paper-ready outputs:

```text
paper/tables/compas_counterfactuals.csv
paper/figures/compas_counterfactual_validity.png
paper/figures/compas_counterfactual_distance.png
paper/figures/compas_counterfactual_runtime.png
```

## Explanation type

DiCE produces counterfactual examples rather than local feature-attribution vectors.

For that reason, DiCE is evaluated separately from SHAP, LIME, occlusion, and linear-coefficient attribution explainers.

## COMPAS constraints

For the cleaned COMPAS data, the benchmark prevents counterfactuals from changing:

- protected attributes
- binary indicator columns

Current protected attributes:

```text
African_American
Female
```

With these constraints, the current actionable feature is:

```text
Number_of_Priors
```

This conservative setting avoids protected-attribute edits and invalid one-hot edits.

## Metrics

The counterfactual benchmark reports:

- requested counterfactual count
- valid counterfactual count
- failed counterfactual count
- validity rate
- mean L0 distance
- mean L1 distance
- mean changed features
- protected-attribute change rate
- counterfactual generation runtime
- mean runtime per query

## Reproduce benchmark results

Run:

```bash
python experiments/run_compas_counterfactual_benchmark.py \
  --output results/compas_counterfactual_benchmark.csv \
  --n-counterfactual 50 \
  --total-cfs 1
```

## Generate paper-ready outputs

Run:

```bash
python experiments/make_compas_counterfactual_outputs.py \
  --input results/compas_counterfactual_benchmark.csv \
  --table-dir paper/tables \
  --figure-dir paper/figures
```

## Current interpretation

The committed COMPAS counterfactual benchmark uses one random state and 50 query instances.

The current result should be interpreted as a COMPAS counterfactual module, not as a multi-dataset counterfactual benchmark.

Do not compare DiCE directly against SHAP or LIME using attribution-only metrics such as deletion fidelity or attribution stability. Counterfactual explanations should be reported with counterfactual-specific metrics.
