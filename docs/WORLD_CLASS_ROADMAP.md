# ExplainBench world-class roadmap

This document turns the paper-review feedback into an implementation plan.

## Immediate truth-in-advertising fixes

- The current artifact ships with `datasets/compas_clean.csv` only. Adult Income and LendingClub should be described as planned datasets until their cleaned CSVs and loaders are committed.
- The current package supports SHAP and LIME wrappers, and has a DiCE wrapper class inside `lime_wrapper.py`; it does not yet provide a complete benchmark protocol.
- The paper should not claim complete quantitative benchmarking until result tables are generated and reproducible scripts are included.

## Minimum benchmark for the revised paper

Datasets:
- COMPAS now.
- Adult Income after adding a cleaned CSV and dataset card.
- LendingClub after adding a cleaned CSV and dataset card.

Models:
- Logistic regression.
- Random forest.
- Gradient-boosted trees.

Explanation methods:
- SHAP.
- LIME.
- DiCE/counterfactual recourse.
- Occlusion/permutation baseline.
- Linear coefficients for transparent linear models.
- Random baseline for sanity checking.

Metrics:
- Predictive performance: accuracy, F1, ROC-AUC.
- Group fairness: demographic parity difference, disparate impact ratio, equal opportunity difference, equalized-odds difference.
- Explanation quality: sparsity, deletion fidelity, top-k stability, cosine stability, runtime.
- Fairness of explanations: subgroup attribution gap, subgroup stability parity, counterfactual recourse burden disparity.
- Counterfactual quality: validity, proximity, sparsity, immutable-feature violation rate, actionability rate.

## Required paper figures

1. System architecture diagram.
2. One-instance comparison of SHAP, LIME, and counterfactual recourse.
3. Metric table across datasets, models, and explainers.
4. Runtime/scalability figure.
5. Subgroup explanation disparity figure.
6. Recourse burden by group figure.

## Artifact quality checklist

- Add `tests/` and GitHub Actions.
- Add `docs/` with dataset cards and metric definitions.
- Add `experiments/` scripts that regenerate every result table.
- Add `results/` CSV outputs used in the paper.
- Add a Dockerfile or environment file.
- Cut a GitHub release and archive it on Zenodo.
- Add `CITATION.cff`.
