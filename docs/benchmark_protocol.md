# ExplainBench Benchmark Protocol

This document describes the current ExplainBench benchmark protocol implemented in this repository.

## Current benchmark scope

The current reproducible benchmark covers the COMPAS dataset only.

Implemented benchmark artifacts:

- `experiments/run_compas_benchmark.py`
- `experiments/run_compas_multiseed.py`
- `experiments/make_compas_tables.py`
- `experiments/make_compas_figures.py`

Generated outputs:

- `results/compas_initial_benchmark.csv`
- `results/compas_multiseed_benchmark.csv`
- `results/compas_multiseed_summary.csv`
- `paper/tables/*.csv`
- `paper/figures/*.png`

Adult Income and LendingClub are not yet part of the reproducible benchmark results in this repository.

## Dataset

The current benchmark uses:

- `datasets/compas_clean.csv`

The dataset loader returns:

- feature matrix `X`
- target vector `y`
- dataset specification metadata

The target column is:

- `Two_yr_Recidivism`

Protected attributes currently used for group diagnostics are:

- `African_American`
- `Female`

## Models

The current benchmark evaluates two sklearn models:

1. Logistic regression pipeline

   - `StandardScaler`
   - `LogisticRegression(max_iter=1000)`

2. Random forest

   - `RandomForestClassifier`
   - `n_estimators=200`
   - `min_samples_leaf=5`
   - `n_jobs=-1`

## Explainers

The current COMPAS benchmark evaluates:

### Logistic regression

- `occlusion`
- `lime`
- `linear_coefficients`
- `shap_linear`

### Random forest

- `occlusion`
- `lime`
- `shap_tree`

All benchmarked explainers return dense attribution matrices with shape:

```text
n_instances x n_features
```

Rows align with explained instances. Columns align with feature columns.

## Train/test split

The benchmark uses a stratified train/test split.

Default configuration:

- test size: `0.30`
- one-seed benchmark random state: `42`
- multi-seed benchmark random states: `0, 1, 2, 3, 4`

## Number of explained instances

Default one-seed and multi-seed configuration:

- `n_explain = 200`
- `n_stability = 50`

The stability subset is taken from the first `n_stability` explained instances.

## LIME configuration

Default LIME configuration:

- `lime_num_samples = 5000`
- `discretize_continuous = False`

LIME explanations are converted into dense attribution vectors aligned to the original feature columns.

## SHAP configuration

The benchmark uses:

- SHAP `LinearExplainer` for logistic regression in standardized feature space
- SHAP `TreeExplainer` for random forest in original feature space

SHAP outputs are normalized into dense 2D attribution arrays.

## Runtime measurement

Runtime is measured around explanation generation.

Reported runtime fields include:

- `explanation_runtime_seconds`
- `mean_runtime_per_instance_seconds`
- `stability_runtime_seconds`

These values are implementation- and hardware-dependent and should be interpreted as empirical runtime diagnostics for the current environment, not as universal constants.

## Stability protocol

The benchmark evaluates perturbation-based local explanation stability.

For COMPAS, perturbations avoid:

- protected attributes
- binary indicator columns

This means the current stability protocol primarily perturbs the non-protected count-like feature:

- `Number_of_Priors`

The perturbation is Gaussian noise scaled by the feature standard deviation:

```text
noise = Normal(0, stability_noise_scale * feature_std)
```

Default:

```text
stability_noise_scale = 0.01
```

Perturbed values are clipped to the training-set feature range.

This stability protocol is a local perturbation diagnostic. It should not be interpreted as a proof of global explanation robustness.

## Multi-seed summary

The multi-seed benchmark runs five random splits and summarizes results by:

- dataset
- model
- explainer

The summary reports mean and sample standard deviation across seeds.

Default seeds:

```text
0, 1, 2, 3, 4
```

## Paper tables and figures

Paper-ready tables are generated from:

```text
results/compas_multiseed_summary.csv
```

using:

```bash
python experiments/make_compas_tables.py \
  --input results/compas_multiseed_summary.csv \
  --output-dir paper/tables
```

Paper-ready figures are generated from:

```text
results/compas_multiseed_summary.csv
```

using:

```bash
python experiments/make_compas_figures.py \
  --input results/compas_multiseed_summary.csv \
  --output-dir paper/figures
```

## Current limitations

The current benchmark is a strong COMPAS module, but it is not yet a complete multi-dataset benchmark.

Current limitations:

- Only COMPAS has reproducible benchmark outputs.
- Adult Income and LendingClub are not yet included in the reproducible benchmark pipeline.
- DiCE counterfactual benchmarking is not yet implemented in the benchmark results.
- Runtime values are hardware-dependent.
- Stability currently perturbs only non-protected, non-binary features.
- The current perturbation protocol is local and should not be overinterpreted as global robustness.
