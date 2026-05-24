# ExplainBench Metrics

This document defines the metrics currently used in the COMPAS benchmark.

## Model performance metrics

### Accuracy

Fraction of test examples correctly classified.

### F1 score

Binary F1 score computed from predicted class labels.

### ROC-AUC

Area under the ROC curve computed from positive-class scores.

If ROC-AUC cannot be computed, the benchmark records `NaN`.

## Group fairness diagnostics

The benchmark computes group diagnostics for binary protected attributes.

Current protected attributes:

- `African_American`
- `Female`

The default convention is:

```text
privileged value = 0
unprivileged value = 1
```

### Selection rate

Fraction of examples predicted as positive within a group.

### Demographic parity difference

```text
selection_rate_unprivileged - selection_rate_privileged
```

### Disparate impact ratio

```text
selection_rate_unprivileged / selection_rate_privileged
```

A small numerical epsilon is added to the denominator for stability.

### Equal opportunity difference

```text
true_positive_rate_unprivileged - true_positive_rate_privileged
```

### False-positive-rate difference

```text
false_positive_rate_unprivileged - false_positive_rate_privileged
```

### Equalized-odds difference, absolute max

```text
max(
  abs(true_positive_rate_unprivileged - true_positive_rate_privileged),
  abs(false_positive_rate_unprivileged - false_positive_rate_privileged)
)
```

## Explanation metrics

### Mean sparsity

For each attribution vector, sparsity is the number of non-negligible attribution values.

The benchmark uses a threshold of:

```text
1e-8
```

Mean sparsity is averaged across explained instances.

### Deletion fidelity, top-k

Deletion fidelity measures the mean change in positive-class score after replacing the top-k absolute-attribution features with background values.

Default:

```text
top_k = 3
```

The background value is the training-set median for each feature.

Larger positive values indicate that removing the top-ranked features lowers the model's positive-class score more strongly.

Important limitation:

Deletion fidelity may be negative when the top absolute-attribution features were pushing the positive-class score downward. This is not automatically a code error.

### Protected-group mean absolute attribution gap

For each protected attribute, the benchmark computes:

```text
mean_abs_attribution_unprivileged - mean_abs_attribution_privileged
```

Positive values indicate larger average absolute attribution magnitude for the unprivileged group.

Current protected attributes:

- `African_American`
- `Female`

## Stability metrics

The benchmark evaluates local perturbation stability by comparing explanations for original and perturbed instances.

### Top-k Jaccard stability

For each instance, compute the top-k absolute-attribution feature set for the original explanation and the perturbed explanation.

Then compute:

```text
|top_k_original intersection top_k_perturbed| / |top_k_original union top_k_perturbed|
```

Default:

```text
k = 3
```

The benchmark reports:

- `stability_top3_jaccard_mean`
- `stability_top3_jaccard_std`

### Cosine stability

For each instance, compute cosine similarity between the original attribution vector and the perturbed attribution vector.

The benchmark reports:

- `stability_cosine_mean`
- `stability_cosine_std`

Cosine stability values near 1 indicate attribution vectors with similar direction.

## Runtime metrics

### Explanation runtime

Wall-clock time used to generate explanations for `n_explain` instances.

Reported as:

- `explanation_runtime_seconds`

### Mean runtime per instance

```text
explanation_runtime_seconds / n_explain
```

Reported as:

- `mean_runtime_per_instance_seconds`

### Stability runtime

Wall-clock time used to generate explanations for the perturbed stability subset.

Reported as:

- `stability_runtime_seconds`

## Multi-seed summary metrics

The multi-seed summary reports metric means and sample standard deviations across seeds.

For a metric named:

```text
metric_name
```

the summary includes:

```text
metric_name_mean
metric_name_std
```

The number of seeds is recorded as:

```text
n_seeds
```
