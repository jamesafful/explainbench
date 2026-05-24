# ExplainBench Datasets

This document describes datasets currently used in the reproducible benchmark pipeline.

## Current reproducible dataset

The current reproducible benchmark uses:

```text
datasets/compas_clean.csv
```

## COMPAS

### Target

The target column is:

```text
Two_yr_Recidivism
```

The benchmark treats this as a binary classification target.

### Feature columns

The current cleaned dataset contains:

```text
Number_of_Priors
score_factor
Age_Above_FourtyFive
Age_Below_TwentyFive
African_American
Asian
Hispanic
Native_American
Other
Female
Misdemeanor
```

The dataset file also contains the target:

```text
Two_yr_Recidivism
```

### Protected attributes

The current benchmark computes group diagnostics for:

```text
African_American
Female
```

The benchmark uses the convention:

```text
0 = privileged/reference group
1 = unprivileged/comparison group
```

This convention is used for metric computation and should be stated explicitly when reporting results.

### Stability perturbation handling

For stability diagnostics, the benchmark avoids perturbing:

- protected attributes
- binary indicator columns

In the current cleaned COMPAS dataset, the perturbation primarily affects:

```text
Number_of_Priors
```

The one-hot and binary indicator columns are held fixed.

### Current limitations

The repository currently includes a cleaned COMPAS CSV, but the full preprocessing provenance should be documented more completely before journal submission.

Needed before final submission:

- source citation for the raw COMPAS data
- preprocessing script or exact preprocessing description
- column inclusion/exclusion rationale
- target definition rationale
- protected-attribute rationale
- data license or redistribution note

## Adult Income

Adult Income is mentioned in earlier project materials, but it is not yet part of the current reproducible benchmark outputs.

Before claiming Adult benchmark support, the repository should include one of the following:

1. a committed cleaned dataset with clear redistribution permission, or
2. a reproducible preparation script that downloads/preprocesses the data.

Needed files before claiming support:

```text
datasets/scripts/prepare_adult.py
experiments/run_adult_benchmark.py
results/adult_initial_benchmark.csv
```

## LendingClub

LendingClub is mentioned in earlier project materials, but it is not yet part of the current reproducible benchmark outputs.

Before claiming LendingClub benchmark support, the repository must clarify:

- source data location
- redistribution restrictions
- preprocessing procedure
- target definition
- feature inclusion/exclusion
- fairness/proxy variables
- train/test split protocol

Needed files before claiming support:

```text
datasets/scripts/prepare_lendingclub.py
experiments/run_lendingclub_benchmark.py
results/lendingclub_initial_benchmark.csv
```
