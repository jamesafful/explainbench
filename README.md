# ExplainBench

**ExplainBench** is an open-source toolkit for reproducible evaluation of local explanation methods in fairness-sensitive tabular machine-learning workflows.

The current artifact is an early benchmark implementation. It includes a cleaned COMPAS dataset, SHAP/LIME wrappers, a Streamlit demo, and the first reproducible benchmark runner for model performance, group fairness, and local explanation diagnostics.

## Current status

Implemented now:

- Cleaned COMPAS dataset loader.
- SHAP wrapper.
- LIME wrapper.
- Optional DiCE counterfactual wrapper.
- Streamlit demo for local explanations.
- Reproducible COMPAS benchmark script.
- Model metrics: accuracy, F1, ROC-AUC.
- Group fairness diagnostics: demographic parity difference, disparate impact ratio, equal opportunity difference, false-positive-rate difference, equalized-odds absolute-max difference.
- Explanation diagnostics: sparsity, deletion fidelity, top-k attribution utilities, cosine similarity, subgroup attribution gap.
- Initial tests for dataset loading and explanation metrics.

Planned but not yet shipped in this artifact:

- Cleaned Adult Income loader and dataset card.
- Cleaned LendingClub loader and dataset card.
- Full SHAP/LIME/DiCE benchmark table across datasets and models.
- Counterfactual recourse burden metrics.
- Stability parity and runtime-scalability experiments.
- Public leaderboard or benchmark submission schema.

## Installation from source

```bash
git clone https://github.com/jamesafful/explainbench.git
cd explainbench
python -m pip install -e .
```

## Run the initial COMPAS benchmark

```bash
python experiments/run_compas_benchmark.py --output results/compas_initial_benchmark.csv
```

The script trains logistic regression and random forest models, computes model performance and fairness diagnostics, and evaluates local explanation baselines using deletion fidelity, sparsity, and subgroup attribution gaps.

## Run tests

```bash
python -m pytest tests
```

## Repository layout

```text
explainbench/          Core Python package
  benchmark.py         Reproducible benchmark runner
  datasets.py          Dataset metadata and loaders
  explainers.py        Dependency-light explanation baselines
  metrics.py           Model, fairness, and explanation metrics
  shap_wrapper.py      SHAP wrapper
  lime_wrapper.py      LIME and optional DiCE wrappers

datasets/              Included cleaned datasets
experiments/           Scripts that generate paper-ready result files
notebooks/             Exploratory notebooks
streamlit_app/         Interactive explanation demo
tests/                 Unit tests
docs/                  Roadmap and benchmark documentation
```

## Research goal

The paper should eventually support this claim:

> ExplainBench provides a reproducible benchmark protocol and software artifact for comparing attribution and counterfactual explanation methods in fairness-critical tabular classification settings, exposing tradeoffs in fidelity, sparsity, stability, runtime, actionability, and subgroup explanation disparity.

## License

MIT License.
