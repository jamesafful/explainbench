# Artifact Readiness Checklist

This checklist tracks the current ExplainBench artifact against common software-artifact review expectations.

## Availability

- [x] Source code is available in a public GitHub repository.
- [x] Repository includes an open-source license file.
- [x] Repository includes citation metadata through `CITATION.cff`.
- [x] Repository includes Zenodo release metadata through `.zenodo.json`.
- [ ] A versioned Zenodo DOI has been created.
- [ ] The paper references the final archived DOI.

## Installation

- [x] Environment file is provided through `environment.yml`.
- [x] Python package can be installed in editable mode.
- [x] Generated build artifacts are not tracked in source control.

## Functionality

- [x] COMPAS attribution benchmark can be executed.
- [x] COMPAS multi-seed benchmark can be executed.
- [x] COMPAS DiCE counterfactual benchmark can be executed.
- [x] Paper-ready attribution tables can be generated.
- [x] Paper-ready attribution figures can be generated.
- [x] Paper-ready counterfactual table and figures can be generated.

## Automated validation

- [x] Unit tests exist.
- [x] Documentation-existence tests exist.
- [x] GitHub Actions runs tests on pull requests.
- [x] GitHub Actions runs tests on pushes to `main`.

## Documentation

- [x] Benchmark protocol is documented.
- [x] Metrics are documented.
- [x] Dataset scope is documented.
- [x] Reproducibility procedure is documented.
- [x] Counterfactual benchmark is documented.
- [x] Artifact reproduction guide is documented.

## Result reproducibility

- [x] Result CSVs are committed for the current COMPAS benchmark.
- [x] Paper-ready tables are generated from result CSVs.
- [x] Paper-ready figures are generated from result CSVs.
- [x] Attribution and counterfactual explanation families are reported separately.
- [ ] Full clean-machine reproduction has been tested outside the development machine.
- [ ] Runtime environment has been archived with the final release.

## Scope control

- [x] Current reproducible scope is COMPAS.
- [x] Adult Income is not claimed as a completed reproducible benchmark.
- [x] LendingClub is not claimed as a completed reproducible benchmark.
- [ ] Adult Income benchmark has been implemented.
- [ ] LendingClub benchmark has been implemented.

## Final pre-submission tasks

- [ ] Update README with final paper title and artifact scope.
- [ ] Create a tagged GitHub release.
- [ ] Archive the release on Zenodo.
- [ ] Add DOI to README, paper, and `CITATION.cff`.
- [ ] Re-run all tests from a clean clone.
- [ ] Re-run all result-generation scripts from a clean clone.
