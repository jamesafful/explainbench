"""Dataset loading utilities for ExplainBench.

The current public artifact ships with COMPAS. Adult Income and LendingClub are
kept as planned datasets until their cleaned CSVs are added to ``datasets/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata needed to run a benchmark dataset."""

    name: str
    filename: str
    target: str
    protected_attributes: tuple[str, ...]
    positive_label: int = 1


DATASETS: dict[str, DatasetSpec] = {
    "compas": DatasetSpec(
        name="compas",
        filename="compas_clean.csv",
        target="Two_yr_Recidivism",
        protected_attributes=("African_American", "Female"),
        positive_label=1,
    ),
}


def available_datasets(data_dir: Optional[str | Path] = None) -> list[str]:
    """Return datasets that have both metadata and a CSV file present."""
    root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "datasets"
    return sorted(name for name, spec in DATASETS.items() if (root / spec.filename).exists())


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return metadata for a named dataset."""
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available metadata: {sorted(DATASETS)}")
    return DATASETS[key]


def load_dataset(name: str = "compas", data_dir: Optional[str | Path] = None) -> tuple[pd.DataFrame, pd.Series, DatasetSpec]:
    """Load a benchmark dataset as ``X, y, spec``.

    Parameters
    ----------
    name:
        Dataset key. Currently ``compas`` is the only CSV included in the repo.
    data_dir:
        Optional directory containing dataset CSV files. Defaults to the repo's
        top-level ``datasets/`` folder when running from source.
    """
    spec = get_dataset_spec(name)
    root = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "datasets"
    path = root / spec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset file at {path}. Add the CSV or pass data_dir explicitly."
        )
    df = pd.read_csv(path)
    if spec.target not in df.columns:
        raise ValueError(f"Target column '{spec.target}' not found in {path}. Columns: {list(df.columns)}")
    X = df.drop(columns=[spec.target])
    y = df[spec.target].astype(int)
    return X, y, spec
