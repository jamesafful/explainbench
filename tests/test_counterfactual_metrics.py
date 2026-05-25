import numpy as np
import pandas as pd
import pytest

from explainbench.counterfactual_metrics import (
    mean_l0_distance,
    mean_l1_distance,
    protected_attribute_change_rate,
    summarize_counterfactuals,
    validity_rate,
)


def test_validity_rate():
    assert validity_rate(10, 7) == 0.7
    assert np.isnan(validity_rate(0, 0))

    with pytest.raises(ValueError):
        validity_rate(5, 6)

    with pytest.raises(ValueError):
        validity_rate(-1, 0)


def test_counterfactual_distance_metrics():
    original = pd.DataFrame(
        {
            "x1": [1.0, 2.0],
            "x2": [0.0, 1.0],
            "protected": [0, 1],
        }
    )
    counterfactuals = pd.DataFrame(
        {
            "x1": [2.0, 2.0],
            "x2": [0.0, 0.0],
            "protected": [0, 1],
        }
    )

    features = ["x1", "x2", "protected"]

    assert mean_l0_distance(original, counterfactuals, features) == 1.0
    assert mean_l1_distance(original, counterfactuals, features) == 1.0
    assert protected_attribute_change_rate(original, counterfactuals, ["protected"]) == 0.0


def test_protected_attribute_change_rate_detects_changes():
    original = pd.DataFrame({"protected": [0, 0, 1]})
    counterfactuals = pd.DataFrame({"protected": [0, 1, 1]})

    assert protected_attribute_change_rate(original, counterfactuals, ["protected"]) == pytest.approx(
        1 / 3
    )


def test_summarize_counterfactuals():
    original = pd.DataFrame(
        {
            "x1": [1.0, 2.0],
            "x2": [0.0, 1.0],
            "protected": [0, 1],
        }
    )
    counterfactuals = pd.DataFrame(
        {
            "x1": [2.0, 2.0],
            "x2": [0.0, 0.0],
            "protected": [0, 1],
        }
    )

    summary = summarize_counterfactuals(
        original=original,
        counterfactuals=counterfactuals,
        feature_columns=["x1", "x2", "protected"],
        protected_attributes=["protected"],
        requested_count=4,
    )

    assert summary["requested_count"] == 4.0
    assert summary["valid_counterfactual_count"] == 2.0
    assert summary["validity_rate"] == 0.5
    assert summary["mean_l0_distance"] == 1.0
    assert summary["mean_l1_distance"] == 1.0
    assert summary["protected_attribute_change_rate"] == 0.0


def test_metric_shape_mismatch_raises():
    original = pd.DataFrame({"x": [1, 2]})
    counterfactuals = pd.DataFrame({"x": [1]})

    with pytest.raises(ValueError):
        mean_l0_distance(original, counterfactuals, ["x"])
