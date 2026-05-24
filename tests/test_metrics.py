import numpy as np
import pandas as pd

from explainbench.metrics import cosine_similarity, jaccard_top_k, sparsity, top_k_features


def test_sparsity_threshold():
    assert sparsity([0.0, 1e-9, 0.2, -0.3], threshold=1e-8) == 2


def test_top_k_features_uses_absolute_values():
    assert top_k_features([0.1, -2.0, 0.5], ["a", "b", "c"], k=2) == ["b", "c"]


def test_jaccard_top_k():
    assert jaccard_top_k([3, 2, 1], [3, 1, 2], k=2) == 1 / 3


def test_cosine_similarity():
    assert np.isclose(cosine_similarity([1, 0], [1, 0]), 1.0)


def test_attribution_stability_identical_explanations_is_perfect():
    from explainbench.metrics import attribution_stability

    values = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 2.0, 0.1],
        ]
    )
    result = attribution_stability(values, values.copy(), k=2)

    assert result["stability_top2_jaccard_mean"] == 1.0
    assert result["stability_top2_jaccard_std"] == 0.0
    assert np.isclose(result["stability_cosine_mean"], 1.0)
    assert np.isclose(result["stability_cosine_std"], 0.0)


def test_attribution_stability_requires_matching_shapes():
    from explainbench.metrics import attribution_stability

    original = np.zeros((2, 3))
    perturbed = np.zeros((2, 4))

    try:
        attribution_stability(original, perturbed, k=2)
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched explanation shapes")
