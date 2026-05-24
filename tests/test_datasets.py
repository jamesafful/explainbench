from explainbench.datasets import available_datasets, load_dataset


def test_compas_loads():
    X, y, spec = load_dataset("compas")
    assert spec.target == "Two_yr_Recidivism"
    assert len(X) == len(y)
    assert "African_American" in X.columns
    assert "Female" in X.columns
    assert "compas" in available_datasets()
