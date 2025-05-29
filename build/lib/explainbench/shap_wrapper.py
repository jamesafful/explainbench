import shap
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class SHAPExplainer:
    def __init__(self, model: BaseEstimator, data: pd.DataFrame, model_type: str = 'tree'):
        """
        Initialize SHAPExplainer for model interpretability using SHAP.

        :param model: Trained machine learning model
        :param data: Dataset used for background distribution
        :param model_type: Model type, one of ['tree', 'linear', 'kernel']
        """
        self.model = model
        self.data = data
        self.model_type = model_type

        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model, data)
        elif model_type == 'linear':
            self.explainer = shap.LinearExplainer(model, data)
        else:
            self.explainer = shap.KernelExplainer(model.predict, data)

    def explain_instance(self, instance: pd.Series) -> shap.Explanation:
        """
        Explain a single prediction using SHAP values.

        :param instance: A single row of input data
        :return: SHAP explanation object
        """
        return self.explainer(instance)

    def explain_global(self, num_samples: int = 100) -> shap.Explanation:
        """
        Generate SHAP values for a sample of the dataset.

        :param num_samples: Number of samples to compute explanations on
        :return: SHAP explanation object for sampled data
        """
        sample = self.data.sample(n=min(num_samples, len(self.data)), random_state=42)
        return self.explainer(sample)

    def plot_summary(self, shap_values: shap.Explanation, features: pd.DataFrame = None):
        """
        Plot a SHAP summary plot (beeswarm).

        :param shap_values: SHAP values returned from explainer
        :param features: Feature values corresponding to SHAP values
        """
        if isinstance(features, pd.DataFrame):
            features_to_plot = features.reset_index(drop=True)
        else:
            features_to_plot = self.data.reset_index(drop=True)

        values = getattr(shap_values, 'values', shap_values)
        if isinstance(values, np.ndarray) and values.ndim == 3:
            print("Detected multiclass SHAP output; using class 1 SHAP values")
            values = values[:, :, 1]

        print("SHAP values shape:", values.shape)
        print("Features shape:", features_to_plot.shape)

        assert values.shape[0] == features_to_plot.shape[0], "SHAP rows must match feature rows"
        assert values.shape[1] == features_to_plot.shape[1], "SHAP columns must match feature columns"

        shap.summary_plot(values, features_to_plot)
