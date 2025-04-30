import shap
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class SHAPExplainer:
    def __init__(self, model: BaseEstimator, data: pd.DataFrame, model_type: str = 'tree'):
        """
        Initialize SHAPExplainer
        
        :param model: Trained ML model
        :param data: Background dataset (used for explainer initialization)
        :param model_type: One of ['tree', 'linear', 'kernel']
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
        Generate SHAP values for a single instance
        
        :param instance: A row from the dataset
        :return: SHAP values explanation
        """
        return self.explainer(instance)

    def explain_global(self, num_samples: int = 100) -> shap.Explanation:
        """
        Generate SHAP values for a sample of the data

        :param num_samples: Number of samples to use
        :return: SHAP values explanation for global analysis
        """
        sample = self.data.sample(n=min(num_samples, len(self.data)), random_state=42)
        return self.explainer(sample)

    def plot_summary(self, shap_values: shap.Explanation, features: pd.DataFrame = None):
        """
        Plot SHAP summary (bar or beeswarm)

        :param shap_values: SHAP values
        :param features: Optional feature dataset (defaults to self.data)
        """
        shap.summary_plot(shap_values.values if hasattr(shap_values, 'values') else shap_values,
                          features or self.data)
