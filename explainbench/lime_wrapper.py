import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class LIMEExplainer:
    def __init__(self, model: BaseEstimator, data: pd.DataFrame):
        """
        Initialize LIMEExplainer

        :param model: Trained ML model
        :param data: Background dataset
        """
        self.model = model
        self.data = data
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=data.values,
            feature_names=data.columns.tolist(),
            class_names=['No', 'Yes'],
            mode='classification'
        )

    def explain_instance(self, instance: pd.Series, num_features: int = 10):
        """
        Generate LIME explanation for a single instance

        :param instance: Data row to explain
        :param num_features: Number of features to include in the explanation
        :return: LIME explanation object
        """
        return self.explainer.explain_instance(
            data_row=instance.values,
            predict_fn=self.model.predict_proba,
            num_features=num_features
        )

# Optional: DiCE for counterfactuals
try:
    import dice_ml
    from dice_ml.utils import helpers
except ImportError:
    dice_ml = None

class CounterfactualExplainer:
    def __init__(self, model: BaseEstimator, data: pd.DataFrame, target: pd.Series):
        """
        Initialize CounterfactualExplainer using DiCE

        :param model: Trained ML model
        :param data: Dataset used for DiCE setup
        :param target: Target variable
        """
        if dice_ml is None:
            raise ImportError("Please install dice-ml to use CounterfactualExplainer")

        data_copy = data.copy()
        data_copy['target'] = target.values

        d = dice_ml.Data(dataframe=data_copy, continuous_features=data.columns.tolist(), outcome_name='target')
        m = dice_ml.Model(model=model, backend="sklearn")
        self.explainer = dice_ml.Dice(d, m)

    def generate(self, instance: pd.Series, total_CFs: int = 3, desired_class: int = 1):
        """
        Generate counterfactuals

        :param instance: Data row to generate counterfactuals for
        :param total_CFs: Number of counterfactuals
        :param desired_class: Target class
        :return: Counterfactual examples
        """
        return self.explainer.generate_counterfactuals(instance.to_frame().T, total_CFs=total_CFs, desired_class=desired_class)
