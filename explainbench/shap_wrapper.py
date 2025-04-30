import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from explainbench.shap_wrapper import SHAPExplainer
from explainbench.lime_wrapper import LIMEExplainer

st.set_page_config(page_title="ExplainBench Demo", layout="wide")
st.title("ExplainBench: Interpretable Machine Learning Toolkit")

# Load data
df = pd.read_csv('../datasets/compas_clean.csv')
target = 'two_year_recid'
categorical = ['sex', 'race', 'age_cat', 'c_charge_degree']

for col in categorical:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=[target])
y = df[target]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Initialize explainers
shap_explainer = SHAPExplainer(model, X, model_type='tree')
lime_explainer = LIMEExplainer(model, X)

# Sidebar instance selector
st.sidebar.header("Select Data Instance")
index = st.sidebar.slider("Row Index", min_value=0, max_value=len(X)-1, value=0)
instance = X.iloc[index]

# SHAP explanation
st.subheader("SHAP Explanation")
shap_value = shap_explainer.explain_instance(instance)
fig1 = shap.plots._waterfall.waterfall_legacy(shap_value[0], show=False)
st.pyplot(fig1)

# LIME explanation
st.subheader("LIME Explanation")
lime_exp = lime_explainer.explain_instance(instance)
lime_html = lime_exp.as_html()
st.components.v1.html(lime_html, height=400, scrolling=True)

st.markdown("---")
st.markdown("Powered by ExplainBench | GitHub: [YourRepoLink]")
