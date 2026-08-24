"""
EMIPredict AI - Model Performance page
Displays the model comparison results from MLflow-tracked training runs.
These numbers are hardcoded from the training notebook's actual output --
a live MLflow tracking server isn't reachable from Streamlit Cloud by default,
so this page presents the final tracked results rather than a live connection.
"""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Model Performance — EMIPredict AI", page_icon="\U0001F4C8", layout="wide")

st.title("\U0001F4C8 Model Performance")
st.markdown(
    "Results from the MLflow-tracked training runs (6 models: 3 classification, 3 regression). "
    "The best model of each type, shown highlighted below, is what powers the **Predict** page."
)

st.info(
    "These results come from the training notebook's MLflow experiment tracking. "
    "This page shows the final logged comparison rather than a live MLflow UI connection, "
    "since the MLflow tracking server used during training runs on Kaggle and isn't "
    "reachable from this deployed app."
)

st.divider()

st.subheader("Classification: EMI Eligibility")

clf_results = pd.DataFrame([
    {"Model": "Logistic Regression", "Accuracy": 0.8158, "Weighted F1": 0.8602, "Macro F1": 0.6684, "ROC-AUC": 0.9706},
    {"Model": "Random Forest", "Accuracy": 0.9196, "Weighted F1": 0.9335, "Macro F1": 0.7820, "ROC-AUC": 0.9928},
    {"Model": "XGBoost", "Accuracy": 0.9411, "Weighted F1": 0.9510, "Macro F1": 0.8364, "ROC-AUC": 0.9979},
])

def highlight_best(row):
    is_best = row['Model'] == 'XGBoost'
    return ['background-color: #1e4620' if is_best else '' for _ in row]

st.dataframe(
    clf_results.style.apply(highlight_best, axis=1).format(
        {c: "{:.4f}" for c in clf_results.columns if c != 'Model'}
    ),
    use_container_width=True, hide_index=True
)
st.success("**Selected model: XGBoost** — highest Macro F1 (0.836), critical for correctly identifying the minority High_Risk class (~4% of applicants).")

fig, ax = plt.subplots(figsize=(9, 4))
clf_results.set_index('Model')[['Accuracy', 'Weighted F1', 'Macro F1', 'ROC-AUC']].plot(kind='bar', ax=ax)
ax.set_ylim(0, 1)
plt.xticks(rotation=0)
st.pyplot(fig)

st.divider()

st.subheader("Regression: Maximum Monthly EMI")

reg_results = pd.DataFrame([
    {"Model": "Linear Regression", "RMSE": 4086.47, "MAE": 2942.08, "R²": 0.7171, "MAPE (%)": 193.53},
    {"Model": "Random Forest", "RMSE": 920.21, "MAE": 328.15, "R²": 0.9857, "MAPE (%)": 6.20},
    {"Model": "XGBoost", "RMSE": 663.59, "MAE": 262.47, "R²": 0.9925, "MAPE (%)": 7.88},
])

def highlight_best_reg(row):
    is_best = row['Model'] == 'XGBoost'
    return ['background-color: #1e4620' if is_best else '' for _ in row]

st.dataframe(
    reg_results.style.apply(highlight_best_reg, axis=1).format(
        {c: "{:.2f}" for c in reg_results.columns if c != 'Model'}
    ),
    use_container_width=True, hide_index=True
)
st.success("**Selected model: XGBoost** — lowest RMSE (₹663.59) and highest R² (0.993), well past the project's target of RMSE below ₹2,000.")

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
reg_results.set_index('Model')['RMSE'].plot(kind='bar', ax=ax[0], color='steelblue', title='RMSE (lower is better)')
reg_results.set_index('Model')['MAE'].plot(kind='bar', ax=ax[1], color='seagreen', title='MAE (lower is better)')
reg_results.set_index('Model')['R²'].plot(kind='bar', ax=ax[2], color='goldenrod', title='R² (higher is better)')
for a in ax:
    plt.setp(a.get_xticklabels(), rotation=20)
plt.tight_layout()
st.pyplot(fig)

st.divider()

st.subheader("Training Configuration")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Classification**
    - Class imbalance handled via `class_weight='balanced'`
    - 80/20 stratified train-test split
    - Primary metric: Macro F1 (weights all 3 classes equally)
    """)
with col2:
    st.markdown("""
    **Regression**
    - 80/20 train-test split
    - Primary metrics: RMSE, R², MAPE
    - Target: max_monthly_emi (continuous, INR)
    """)

st.caption(
    "Full experiment tracking (parameters, all logged metrics, model artifacts) is available "
    "via MLflow in the training notebook — run `mlflow ui` against the notebook's `mlflow.db` "
    "for the interactive dashboard."
)
