"""
EMIPredict AI - Intelligent Financial Risk Assessment Platform
Home page. Run with: streamlit run app.py
"""

import streamlit as st
from utils import load_artifacts, REQUIRED_FILES

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="\U0001F4B0",
    layout="wide",
)

st.title("\U0001F4B0 EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")

st.markdown(
    "A data-driven platform for EMI eligibility assessment and maximum safe loan amount "
    "recommendation, built on models trained on 404,800 financial profiles."
)

artifacts, missing = load_artifacts()

if missing:
    st.error(
        "**Model files not found.** This app needs the following files in the same folder "
        "as `app.py`:\n\n" + "\n".join(f"- `{f}`" for f in missing) +
        "\n\nDownload these from your Kaggle notebook's `/kaggle/working` output and place them here."
    )
else:
    st.success("Models loaded and ready.")

st.divider()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Training Records", "404,800")
with col2:
    st.metric("Classifier Accuracy", "94.1%")
with col3:
    st.metric("Regressor R²", "0.993")
with col4:
    st.metric("EMI Scenarios", "5")

st.divider()

st.markdown("### What this platform does")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    **\U0001F52E Real-Time Prediction**
    Enter an applicant's financial profile and get an instant EMI eligibility
    classification (Eligible / High_Risk / Not_Eligible) plus a recommended
    maximum safe monthly EMI amount.
    """)
    st.markdown("""
    **\U0001F4CA Data Explorer**
    Explore the training dataset's patterns -- eligibility distribution,
    credit score relationships, scenario breakdowns -- interactively.
    """)
with c2:
    st.markdown("""
    **\U0001F4C8 Model Performance**
    Review how the classification and regression models compare, including
    the MLflow-tracked evaluation metrics from training.
    """)
    st.markdown("""
    **\U0001F5C2\uFE0F Data Management**
    Add, view, and manage applicant records in a session-based table --
    useful for reviewing a batch of applications in one place.
    """)

st.divider()
st.caption("Use the sidebar to navigate between pages.")

with st.expander("About this model — read before trusting the results"):
    st.markdown("""
    - Classification model: **XGBoost**, tuned via GridSearchCV, trained with class weighting
      to address a significant class imbalance (~77% Not_Eligible, ~18% Eligible, ~4% High_Risk
      in the training data).
    - Regression model: **XGBoost**, predicting maximum safe monthly EMI in INR.
    - Both models were trained on synthetic-but-realistic financial data across 5 EMI scenarios
      (E-commerce, Home Appliances, Vehicle, Personal Loan, Education).
    - This tool is a decision-support aid, not a substitute for human underwriting judgment --
      predictions should be reviewed, especially for High_Risk classifications.
    """)
