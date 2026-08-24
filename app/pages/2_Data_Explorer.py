"""
EMIPredict AI - Data Explorer page
Interactive exploration of the training dataset. Expects emi_prediction_dataset.csv
in the same folder as app.py (optional -- page degrades gracefully if not present,
since a live deployment may not want to ship the raw training data).
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data Explorer — EMIPredict AI", page_icon="\U0001F4CA", layout="wide")

st.title("\U0001F4CA Data Explorer")

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "emi_prediction_dataset.csv")


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH, low_memory=False)
    # light cleaning for display purposes only
    import re
    def clean_numeric(series):
        def _clean(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip()
            if s.lower() == 'nan':
                return np.nan
            s = re.sub(r'(\.0)+$', '', s)
            try:
                return float(s)
            except ValueError:
                return np.nan
        return series.apply(_clean)
    for col in ['age', 'monthly_salary', 'bank_balance']:
        df[col] = clean_numeric(df[col])
    df['credit_score'] = df['credit_score'].clip(lower=300, upper=850)
    return df


df = load_data()

if df is None:
    st.warning(
        "Training dataset (`emi_prediction_dataset.csv`) not found alongside this app. "
        "This page needs the raw dataset to render — it's optional for deployment since "
        "you may not want to ship the training data publicly. Predictions on the "
        "**Predict** page work independently of this page."
    )
    st.stop()

st.markdown(f"Exploring **{len(df):,}** applicant records across **5** EMI scenarios.")

st.sidebar.header("Filters")
scenario_filter = st.sidebar.multiselect(
    "EMI Scenario", options=sorted(df['emi_scenario'].unique()), default=None
)
eligibility_filter = st.sidebar.multiselect(
    "Eligibility", options=sorted(df['emi_eligibility'].unique()), default=None
)

filtered_df = df.copy()
if scenario_filter:
    filtered_df = filtered_df[filtered_df['emi_scenario'].isin(scenario_filter)]
if eligibility_filter:
    filtered_df = filtered_df[filtered_df['emi_eligibility'].isin(eligibility_filter)]

st.caption(f"Showing {len(filtered_df):,} of {len(df):,} records after filters.")

tab1, tab2, tab3, tab4 = st.tabs(["Eligibility Overview", "Financial Patterns", "By Scenario", "Raw Data"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Eligibility Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        filtered_df['emi_eligibility'].value_counts().plot(
            kind='bar', color=['firebrick', 'goldenrod', 'seagreen'], ax=ax
        )
        ax.set_ylabel("Count")
        st.pyplot(fig)
    with col2:
        st.subheader("Credit Score by Eligibility")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='emi_eligibility', y='credit_score', data=filtered_df,
                    order=['Not_Eligible', 'High_Risk', 'Eligible'], palette='coolwarm', ax=ax)
        st.pyplot(fig)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Max Monthly EMI Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df['max_monthly_emi'], bins=40, kde=True, color='steelblue', ax=ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Monthly Salary Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df['monthly_salary'].dropna(), bins=40, kde=True, color='seagreen', ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numeric_cols = ['age', 'monthly_salary', 'credit_score', 'bank_balance', 'current_emi_amount',
                     'requested_amount', 'requested_tenure', 'max_monthly_emi']
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tab3:
    st.subheader("Eligibility Rate by EMI Scenario")
    fig, ax = plt.subplots(figsize=(10, 5))
    pd.crosstab(filtered_df['emi_scenario'], filtered_df['emi_eligibility'], normalize='index').plot(
        kind='bar', stacked=True, colormap='RdYlGn_r', ax=ax
    )
    ax.set_ylabel("Proportion")
    plt.xticks(rotation=20)
    st.pyplot(fig)

    st.subheader("Average Requested Amount by Scenario")
    avg_by_scenario = filtered_df.groupby('emi_scenario')['requested_amount'].mean().sort_values(ascending=False)
    st.bar_chart(avg_by_scenario)

with tab4:
    st.subheader("Raw Data (filtered)")
    st.dataframe(filtered_df.head(500), use_container_width=True)
    st.caption("Showing first 500 rows of the filtered selection.")
