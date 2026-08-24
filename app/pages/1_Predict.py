"""
EMIPredict AI - Real-Time Prediction page
"""

import streamlit as st
import pandas as pd
from utils import load_artifacts, predict, CATEGORY_OPTIONS

st.set_page_config(page_title="Predict — EMIPredict AI", page_icon="\U0001F52E", layout="wide")

st.title("\U0001F52E Real-Time EMI Eligibility & Amount Prediction")

artifacts, missing = load_artifacts()
if missing:
    st.error("Model files missing. Go to the Home page for details on what's needed.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("Enter the applicant's details below to get an instant eligibility assessment.")

with st.form("prediction_form"):
    st.subheader("Personal & Employment Details")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=75, value=35)
        gender = st.selectbox("Gender", CATEGORY_OPTIONS['gender'])
    with c2:
        marital_status = st.selectbox("Marital Status", CATEGORY_OPTIONS['marital_status'])
        education = st.selectbox("Education", CATEGORY_OPTIONS['education'])
    with c3:
        employment_type = st.selectbox("Employment Type", CATEGORY_OPTIONS['employment_type'])
        company_type = st.selectbox("Company Type", CATEGORY_OPTIONS['company_type'])
    with c4:
        years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value=40.0, value=5.0, step=0.5)
        house_type = st.selectbox("House Type", CATEGORY_OPTIONS['house_type'])

    st.subheader("Income & Housing")
    c1, c2, c3 = st.columns(3)
    with c1:
        monthly_salary = st.number_input("Monthly Salary (INR)", min_value=0, value=50000, step=1000)
        monthly_rent = st.number_input("Monthly Rent (INR)", min_value=0, value=10000, step=500)
    with c2:
        family_size = st.number_input("Family Size", min_value=1, max_value=15, value=3)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)
    with c3:
        st.write("")  # spacing

    st.subheader("Monthly Expenses")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        school_fees = st.number_input("School Fees", min_value=0, value=0, step=500)
    with c2:
        college_fees = st.number_input("College Fees", min_value=0, value=0, step=500)
    with c3:
        travel_expenses = st.number_input("Travel Expenses", min_value=0, value=2000, step=500)
    with c4:
        groceries_utilities = st.number_input("Groceries/Utilities", min_value=0, value=6000, step=500)
    with c5:
        other_monthly_expenses = st.number_input("Other Expenses", min_value=0, value=1500, step=500)

    st.subheader("Credit & Financial History")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        existing_loans = st.selectbox("Existing Loans", CATEGORY_OPTIONS['existing_loans'])
    with c2:
        current_emi_amount = st.number_input("Current EMI Amount (INR)", min_value=0, value=0, step=500)
    with c3:
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=700)
    with c4:
        bank_balance = st.number_input("Bank Balance (INR)", min_value=0, value=100000, step=5000)
    emergency_fund = st.number_input("Emergency Fund (INR)", min_value=0, value=30000, step=5000)

    st.subheader("Loan Application Details")
    c1, c2, c3 = st.columns(3)
    with c1:
        emi_scenario = st.selectbox("EMI Scenario", CATEGORY_OPTIONS['emi_scenario'])
    with c2:
        requested_amount = st.number_input("Requested Amount (INR)", min_value=1000, value=200000, step=5000)
    with c3:
        requested_tenure = st.number_input("Requested Tenure (months)", min_value=1, max_value=120, value=24)

    submitted = st.form_submit_button("Predict Eligibility", type="primary", use_container_width=True)

if submitted:
    raw_input = {
        'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
        'monthly_salary': monthly_salary, 'employment_type': employment_type,
        'years_of_employment': years_of_employment, 'company_type': company_type,
        'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size,
        'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees,
        'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
        'current_emi_amount': current_emi_amount, 'credit_score': credit_score,
        'bank_balance': bank_balance, 'emergency_fund': emergency_fund,
        'emi_scenario': emi_scenario, 'requested_amount': requested_amount,
        'requested_tenure': requested_tenure,
    }

    with st.spinner("Running prediction..."):
        result = predict(raw_input, artifacts)

    st.divider()
    st.subheader("Result")

    eligibility = result['eligibility']
    color_map = {'Eligible': 'green', 'High_Risk': 'orange', 'Not_Eligible': 'red'}
    st.markdown(f"### Eligibility: :{color_map.get(eligibility, 'gray')}[{eligibility}]")

    if eligibility == 'High_Risk':
        st.warning(
            "This is the model's hardest class to distinguish reliably (lowest per-class "
            "confidence historically) — recommend manual underwriter review rather than "
            "fully automated action."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Class probabilities**")
        proba_df = pd.DataFrame(
            list(result['eligibility_probabilities'].items()), columns=['Class', 'Probability']
        ).sort_values('Probability', ascending=False)
        st.dataframe(proba_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Recommended maximum monthly EMI**")
        st.metric("Max Monthly EMI", f"₹{result['max_monthly_emi']:,.0f}")
        if eligibility == 'Not_Eligible':
            st.caption("Note: for Not_Eligible applicants, this figure represents the model's floor estimate, not a real recommendation to lend.")

    # Log to session history for the Data Management page
    record = dict(raw_input)
    record['predicted_eligibility'] = eligibility
    record['predicted_max_emi'] = round(result['max_monthly_emi'], 2)
    st.session_state.history.append(record)
    st.success(f"Prediction saved to history ({len(st.session_state.history)} total records this session).")
