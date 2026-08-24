"""
EMIPredict AI - Shared utilities
Loads the trained models/artifacts and replicates the exact preprocessing
pipeline from the training notebook (feature engineering -> one-hot encoding
-> scaling) so a single raw applicant record can be turned into a model-ready
row at inference time.
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

CATEGORICAL_COLS = ['gender', 'marital_status', 'education', 'employment_type',
                     'company_type', 'house_type', 'existing_loans', 'emi_scenario']

NUMERIC_FEATURES = ['age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
                     'family_size', 'dependents', 'school_fees', 'college_fees',
                     'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
                     'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund',
                     'requested_amount', 'requested_tenure', 'total_monthly_expenses',
                     'debt_to_income', 'expense_to_income', 'affordability_ratio',
                     'disposable_income', 'risk_score']

CATEGORY_OPTIONS = {
    'gender': ['Male', 'Female'],
    'marital_status': ['Married', 'Single'],
    'education': ['High School', 'Graduate', 'Post Graduate', 'Professional'],
    'employment_type': ['Private', 'Government', 'Self-employed'],
    'company_type': ['Mid-size', 'MNC', 'Startup', 'Large Indian', 'Small'],
    'house_type': ['Rented', 'Family', 'Own'],
    'existing_loans': ['Yes', 'No'],
    'emi_scenario': ['E-commerce Shopping EMI', 'Home Appliances EMI', 'Vehicle EMI',
                      'Personal Loan EMI', 'Education EMI'],
}

REQUIRED_FILES = [
    'emi_eligibility_model.joblib',
    'max_emi_model.joblib',
    'emi_feature_scaler.joblib',
    'emi_eligibility_label_encoder.joblib',
    'emi_feature_columns.joblib',
]


@st.cache_resource(show_spinner="Loading models...")
def load_artifacts():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        return None, missing

    clf_model = joblib.load(os.path.join(MODEL_DIR, 'emi_eligibility_model.joblib'))
    reg_model = joblib.load(os.path.join(MODEL_DIR, 'max_emi_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'emi_feature_scaler.joblib'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'emi_eligibility_label_encoder.joblib'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'emi_feature_columns.joblib'))

    artifacts = {
        'clf_model': clf_model,
        'reg_model': reg_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_cols': feature_cols,
    }
    return artifacts, []


def engineer_features(raw: dict) -> dict:
    """Recreate the exact engineered features from the training notebook for one applicant."""
    salary = raw['monthly_salary'] if raw['monthly_salary'] > 0 else np.nan

    total_monthly_expenses = (
        raw['monthly_rent'] + raw['school_fees'] + raw['college_fees'] +
        raw['travel_expenses'] + raw['groceries_utilities'] + raw['other_monthly_expenses'] +
        raw['current_emi_amount']
    )
    debt_to_income = (raw['current_emi_amount'] + raw['monthly_rent']) / salary if salary else 0.0
    expense_to_income = total_monthly_expenses / salary if salary else 0.0
    tenure = raw['requested_tenure'] if raw['requested_tenure'] > 0 else np.nan
    affordability_ratio = (raw['requested_amount'] / tenure) / salary if (salary and tenure) else 0.0
    disposable_income = raw['monthly_salary'] - total_monthly_expenses
    risk_score = (
        (raw['credit_score'] / 850) * 0.5 +
        (min(raw['years_of_employment'], 15) / 15) * 0.3 -
        (min(raw['dependents'], 5) / 5) * 0.2
    )

    engineered = dict(raw)
    engineered.update({
        'total_monthly_expenses': total_monthly_expenses,
        'debt_to_income': 0.0 if pd.isna(debt_to_income) else debt_to_income,
        'expense_to_income': 0.0 if pd.isna(expense_to_income) else expense_to_income,
        'affordability_ratio': 0.0 if pd.isna(affordability_ratio) else affordability_ratio,
        'disposable_income': disposable_income,
        'risk_score': risk_score,
    })
    return engineered


def preprocess_input(raw: dict, artifacts: dict) -> pd.DataFrame:
    """Turn one raw applicant dict into a model-ready row: engineer -> one-hot -> scale -> align."""
    engineered = engineer_features(raw)
    row_df = pd.DataFrame([engineered])

    # One-hot encode categoricals the same way pd.get_dummies(..., drop_first=True) did in training
    row_encoded = pd.get_dummies(row_df, columns=CATEGORICAL_COLS, drop_first=True)

    # Align to the exact training feature column set/order -- any dummy column not
    # produced by this single row (because that category wasn't the "active" one) is
    # filled with 0, matching how one-hot encoding works for unseen combinations.
    feature_cols = artifacts['feature_cols']
    row_aligned = row_encoded.reindex(columns=feature_cols, fill_value=0)

    # Scale numeric features with the exact fitted scaler from training
    scaler = artifacts['scaler']
    row_aligned[NUMERIC_FEATURES] = scaler.transform(row_aligned[NUMERIC_FEATURES])

    return row_aligned


def predict(raw: dict, artifacts: dict) -> dict:
    """Run both models on one applicant record and return eligibility + max EMI + probabilities."""
    X = preprocess_input(raw, artifacts)

    clf_model = artifacts['clf_model']
    reg_model = artifacts['reg_model']
    label_encoder = artifacts['label_encoder']

    clf_pred_enc = clf_model.predict(X)[0]
    clf_proba = clf_model.predict_proba(X)[0]
    clf_pred_label = label_encoder.inverse_transform([clf_pred_enc])[0]

    reg_pred = reg_model.predict(X)[0]
    # The training data's max_monthly_emi floor is documented as INR 500 -- models can
    # extrapolate below this (or below 0) for very weak applicants near the training
    # distribution's edge, which isn't a meaningful "maximum EMI" in practice. Clip to the
    # documented valid range rather than surfacing a nonsensical negative recommendation.
    reg_pred = max(500.0, float(reg_pred))

    class_labels = label_encoder.inverse_transform(np.arange(len(clf_proba)))
    proba_dict = {cls: float(p) for cls, p in zip(class_labels, clf_proba)}

    return {
        'eligibility': clf_pred_label,
        'eligibility_probabilities': proba_dict,
        'max_monthly_emi': reg_pred,
    }
