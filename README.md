# EMIPredict AI: Financial Risk Assessment

A Streamlit app that takes an applicant's financial profile and predicts two things:

1. **Eligibility** for an EMI loan: Eligible, High Risk or Not Eligible.
2. **Maximum safe monthly EMI** for that applicant.

It is trained on 404,800 financial profiles across 5 loan types: e-commerce shopping, home appliances, vehicles, personal loans and education.

**[Live app](https://emipredict-ai-financial-risk-platform-ahad.streamlit.app/)** · **[Training notebook](notebook/EMIPredict_AI_Notebook.ipynb)** ·

---

## Why I built this

Checking whether you can afford an EMI before signing up is hard, and lenders who review every application by hand cannot scale. I wanted a tool that takes the same inputs an underwriter looks at (income, expenses, credit history, the loan requested) and gives a quick first answer.

It is a decision-support demo. It does not replace a lender's review.

## The data problem

Three columns that should have been plain numbers (`age`, `monthly_salary`, `bank_balance`) were stored as text, and many values had a doubled suffix, like `"64300.0.0"` instead of `64300.0`. Calling `pd.to_numeric()` on them would have turned most of those rows into NaN with no warning, and I would not have noticed until the model behaved oddly.

I fixed it with a small regex cleaner. I also:

- merged 8 spellings of `gender` (`Female`, `female`, `FEMALE`, `F` and others) into clean values
- clipped `credit_score` back into its valid 300 to 850 range, since the raw data had values from 0 to 1200

Next time I will check the unique values of a column before I trust a type conversion.

## How it works

**Features.** Instead of giving the model raw salary and expense numbers, I built the ratios an underwriter would check: debt-to-income, expense-to-income, an affordability ratio for the requested loan, disposable income, and a risk score that combines credit history, job stability and dependents.

**Models.** Two problems, six models:

- Classification: Logistic Regression, Random Forest, XGBoost
- Regression: Linear Regression, Random Forest, XGBoost
- Every run is logged in MLflow, so I could compare models properly.

**Class imbalance.** About 77% of applicants are Not Eligible, 18% are Eligible and about 4% are High Risk. A model that always says "Not Eligible" would score 77% accuracy and be useless. I used class weighting and judged the models on Macro F1 and ROC-AUC, not accuracy.

## Results

**Eligibility (classification)**

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 81.6% | 0.668 | 0.971 |
| Random Forest | 92.0% | 0.782 | 0.993 |
| **XGBoost** | **94.1%** | **0.836** | **0.998** |

**Maximum monthly EMI (regression)**

| Model | RMSE (₹) | MAE (₹) | R² |
|---|---|---|---|
| Linear Regression | 4,086 | 2,942 | 0.717 |
| Random Forest | 920 | 328 | 0.986 |
| **XGBoost** | **664** | **262** | **0.993** |

XGBoost won both tasks. The project brief asked for classification accuracy above 90% and regression RMSE under ₹2,000, and both targets were met. The final regressor has a MAPE of 7.88%, so its predictions are off by about 8% on average, even with an R² of 0.993.

## Limitations

- **The scores are very high.** On a large tabular dataset like this, scores this high often mean the labels follow fixed rules built from the input columns, so the model may be relearning those rules. It has not been tested against real lending decisions, so these numbers do not show how it would do with a real lender.
- **The Data Explorer page needs the raw dataset.** The data came from Kaggle through the internship, and I do not have clear rights to redistribute it, so it is not in this repo. The page shows a fallback message. The Predict page works without it.
- **Data Management is not a database.** It uses Streamlit session state, so everything is gone when you refresh. A real lending platform would need a proper database and multi-user handling.
- **High Risk is the weakest class.** It is the smallest class and the one the model is least sure about. The app flags High Risk predictions for manual review and says so on the page.

## The app

- **Predict:** enter applicant details and get an eligibility result, a recommended maximum EMI and the probability for each class.
- **Data Explorer:** interactive charts of the training data, filterable by scenario and eligibility class.
- **Model Performance:** the comparison tables above, with the winning models highlighted.
- **Data Management:** a table of this session's predictions that you can export to CSV.

### Screenshots

Home page with model status and key metrics:

<img width="1896" height="907" alt="EMIPredict AI home page with model status" src="https://github.com/user-attachments/assets/870d8f4a-e024-4241-8eb3-c52bef59c6f0" />

Prediction with class probabilities:

<img width="1897" height="891" alt="Prediction result with confidence breakdown" src="https://github.com/user-attachments/assets/ab7a774d-e725-4271-ab06-3608a00408a5" />

Model comparison:

<img width="1895" height="912" alt="Model comparison dashboard" src="https://github.com/user-attachments/assets/6e98ff8c-467f-4be9-a76b-cf55da6dd0cb" />

<details>
<summary>More screenshots</summary>

<img width="1891" height="908" alt="Home page, metrics section" src="https://github.com/user-attachments/assets/75f4caad-8db9-4080-9056-af5f55405d87" />
<img width="1891" height="822" alt="Prediction form" src="https://github.com/user-attachments/assets/78cd4ceb-a5d6-48a9-a77c-7592a2e124b7" />
<img width="1512" height="713" alt="Prediction details" src="https://github.com/user-attachments/assets/8a587cb0-65f1-4399-b218-84588973e075" />
<img width="1515" height="696" alt="Classification model comparison" src="https://github.com/user-attachments/assets/e51f5590-7035-4fa0-abd8-03545c4aca2a" />
<img width="1517" height="775" alt="Regression model comparison" src="https://github.com/user-attachments/assets/270e6431-a15e-461d-a190-8e59ea8d90e6" />
<img width="1507" height="393" alt="Model comparison summary" src="https://github.com/user-attachments/assets/20ed68b5-33a5-4607-a695-b405a2bc3455" />

</details>

## Project structure

```
├── app/
│   ├── app.py                          # Streamlit entry point
│   ├── utils.py                        # Shared helper code
│   ├── pages/                          # Predict, Data Explorer, Model Performance, Data Management
│   ├── emi_eligibility_model.joblib    # XGBoost classifier
│   ├── max_emi_model.joblib            # XGBoost regressor
│   ├── emi_feature_scaler.joblib       # Feature scaler
│   ├── emi_eligibility_label_encoder.joblib
│   ├── emi_feature_columns.joblib      # Feature names and order
│   └── requirements.txt
├── notebook/
│   └── EMIPredict_AI_Notebook.ipynb    # EDA, features, models, MLflow
└── README.md
```

## Run it yourself

```bash
git clone https://github.com/AhadAhmad0/emipredict-ai-financial-risk-platform.git
cd emipredict-ai-financial-risk-platform/app
pip install -r requirements.txt
streamlit run app.py
```

The five saved files are already in `app/`: two models, a scaler, a label encoder and the feature list. To retrain from scratch, run the notebook in `notebook/`.

## Stack

Python, pandas, scikit-learn, XGBoost, MLflow, Streamlit, Streamlit Community Cloud

## Author

Ahad Ahmad
- GitHub: [@AhadAhmad0](https://github.com/AhadAhmad0)
- LinkedIn: [linkedin.com/in/ahadahmad7](https://linkedin.com/in/ahadahmad7/)
