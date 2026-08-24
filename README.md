# EMIPredict AI 💰

*Intelligent Financial Risk Assessment Platform*

**[Live App](https://emipredict-ai-financial-risk-platform-ahad.streamlit.app/)** &nbsp;·&nbsp; **[Training Notebook](notebook/emi-prediction.ipynb)** &nbsp;·&nbsp; Built for the Labmentix AI/ML Internship

---

## What this actually does

A lot of people take on EMIs they can't really afford — not because they're irresponsible, but because there's no easy way to check "can I actually handle this?" before signing up. Lenders have the same problem from the other side: manually underwriting every application doesn't scale.

This project tackles both sides at once. Feed it an applicant's financial profile — income, expenses, credit history, the loan they're asking for — and it tells you two things:

1. **Are they eligible?** (Eligible / High Risk / Not Eligible)
2. **If so, what's a safe maximum monthly EMI for them?**

It's trained on 404,800 financial profiles across 5 loan types (E-commerce shopping, home appliances, vehicles, personal loans, education), wrapped in a Streamlit app you can actually click through.

## The data was messier than expected

Before any of the modelling was interesting, I had to deal with a genuinely annoying data quality issue: three columns that should've been plain numbers — `age`, `monthly_salary`, `bank_balance` — were stored as text, and a chunk of the values had corrupted, repeated `.0.0` suffixes. Something like `"64300.0.0"` instead of `64300.0`. If I'd just thrown `pd.to_numeric()` at it, most of those rows would've silently turned into NaN and I probably wouldn't have noticed until the model started behaving strangely.

Fixed it with a small regex cleaner instead. Also had to normalize 8 different casing variants of `gender` (`Female`, `female`, `FEMALE`, `F`...) and clip `credit_score` back into its documented 300–850 range, since the raw data had values as low as 0 and as high as 1200.

Lesson from this one: always eyeball your unique values before trusting a type conversion on a big dataset.

## What's under the hood

**Feature engineering** — rather than throwing raw salary and expense numbers at the model and hoping it figures things out, I built the same ratios a real underwriter would look at: debt-to-income, expense-to-income, an affordability ratio based on the requested loan, disposable income, and a composite risk score blending credit history, employment stability, and dependents.

**Two problems, six models** — three classifiers (Logistic Regression, Random Forest, XGBoost) and three regressors (Linear Regression, Random Forest, XGBoost), every single training run logged to **MLflow** so I could compare them properly instead of just eyeballing print statements.

**The eligibility classes are lopsided** — about 77% of applicants are Not_Eligible, 18% Eligible, and only ~4% High_Risk. That imbalance is exactly the kind of thing that lets a lazy model look good on paper (just predict "Not_Eligible" every time and you're 77% accurate) while being useless in practice. Handled it with class weighting and judged models on Macro F1 and ROC-AUC instead of raw accuracy.

## Results

**Classification (EMI Eligibility)**

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 81.6% | 0.668 | 0.971 |
| Random Forest | 92.0% | 0.782 | 0.993 |
| **XGBoost** ⭐ | **94.1%** | **0.836** | **0.998** |

**Regression (Max Monthly EMI)**

| Model | RMSE (₹) | MAE (₹) | R² |
|---|---|---|---|
| Linear Regression | 4,086 | 2,942 | 0.717 |
| Random Forest | 920 | 328 | 0.986 |
| **XGBoost** ⭐ | **664** | **262** | **0.993** |

XGBoost won both, comfortably. For context, the project brief asked for classification accuracy above 90% and regression RMSE under ₹2,000 — this clears both with room to spare.

## The app

Four pages beyond the landing page:

- **Predict** — fill in an applicant's details, get an instant eligibility call plus a recommended max EMI, with class probabilities so you can see how confident the model actually is (and it flags High_Risk predictions for manual review rather than pretending to be certain).
- **Data Explorer** — interactive charts over the training data, filterable by scenario and eligibility class.
- **Model Performance** — the comparison tables above, rendered properly, with the winning models highlighted.
- **Data Management** — a session-based table of everything you've predicted this session, exportable to CSV.

### App Preview

**Home — model status and key metrics at a glance:**
<img width="1896" height="907" alt="Screenshot 2026-08-24 171851" src="https://github.com/user-attachments/assets/870d8f4a-e024-4241-8eb3-c52bef59c6f0" />
<img width="1891" height="908" alt="Screenshot 2026-08-24 171903" src="https://github.com/user-attachments/assets/75f4caad-8db9-4080-9056-af5f55405d87" />

**Real-time prediction with confidence breakdown:**
<img width="1897" height="891" alt="Screenshot 2026-08-24 171937" src="https://github.com/user-attachments/assets/ab7a774d-e725-4271-ab06-3608a00408a5" />
<img width="1891" height="822" alt="Screenshot 2026-08-24 171946" src="https://github.com/user-attachments/assets/78cd4ceb-a5d6-48a9-a77c-7592a2e124b7" />
<img width="1512" height="713" alt="Screenshot 2026-08-24 171956" src="https://github.com/user-attachments/assets/8a587cb0-65f1-4399-b218-84588973e075" />

**Model comparison dashboard:**
<img width="1895" height="912" alt="Screenshot 2026-08-24 172012" src="https://github.com/user-attachments/assets/6e98ff8c-467f-4be9-a76b-cf55da6dd0cb" />
<img width="1515" height="696" alt="Screenshot 2026-08-24 172025" src="https://github.com/user-attachments/assets/e51f5590-7035-4fa0-abd8-03545c4aca2a" />
<img width="1517" height="775" alt="Screenshot 2026-08-24 172037" src="https://github.com/user-attachments/assets/270e6431-a15e-461d-a190-8e59ea8d90e6" />
<img width="1507" height="393" alt="Screenshot 2026-08-24 172045" src="https://github.com/user-attachments/assets/20ed68b5-33a5-4607-a695-b405a2bc3455" />


## Being upfront about the limits

- **Data Explorer needs the raw dataset to render**, and I didn't include it in this repo — it came from Kaggle via the internship and I don't have clear redistribution rights for it. The page shows an honest fallback message instead of a public dataset it shouldn't have. Predictions on the Predict page work completely independently of this.
- **Data Management isn't a real database.** It's Streamlit's session state — refresh the page and it's gone. Fine for a demo, not what you'd ship for an actual lending platform (that'd need Postgres or similar, with proper multi-user handling).
- **This is a decision-support tool, not an autopilot.** Especially for High_Risk classifications — the model flags these as its least confident category, and the app says so out loud rather than pretending otherwise.

## Running it yourself

```bash
git clone https://github.com/AhadAhmad0/emipredict-ai-financial-risk-platform.git
cd emipredict-ai-financial-risk-platform/app
pip install -r requirements.txt
streamlit run app.py
```

You'll need the 5 model files (already included in `app/`) — if you want to retrain from scratch, the full notebook is in `notebook/`.

## Stack

Python, pandas, scikit-learn, XGBoost, MLflow, Streamlit, Streamlit Cloud

---

**Ahad Ahmad** — [GitHub](https://github.com/AhadAhmad0) · [LinkedIn](https://linkedin.com/in/ahadahmad7/)
