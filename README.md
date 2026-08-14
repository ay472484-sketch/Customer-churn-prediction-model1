# 👥 Customer Churn Prediction

A Streamlit deployment for the Customer Churn Prediction ML project using the supplied `Churn_Modelling.csv` dataset.

## Model

Preprocessing + One-Hot Encoding + Logistic Regression.

The final model is trained on all 10,000 records in the supplied dataset and saved as `customer_churn_model.joblib`.

## Dataset features

The app uses: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, and `EstimatedSalary`. The target is `Exited`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

Keep `app.py`, `customer_churn_model.joblib`, and `requirements.txt` in the same folder.
