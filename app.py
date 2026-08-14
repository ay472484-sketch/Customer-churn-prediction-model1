import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Customer Churn Prediction", page_icon="👥", layout="centered")
MODEL_PATH = Path(__file__).parent / "customer_churn_model.joblib"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

st.title("👥 Customer Churn Prediction")
st.write("Enter customer details to estimate the probability of churn.")
model = load_model()
if model is None:
    st.error("Model file not found.")
    st.info("Place customer_churn_model.joblib in the same folder as app.py.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    credit_score = st.number_input("Credit Score", 300, 900, 650, 1)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", 18, 100, 40, 1)
    tenure = st.number_input("Tenure (years)", 0, 20, 5, 1)
with col2:
    balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=1000.0)
    num_products = st.number_input("Number of Products", 1, 10, 2, 1)
    has_card = st.selectbox("Has Credit Card", [1, 0])
    active_member = st.selectbox("Is Active Member", [1, 0])
    salary = st.number_input("Estimated Salary", min_value=0.0, value=100000.0, step=1000.0)

if st.button("🔮 Predict Churn", use_container_width=True):
    row = pd.DataFrame([{
        "CreditScore": credit_score, "Geography": geography, "Gender": gender,
        "Age": age, "Tenure": tenure, "Balance": balance,
        "NumOfProducts": num_products, "HasCrCard": has_card,
        "IsActiveMember": active_member, "EstimatedSalary": salary,
    }])
    try:
        probability = float(model.predict_proba(row)[:, 1][0])
        prediction = int(model.predict(row)[0])
        st.metric("Churn Probability", f"{probability:.1%}")
        if prediction == 1:
            st.error("⚠️ Customer is predicted to churn.")
        else:
            st.success("✅ Customer is predicted to stay.")
    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))

st.divider()
st.caption("Machine Learning Project • Logistic Regression • Customer Retention")
