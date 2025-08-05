# app/streamlit_app.py

import streamlit as st
import requests
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="LegalClaimGPT", layout="centered")

API_URL = "http://127.0.0.1:8000/predict"
MODEL_PATH = "ml/model/settlement_model.pkl"

# Load model for SHAP
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
explainer = shap.Explainer(model)

st.title("💼 LegalClaimGPT Settlement Estimator")

st.markdown("Estimate personal injury settlements using AI + case features.")

# User inputs
summary = st.text_area("Case Summary", height=150)
injuries = st.text_input("Injuries (comma-separated)", "spinal cord injury, fracture")
medical_bills = st.number_input("Medical Bills (USD)", min_value=0)
lost_wages = st.number_input("Lost Wages (USD)", min_value=0)
age = st.number_input("Plaintiff Age", min_value=0)
gender = st.selectbox("Gender", ["Female", "Male"])

if st.button("Predict Settlement 💰"):
    # Prepare input for API
    payload = {
        "summary": summary,
        "injuries": [i.strip() for i in injuries.split(",") if i.strip()],
        "medical_bills": medical_bills,
        "lost_wages": lost_wages,
        "age": age,
        "gender": gender,
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        prediction = result["predicted_settlement"]

        st.success(f"💰 Estimated Settlement: **${prediction:,.2f}**")

        # Feature extraction for SHAP
        features = {
            "num_injuries": len(payload["injuries"]),
            "has_severe_injury": int(any(word in injuries.lower() for word in ["brain", "spinal", "burn"])),
            "medical_bills": float(medical_bills),
            "lost_wages": float(lost_wages),
            "age": int(age),
            "is_male": 1 if gender.lower() == "male" else 0,
        }

        X_input = pd.DataFrame([features])
        shap_values = explainer(X_input)

        st.subheader("🔍 Feature Impact")
        st.markdown("SHAP values show how each input influenced the prediction.")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

        # Display naive confidence (±10% band as proxy)
        lower = prediction * 0.9
        upper = prediction * 1.1
        st.subheader("📊 Confidence Range")
        st.info(f"Estimated range: ${lower:,.2f} - ${upper:,.2f}")

    except requests.exceptions.RequestException as e:
        st.error("🔌 Could not connect to the API. Is it running?")
