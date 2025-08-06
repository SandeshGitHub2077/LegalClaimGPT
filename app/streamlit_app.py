# app/streamlit_app.py

import streamlit as st
import requests
import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import spacy
import re

st.set_page_config(page_title="LegalClaimGPT", layout="centered")

API_URL = "http://127.0.0.1:8000/predict"
MODEL_PATH = "ml/model/settlement_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
explainer = shap.Explainer(model)
nlp = spacy.load("en_core_web_sm")

st.title("💼 LegalClaimGPT Settlement Estimator")
st.markdown("Estimate personal injury settlements using AI + case features.")

def extract_features_from_summary(text):
    doc = nlp(text)

    gender = "Female" if re.search(r'\b(woman|female|she|her)\b', text.lower()) else "Male"

    age_match = re.search(r"(\d+)-year-old", text)
    age = int(age_match.group(1)) if age_match else 0

    injuries = []
    injury_keywords = ["fracture", "injury", "brain", "burn", "spinal", "whiplash", "concussion"]
    for token in doc:
        if token.text.lower() in injury_keywords:
            injuries.append(" ".join([child.text for child in token.subtree]))

    bills_match = re.search(r"\$?([\d,]+)\s*(in)?\s*(medical bills|bills|treatment)", text.lower())
    medical_bills = int(bills_match.group(1).replace(",", "")) if bills_match else 0

    wage_match = re.search(r"lost\s+(\d+)\s+months?", text.lower())
    lost_wages = int(wage_match.group(1)) * 6000 if wage_match else 0

    return {
        "summary": text,
        "injuries": list(set(injuries)) or ["unspecified"],
        "medical_bills": medical_bills,
        "lost_wages": lost_wages,
        "age": age,
        "gender": gender
    }

mode = st.radio("Select input mode", ["Manual Entry", "Paste Case Summary"])

if mode == "Manual Entry":
    summary = st.text_area("Case Summary", height=150)
    injuries_input = st.text_input("Injuries (comma-separated)", "spinal cord injury, fracture")
    injuries = [inj.strip() for inj in injuries_input.split(",")]
    medical_bills = st.number_input("Medical Bills (USD)", min_value=0)
    lost_wages = st.number_input("Lost Wages (USD)", min_value=0)
    age = st.number_input("Plaintiff Age", min_value=0)
    gender = st.selectbox("Gender", ["Female", "Male"])

else:
    pasted_summary = st.text_area("Paste full case description")
    if pasted_summary:
        extracted = extract_features_from_summary(pasted_summary)
        st.success("Auto-extracted features:")
        st.write(extracted)

        summary = extracted["summary"]
        injuries = extracted["injuries"]
        medical_bills = extracted["medical_bills"]
        lost_wages = extracted["lost_wages"]
        age = extracted["age"]
        gender = extracted["gender"]
    else:
        st.stop()

if st.button("Predict Settlement 💰"):
    payload = {
        "summary": summary,
        "injuries": injuries,
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

        features = {
            "num_injuries": len(payload["injuries"]),
            "has_severe_injury": int(any(severe in injury.lower() for injury in payload["injuries"] for severe in ["brain", "spinal", "burn"])),
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

        lower = prediction * 0.9
        upper = prediction * 1.1
        st.subheader("📊 Confidence Range")
        st.info(f"Estimated range: ${lower:,.2f} - ${upper:,.2f}")

    except requests.exceptions.RequestException as e:
        st.error("🔌 Could not connect to the API. Is it running?")