# ml/features.py

import json
import numpy as np
import pandas as pd

def load_summaries(path="data/processed/summaries.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_features(cases):
    records = []
    for case in cases:
        summary = case.get("summary", "")
        injuries = case.get("injuries", [])
        medical_bills = case.get("medical_bills", 0)
        lost_wages = case.get("lost_wages", 0)
        age = case.get("age", 0)
        gender = case.get("gender", "Unknown")
        settlement = case.get("settlement_amount", None)

        records.append({
            "num_injuries": len(injuries),
            "has_severe_injury": int(any(word in str(injuries).lower() for word in ["brain", "spinal", "burn"])),
            "medical_bills": float(medical_bills) if medical_bills is not None else 0.0,
            "lost_wages": float(lost_wages) if lost_wages is not None else 0.0,
            "age": int(age) if age else 0,
            "is_male": 1 if gender.lower() == "male" else 0,
            "settlement_amount": float(settlement) if settlement is not None else 0.0,
        })

    df = pd.DataFrame(records)
    df = df.dropna(subset=["settlement_amount"])  # ✅ Drop cases without settlement amount
    return df
