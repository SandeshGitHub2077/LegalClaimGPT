# ml/predictor.py

import os
import joblib
import json
import pandas as pd
from features import extract_features

MODEL_PATH = "ml/model/settlement_model.pkl"
model = joblib.load(MODEL_PATH)

def predict_batch(summaries_path="data/processed/summaries.json"):
    with open(summaries_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    df = extract_features(cases)
    X = df.drop("settlement_amount", axis=1, errors="ignore")
    predictions = model.predict(X)

    for case, pred in zip(cases, predictions):
        case["predicted_settlement"] = round(pred, 2)

    return cases

def predict_single(summary_data: dict):
    features_df = extract_features([summary_data])
    X = features_df.drop("settlement_amount", axis=1, errors="ignore")
    prediction = model.predict(X)[0]
    return round(prediction, 2)

if __name__ == "__main__":
    print("🎯 Choose prediction mode:")
    print("1. Predict all cases in batch")
    print("2. Paste a single new case summary\n")

    mode = input("Enter 1 or 2 [1/2] (1): ").strip() or "1"

    if mode == "1":
        print("\n📁 Loading summaries from file...")
        results = predict_batch()
        print(f"\n✅ Batch prediction complete! {len(results)} cases processed.")
        for case in results:
            name = case.get("case_name", "unknown")[:40]
            print(f" - {name} → ${case['predicted_settlement']} 💰")
    else:
        print("\n📋 Paste your summary JSON (use correct keys):")
        user_input = input()
        try:
            case = json.loads(user_input)
            prediction = predict_single(case)
            print(f"\n💡 Predicted settlement amount: ${prediction}")
        except Exception as e:
            print("❌ Invalid input format or prediction failed.")
            print(str(e))
