# ml/explain.py

import os
import shap
import joblib
import matplotlib.pyplot as plt
from features import load_summaries, extract_features

def explain_model():
    print("🔍 Loading model and data...")
    model = joblib.load("ml/model/settlement_model.pkl")
    cases = load_summaries()
    df = extract_features(cases)
    
    X = df.drop("settlement_amount", axis=1)

    print("⚙️ Computing SHAP values...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    os.makedirs("plots", exist_ok=True)

    print("📊 Generating bar plot...")
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Feature Importance (Bar)")
    plt.tight_layout()
    plt.savefig("plots/shap_bar.png")
    plt.close()

    print("📊 Generating beeswarm plot...")
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP Summary (Beeswarm)")
    plt.tight_layout()
    plt.savefig("plots/shap_beeswarm.png")
    plt.close()

    print("✅ SHAP plots saved to /plots")

if __name__ == "__main__":
    explain_model()
