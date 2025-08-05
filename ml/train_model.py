# ml/train_model.py

import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from features import load_summaries, extract_features

def train():
    cases = load_summaries()
    df = extract_features(cases)

    X = df.drop("settlement_amount", axis=1)
    y = df["settlement_amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("✅ Model trained.")
    print("📊 MAE:", mean_absolute_error(y_test, y_pred))
    print("📈 R²:", r2_score(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "ml/model/settlement_model.pkl")
    print("💾 Model saved to model/settlement_model.pkl")

if __name__ == "__main__":
    train()
