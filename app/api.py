from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("ml/model/settlement_model.pkl")

class CaseInput(BaseModel):
    summary: str
    injuries: list
    medical_bills: float
    lost_wages: float
    age: int
    gender: str

@app.get("/")
def read_root():
    return {"message": "LegalClaimGPT Settlement Prediction API is running."}

@app.post("/predict")
def predict(case: CaseInput):
    try:
        features = {
            "num_injuries": len(case.injuries),
            "has_severe_injury": int(any(word in str(case.injuries).lower() for word in ["brain", "spinal", "burn"])),
            "medical_bills": float(case.medical_bills),
            "lost_wages": float(case.lost_wages),
            "age": int(case.age),
            "is_male": 1 if case.gender.lower() == "male" else 0,
        }

        X = [list(features.values())]  # Wrap in list to make it 2D
        prediction = model.predict(X)[0]
        return {"predicted_settlement": round(float(prediction), 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
