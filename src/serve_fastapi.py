from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Monthly Demand Prediction API")

model = joblib.load("models/model.pkl")

class PredictionInput(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    lag_4: float
    lag_6: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    df = pd.DataFrame([input_data.dict()])
    pred = model.predict(df)[0]
    return {"prediction": float(pred)}
