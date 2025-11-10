import joblib
import pandas as pd

def predict(new_data_dict):
    """Make prediction on a single input dictionary with lag features."""
    model = joblib.load("models/model.pkl")
    X = pd.DataFrame([new_data_dict])
    prediction = model.predict(X)[0]
    return prediction
