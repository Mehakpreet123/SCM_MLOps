import argparse
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import json

def evaluate(model_path, data_path):
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    X = df[[col for col in df.columns if col.startswith("lag_")]]

    for col in ["month", "year"]:
        if col in df.columns:
            X[col] = df[col]
    y = df["Total_Purchases"]

    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    metrics = {"mae": mae, "rmse": rmse}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.pkl")
    parser.add_argument("--data", default="data/processed/monthly_data.csv")
    args = parser.parse_args()
    evaluate(args.model, args.data)
