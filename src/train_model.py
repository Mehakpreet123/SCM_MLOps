# src/train_model.py
import argparse
import os
import pandas as pd
import yaml
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import mlflow
import mlflow.xgboost

def train_model(data_path, params_path):
    # ---------------------------
    # 1. Load hyperparameters
    # ---------------------------
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # ---------------------------
    # 2. Load training data
    # ---------------------------
    df = pd.read_csv(data_path)
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    lags = params["train"]["lags"]

    # Check all lag features exist
    missing_lags = [f"lag_{l}" for l in lags if f"lag_{l}" not in df.columns]
    if missing_lags:
        raise ValueError(f"❌ Missing lag features: {missing_lags}")

    # ---------------------------
    # 3. Prepare features & target
    # ---------------------------
    X = df[[f"lag_{l}" for l in lags]]
    y = df["Total_Purchases"]

    # Optional: add time-based features if available
    if "month" in df.columns:
        X["month"] = df["month"]
    if "year" in df.columns:
        X["year"] = df["year"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["train"].get("test_size", 0.2),
        shuffle=False,
    )

    # ---------------------------
    # 4. Initialize model
    # ---------------------------
    model = XGBRegressor(
        n_estimators=params["train"]["n_estimators"],
        learning_rate=params["train"]["learning_rate"],
        max_depth=params["train"]["max_depth"],
        subsample=params["train"].get("subsample", 0.8),
        colsample_bytree=params["train"].get("colsample_bytree", 0.8),
        random_state=params["train"]["random_state"],
    )

    # ---------------------------
    # 5. Set up MLflow tracking
    # ---------------------------
    tracking_uri = params["experiment"]["mlflow_tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["experiment"]["mlflow_experiment"])

    # ---------------------------
    # 6. Train, evaluate, log
    # ---------------------------
    with mlflow.start_run():
        mlflow.log_params(params["train"])

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Log metrics
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse})

        # Log model
        mlflow.xgboost.log_model(model, artifact_path="model")

        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        print("\n✅ Model training complete!")
        print(f"📦 Saved model → models/model.pkl")
        print(f"📊 MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        print(f"🌐 MLflow run logged at: {tracking_uri}")

    return mae, rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost demand model")
    parser.add_argument("--data", default="data/processed/features.csv", help="Path to processed data file")
    parser.add_argument("--params", default="params.yaml", help="Path to parameters YAML file")
    args = parser.parse_args()

    train_model(args.data, args.params)
