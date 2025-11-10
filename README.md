# 📈 Monthly Demand Forecasting

![Python](https://img.shields.io/badge/Python-3.11-blue)
![DVC](https://img.shields.io/badge/DVC-pipeline-orange)
![MLflow](https://img.shields.io/badge/MLflow-experiments-green)
![AWS S3](https://img.shields.io/badge/S3-storage-yellow)

This project implements a **monthly demand forecasting pipeline** using Python, **DVC**, **MLflow**, and **AWS S3**. The dataset is fetched from Kaggle and versioned using DVC, while MLflow tracks experiments.

---

## 🚀 Features

- Fetch retail dataset from Kaggle
- Preprocess and aggregate monthly demand
- Generate lag features for forecasting
- Train and evaluate ML models
- Experiment tracking with MLflow
- Data versioning with DVC + S3

---

## 🗂 Project Structure



SCM_MLOps/
├── data/
│ ├── raw/ # Raw data ingested from Kaggle
│ └── processed/ # Preprocessed and feature-engineered data
├── models/
│ └── model.pkl # Trained model file
├── src/
│ ├── data_ingestion.py # Script to download dataset from Kaggle
│ ├── data_preprocessing.py # Script to clean and aggregate data
│ ├── feature.py # Script to create lag features
│ ├── train_model.py # Script to train the model
│ └── evaluate.py # Script to evaluate the model
├── params.yaml # Model and training parameters
├── metrics.json # Evaluation metrics
├── .dvc/ # DVC metadata
├── .dvcignore
├── requirements.txt
├── .env # Environment variables
└── README.md

---

## Setup Instructions

### 1. Create Python Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate # On Mac/Linux
pip install -r requirements.txt

git init
git add .
git commit -m "Initial commit - project setup and requirements installed"

dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC for data versioning"

git rm -r --cached 'data/processed/monthly_data.csv'
git commit -m "stop tracking data/processed/monthly_data.csv"
git add 'data/processed/monthly_data.csv.dvc' 'data/processed/.gitignore'

aws configure
# Enter Access Key, Secret Key, Region (e.g., ap-south-1), and output format (json)
dvc remote add -d s3remote s3://scm-mlops-data
dvc push

dvc stage add -n data_ingestion \
  -d src/data_ingestion.py \
  -o data/raw \
  python src/data_ingestion.py \
  --dataset sahilprajapati143/retail-analysis-large-dataset \
  --output_dir data/raw

dvc stage add -n data_preprocessing \
  -d src/data_preprocessing.py \
  -d data/raw \
  -o data/processed/monthly_data.csv \
  python src/data_preprocessing.py \
  --in_dir data/raw \
  --out_file data/processed/monthly_data.csv

dvc stage add -n feature_engineering \
  -d src/feature.py \
  -d data/processed/monthly_data.csv \
  -o data/processed/monthly_data_lags.csv \
  python src/feature.py \
  --input_file data/processed/monthly_data.csv \
  --output_file data/processed/monthly_data_lags.csv \
  --lags 1 2 3

dvc stage add -n train_model \
  -d src/train_model.py \
  -d data/processed/monthly_data_lags.csv \
  -d params.yaml \
  -o models/model.pkl \
  python src/train_model.py \
  --data data/processed/monthly_data_lags.csv \
  --params params.yaml

dvc stage add -n evaluate_model \
  -d src/evaluate.py \
  -d models/model.pkl \
  -d data/processed/monthly_data.csv \
  -o metrics.json \
  python src/evaluate.py \
  --model models/model.pkl \
  --data data/processed/monthly_data.csv

mlflow ui


