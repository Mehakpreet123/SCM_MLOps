# Use official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV PYTHONUNBUFFERED=1

# Default command: run training
CMD ["python", "src/train_model.py", "--data", "data/processed/monthly_data.csv", "--params", "params.yaml"]
