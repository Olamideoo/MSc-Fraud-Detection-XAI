from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "MODELS" / "final_lightgbm_model.joblib"
FEATURE_NAMES_PATH = BASE_DIR / "MODELS" / "model_feature_names.joblib"

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)


# Create FastAPI application
app = FastAPI( title="Fraud Detection API",
    description="API for Fraud Detection using the Optimized LightGBM Model",
    version="1.0")




# Input schema
class Transaction(BaseModel):
    transaction_amount: float
    login_attempts: int
    device_risk_score: float
    transfer_frequency: float
    anomaly_score: float
    account_age_days: float
    transaction_time_hour: int
    failed_transactions_last_30d: int
    avg_monthly_balance: float
    daily_transaction_count: int
    geo_distance_km: float
    session_duration_minutes: float
    transaction_velocity_score: float
    card_present_flag: int
    international_transaction_flag: int
    suspicious_ip_flag: int
    new_account: int
    high_value_transaction: int
    high_login_attempts: int
    high_velocity: int
    night_transaction: int
    far_distance_transaction: int
    high_device_risk: int

    payment_channel_Mobile_App: int
    payment_channel_POS_Terminal: int
    payment_channel_Web_Banking: int

    authentication_type_OTP: int
    authentication_type_Password_Only: int
    authentication_type_Two_Factor_Authentication: int


# Home endpoint
@app.get("/")
def home():
    return { "message": "Welcome to the Fraud Detection API"}


# Prediction endpoint
@app.post("/predict")
def predict_transaction(transaction: Transaction):

    # Convert input to dictionary
    transaction_data = transaction.model_dump()

    # Rename fields to match the trained model's feature names
    transaction_data = {
        "transaction_amount": transaction_data["transaction_amount"],
        "login_attempts": transaction_data["login_attempts"],
        "device_risk_score": transaction_data["device_risk_score"],
        "transfer_frequency": transaction_data["transfer_frequency"],
        "anomaly_score": transaction_data["anomaly_score"],
        "account_age_days": transaction_data["account_age_days"],
        "transaction_time_hour": transaction_data["transaction_time_hour"],
        "failed_transactions_last_30d": transaction_data["failed_transactions_last_30d"],
        "avg_monthly_balance": transaction_data["avg_monthly_balance"],
        "daily_transaction_count": transaction_data["daily_transaction_count"],
        "geo_distance_km": transaction_data["geo_distance_km"],
        "session_duration_minutes": transaction_data["session_duration_minutes"],
        "transaction_velocity_score": transaction_data["transaction_velocity_score"],
        "card_present_flag": transaction_data["card_present_flag"],
        "international_transaction_flag": transaction_data["international_transaction_flag"],
        "suspicious_ip_flag": transaction_data["suspicious_ip_flag"],
        "new_account": transaction_data["new_account"],
        "high_value_transaction": transaction_data["high_value_transaction"],
        "high_login_attempts": transaction_data["high_login_attempts"],
        "high_velocity": transaction_data["high_velocity"],
        "night_transaction": transaction_data["night_transaction"],
        "far_distance_transaction": transaction_data["far_distance_transaction"],
        "high_device_risk": transaction_data["high_device_risk"],
        "payment_channel_Mobile App": transaction_data["payment_channel_Mobile_App"],
        "payment_channel_POS Terminal": transaction_data["payment_channel_POS_Terminal"],
        "payment_channel_Web Banking": transaction_data["payment_channel_Web_Banking"],
        "authentication_type_OTP": transaction_data["authentication_type_OTP"],
        "authentication_type_Password Only": transaction_data["authentication_type_Password_Only"],
        "authentication_type_Two-Factor Authentication": transaction_data["authentication_type_Two_Factor_Authentication"] }

    # Create DataFrame in the exact feature order used during training
    input_df = pd.DataFrame([transaction_data], columns=feature_names)

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Get probabilities
    probabilities = model.predict_proba(input_df)[0]

    # Fraud probability
    fraud_probability = float(probabilities[1])

    # Convert prediction to readable label
    result = "Fraud" if prediction == 1 else "Legitimate"

    return { "prediction": result,
             "fraud_probability": fraud_probability}