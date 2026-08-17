import joblib
import pandas as pd

# Load the trained model
model = joblib.load("Models/final_lightgbm_model.joblib")

# Load the feature names
feature_names = joblib.load("Models/model_feature_names.joblib")

print("Model loaded successfully")
print(f"Number of features: {len(feature_names)}")
print("Features:")
print(feature_names)

# Create a test transaction with all 29 features
test_data = pd.DataFrame(
    [[
        500.0,   # transaction_amount
        2,       # login_attempts
        0.3,      # device_risk_score
        5,       # transfer_frequency
        0.2,      # anomaly_score
        500,      # account_age_days
        14,       # transaction_time_hour
        1,       # failed_transactions_last_30d
        2500.0,   # avg_monthly_balance
        3,       # daily_transaction_count
        10.0,     # geo_distance_km
        15.0,     # session_duration_minutes
        0.4,      # transaction_velocity_score
        1,       # card_present_flag
        0,       # international_transaction_flag
        0,       # suspicious_ip_flag
        0,       # new_account
        0,       # high_value_transaction
        0,       # high_login_attempts
        0,       # high_velocity
        0,       # night_transaction
        0,       # far_distance_transaction
        0,       # high_device_risk
        1,       # payment_channel_Mobile App
        0,       # payment_channel_POS Terminal
        0,       # payment_channel_Web Banking
        1,       # authentication_type_OTP
        0,       # authentication_type_Password Only
        0        # authentication_type_Two-Factor Authentication
    ]],
    columns=feature_names
)

# Make prediction
prediction = model.predict(test_data)

# Get probability
probability = model.predict_proba(test_data)

print("\nPrediction:", prediction)
print("Probability:", probability)