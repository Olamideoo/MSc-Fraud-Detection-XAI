import streamlit as st
import requests
import joblib
import pandas as pd
import numpy as np
import shap

from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "MODELS" / "final_lightgbm_model.joblib"
FEATURE_NAMES_PATH = BASE_DIR / "MODELS" / "model_feature_names.joblib"
DATA_PATH = BASE_DIR / "Data" / "cleaned_fraud_dataset.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)

    return model, feature_names


model, feature_names = load_model()

# Make sure feature names are stored as a list
feature_names = list(feature_names)


# ============================================================
# CREATE BACKGROUND DATA FOR LIME
# ============================================================

@st.cache_data
def create_lime_background():

    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    data = pd.read_csv(DATA_PATH)


    # --------------------------------------------------------
    # Create engineered features
    # --------------------------------------------------------

    data["new_account"] = (
        data["account_age_days"] < 365
    ).astype(int)

    data["high_value_transaction"] = (
        data["transaction_amount"] > 18619.65
    ).astype(int)

    data["high_login_attempts"] = (
        data["login_attempts"] > 9
    ).astype(int)

    data["high_velocity"] = (
        data["transaction_velocity_score"] > 75.4
    ).astype(int)

    data["night_transaction"] = (
        (data["transaction_time_hour"] >= 22) |
        (data["transaction_time_hour"] <= 5)
    ).astype(int)

    data["far_distance_transaction"] = (
        data["geo_distance_km"] > 11376.25
    ).astype(int)

    data["high_device_risk"] = (
        data["device_risk_score"] > 75.5
    ).astype(int)


    # --------------------------------------------------------
    # Remove target
    # --------------------------------------------------------

    X = data.drop(
        columns=["fraud_flag"],
        errors="ignore"
    )


    # --------------------------------------------------------
    # One-hot encode categorical variables
    # --------------------------------------------------------

    X = pd.get_dummies(
        X,
        columns=[
            "payment_channel",
            "authentication_type"
        ],
        dtype=int
    )


    # --------------------------------------------------------
    # Make feature names match trained model
    # --------------------------------------------------------

    X.columns = (
        X.columns
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )


    # --------------------------------------------------------
    # Make sure background data has exactly the same
    # features and order as the trained model
    # --------------------------------------------------------

    X = X.reindex(
        columns=feature_names,
        fill_value=0
    )

    return X


lime_background = create_lime_background()


# ============================================================
# CREATE LIME EXPLAINER
# ============================================================

@st.cache_resource
def create_lime_explainer(background, feature_names):

    explainer = LimeTabularExplainer(
        training_data=background.values,
        feature_names=feature_names,
        class_names=[
            "Legitimate",
            "Fraud"
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    return explainer


lime_explainer = create_lime_explainer(
    lime_background,
    feature_names
)


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_shap_explainer(_model):

    return shap.TreeExplainer(_model)


shap_explainer = create_shap_explainer(model)


# ============================================================
# TITLE
# ============================================================

st.title("🔍 Fraud Detection System")

st.write(
    "Enter transaction information below to assess the "
    "likelihood of fraudulent activity."
)


# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.header("Transaction Details")

col1, col2 = st.columns(2)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    transaction_amount = st.number_input(
        "Transaction Amount (₦)",
        min_value=0.0,
        value=5000.0,
        step=1000.0
    )

    login_attempts = st.number_input(
        "Login Attempts",
        min_value=0,
        value=2,
        step=1
    )

    device_risk_score = st.number_input(
        "Device Risk Score",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0,
        help="Risk score ranges from 0 to 100."
    )

    transfer_frequency = st.number_input(
        "Transfer Frequency",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

    anomaly_score = st.number_input(
        "Anomaly Score",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )

    account_age_days = st.number_input(
        "Account Age (days)",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

    transaction_time_hour = st.number_input(
        "Transaction Time (hour)",
        min_value=0,
        max_value=23,
        value=14,
        step=1
    )

    failed_transactions_last_30d = st.number_input(
        "Failed Transactions (last 30 days)",
        min_value=0,
        value=1,
        step=1
    )

    avg_monthly_balance = st.number_input(
        "Average Monthly Balance (₦)",
        min_value=0.0,
        value=25000.0,
        step=1000.0
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    daily_transaction_count = st.number_input(
        "Daily Transaction Count",
        min_value=0,
        value=3,
        step=1
    )

    geo_distance_km = st.number_input(
        "Geographic Distance (km)",
        min_value=0.0,
        value=100.0,
        step=100.0
    )

    session_duration_minutes = st.number_input(
        "Session Duration (minutes)",
        min_value=0.0,
        value=15.0,
        step=1.0
    )

    transaction_velocity_score = st.number_input(
        "Transaction Velocity Score",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=1.0,
        help="Velocity score ranges from 0 to 100."
    )

    card_present_flag = st.selectbox(
        "Card Present",
        ["Yes", "No"]
    )

    international_transaction_flag = st.selectbox(
        "International Transaction",
        ["No", "Yes"]
    )

    suspicious_ip_flag = st.selectbox(
        "Suspicious IP",
        ["No", "Yes"]
    )

    new_account = st.selectbox(
        "New Account",
        ["No", "Yes"]
    )


# ============================================================
# PAYMENT AND AUTHENTICATION
# ============================================================

st.header("Payment and Authentication")

col3, col4 = st.columns(2)


with col3:

    payment_channel = st.selectbox(
        "Payment Channel",
        [
            "Mobile App",
            "POS Terminal",
            "Web Banking"
        ]
    )


with col4:

    authentication_type = st.selectbox(
        "Authentication Type",
        [
            "OTP",
            "Password Only",
            "Two-Factor Authentication"
        ]
    )


# ============================================================
# ANALYSE TRANSACTION
# ============================================================

st.divider()

if st.button(
    "🔎 Analyse Transaction",
    type="primary"
):

    # ========================================================
    # CONVERT YES/NO VALUES TO 0/1
    # ========================================================

    card_present = int(
        card_present_flag == "Yes"
    )

    international = int(
        international_transaction_flag == "Yes"
    )

    suspicious_ip = int(
        suspicious_ip_flag == "Yes"
    )

    new_account_value = int(
        new_account == "Yes"
    )


    # ========================================================
    # ENGINEERED FEATURES
    # ========================================================

    high_value_transaction = int(
        transaction_amount > 18619.65
    )

    high_login_attempts = int(
        login_attempts > 9
    )

    high_velocity = int(
        transaction_velocity_score > 75.4
    )

    night_transaction = int(
        transaction_time_hour >= 22
        or transaction_time_hour <= 5
    )

    far_distance_transaction = int(
        geo_distance_km > 11376.25
    )

    high_device_risk = int(
        device_risk_score > 75.5
    )


    # ========================================================
    # ONE-HOT ENCODING
    # ========================================================

    payment_mobile_app = int(
        payment_channel == "Mobile App"
    )

    payment_pos_terminal = int(
        payment_channel == "POS Terminal"
    )

    payment_web_banking = int(
        payment_channel == "Web Banking"
    )

    authentication_otp = int(
        authentication_type == "OTP"
    )

    authentication_password_only = int(
        authentication_type == "Password Only"
    )

    authentication_two_factor = int(
        authentication_type ==
        "Two-Factor Authentication"
    )


    # ========================================================
    # CREATE DATA FOR FASTAPI
    # ========================================================

    transaction_data = {

        "transaction_amount":
            transaction_amount,

        "login_attempts":
            login_attempts,

        "device_risk_score":
            device_risk_score,

        "transfer_frequency":
            transfer_frequency,

        "anomaly_score":
            anomaly_score,

        "account_age_days":
            account_age_days,

        "transaction_time_hour":
            transaction_time_hour,

        "failed_transactions_last_30d":
            failed_transactions_last_30d,

        "avg_monthly_balance":
            avg_monthly_balance,

        "daily_transaction_count":
            daily_transaction_count,

        "geo_distance_km":
            geo_distance_km,

        "session_duration_minutes":
            session_duration_minutes,

        "transaction_velocity_score":
            transaction_velocity_score,

        "card_present_flag":
            card_present,

        "international_transaction_flag":
            international,

        "suspicious_ip_flag":
            suspicious_ip,

        "new_account":
            new_account_value,

        "high_value_transaction":
            high_value_transaction,

        "high_login_attempts":
            high_login_attempts,

        "high_velocity":
            high_velocity,

        "night_transaction":
            night_transaction,

        "far_distance_transaction":
            far_distance_transaction,

        "high_device_risk":
            high_device_risk,

        "payment_channel_Mobile_App":
            payment_mobile_app,

        "payment_channel_POS_Terminal":
            payment_pos_terminal,

        "payment_channel_Web_Banking":
            payment_web_banking,

        "authentication_type_OTP":
            authentication_otp,

        "authentication_type_Password_Only":
            authentication_password_only,

        "authentication_type_Two_Factor_Authentication":
            authentication_two_factor
    }


    # ========================================================
    # CREATE DATAFRAME FOR SHAP
    # ========================================================

    input_df = pd.DataFrame(
        [transaction_data]
    )


    # ========================================================
    # FORCE EXACT MODEL FEATURE ORDER
    # ========================================================

    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )


    # ========================================================
    # VIEW FEATURES SENT TO API
    # ========================================================

    with st.expander(
        "🔧 View Features Sent to API"
    ):

        st.json(transaction_data)

        st.write(
            "Number of API features:",
            len(transaction_data)
        )

        st.write(
            "Number of model features:",
            len(feature_names)
        )


    # ========================================================
    # SEND DATA TO FASTAPI
    # ========================================================

    try:

        response = requests.post(
            "https://msc-fraud-detection-xai.onrender.com/predict",
            json=transaction_data,
            timeout=30
        )


        # ====================================================
        # SUCCESSFUL RESPONSE
        # ====================================================

        if response.status_code == 200:

            result = response.json()

            st.success(
                "Transaction analysed successfully."
            )


            # =================================================
            # GET MODEL RESULT
            # =================================================

            fraud_probability = float(
                result["fraud_probability"]
            )

            prediction = result["prediction"]


            # =================================================
            # RISK LEVEL
            #
            # IMPORTANT:
            # This does NOT change the model prediction.
            # It is only a dashboard interpretation of
            # the returned fraud probability.
            # =================================================

            if fraud_probability >= 0.70:

                risk_level = "High"

            elif fraud_probability >= 0.40:

                risk_level = "Elevated"

            elif fraud_probability >= 0.20:

                risk_level = "Moderate"

            else:

                risk_level = "Low"


            # =================================================
            # MODEL ASSESSMENT
            # =================================================

            st.subheader(
                "📋 Model Assessment"
            )

            result_col1, result_col2, result_col3 = st.columns(3)


            with result_col1:

                st.metric(
                    "Model Classification",
                    prediction
                )


            with result_col2:

                st.metric(
                    "Fraud Probability",
                    f"{fraud_probability * 100:.2f}%"
                )


            with result_col3:

                st.metric(
                    "Risk Level",
                    risk_level
                )


            # =================================================
            # INTERPRETATION OF RESULT
            # =================================================

            if prediction == "Fraud":

                st.error(
                    "⚠️ FRAUD DETECTED"
                )

            else:

                st.success(
                    "✅ MODEL CLASSIFICATION: LEGITIMATE"
                )


            # =================================================
            # RISK MESSAGE
            # =================================================

            if risk_level == "High":

                st.warning(
                    "The model's fraud probability is high. "
                    "Although the classification is determined "
                    "by the model, this transaction represents "
                    "a high-risk case according to the dashboard "
                    "risk bands."
                )

            elif risk_level == "Elevated":

                st.warning(
                    "The model classified this transaction as "
                    f"{prediction.lower()}, but the predicted "
                    "fraud probability indicates an elevated "
                    "level of risk."
                )

            elif risk_level == "Moderate":

                st.info(
                    "The transaction has a moderate predicted "
                    "fraud risk, although the model classification "
                    f"is {prediction.lower()}."
                )

            else:

                st.info(
                    "The model indicates a relatively low "
                    "fraud probability for this transaction."
                )


            # =================================================
            # PROBABILITY BAR
            # =================================================

            st.progress(
                min(
                    max(
                        fraud_probability,
                        0.0
                    ),
                    1.0
                )
            )


            # =================================================
            # WHY DID THE MODEL MAKE THIS PREDICTION?
            # =================================================

            st.subheader(
                "🔍 Why did the model make this prediction?"
            )


            if prediction == "Fraud":

                st.warning(
                    "The model classified this transaction as "
                    "fraudulent because the combination of the "
                    "submitted transaction characteristics "
                    "produced a high predicted fraud probability. "
                    "The explanations below identify the features "
                    "that contributed most strongly to this "
                    "individual prediction."
                )

            else:

                st.info(
                    "The model classified this transaction as "
                    "legitimate. The explanations below identify "
                    "the features that contributed most strongly "
                    "to this individual prediction."
                )


            # =================================================
            # SHAP EXPLANATION
            # =================================================

            st.header(
                "📊 SHAP Explanation"
            )

            st.write(
                "SHAP explains how individual features influenced "
                "this specific model prediction. Positive SHAP "
                "values push the model towards the Fraud class, "
                "while negative SHAP values push the model away "
                "from the Fraud class."
            )

            st.caption(
                "Note: SHAP values represent model contribution "
                "and are not percentages or probabilities."
            )


            # =================================================
            # CALCULATE SHAP VALUES
            # =================================================

            shap_output = shap_explainer(
                input_df
            )

            shap_values = shap_output.values


            # -------------------------------------------------
            # Handle SHAP output dimensions
            # -------------------------------------------------

            if shap_values.ndim == 3:

                shap_values = (
                    shap_values[0, :, 1]
                )

            elif shap_values.ndim == 2:

                shap_values = (
                    shap_values[0]
                )

            else:

                shap_values = np.asarray(
                    shap_values
                ).flatten()


            # =================================================
            # CREATE SHAP DATAFRAME
            # =================================================

            shap_df = pd.DataFrame({

                "Feature":
                    feature_names,

                "Value":
                    input_df.iloc[0].values,

                "SHAP Value":
                    shap_values
            })


            # =================================================
            # DETERMINE SHAP IMPACT
            # =================================================

            shap_df["Impact"] = np.where(
                shap_df["SHAP Value"] > 0,
                "Increases fraud risk",
                "Decreases fraud risk"
            )


            # =================================================
            # ABSOLUTE IMPACT
            # =================================================

            shap_df["Absolute Impact"] = (
                shap_df["SHAP Value"].abs()
            )


            # =================================================
            # SORT FEATURES
            # =================================================

            shap_df = shap_df.sort_values(
                "Absolute Impact",
                ascending=False
            )


            # =================================================
            # TOP SHAP FEATURES
            # =================================================

            top_shap = (
                shap_df
                .head(10)
                .copy()
            )


            st.subheader(
                "Top features influencing this prediction"
            )


            # =================================================
            # SHAP CHART
            # =================================================

            st.bar_chart(
                top_shap.set_index(
                    "Feature"
                )["SHAP Value"]
            )


            # =================================================
            # SHAP TABLE
            # =================================================

            st.dataframe(
                top_shap[
                    [
                        "Feature",
                        "Value",
                        "SHAP Value",
                        "Impact"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # HUMAN-READABLE SHAP SUMMARY
            # =================================================

            st.subheader(
                "🧠 Main factors influencing the prediction"
            )


            positive_features = top_shap[
                top_shap["SHAP Value"] > 0
            ]

            negative_features = top_shap[
                top_shap["SHAP Value"] < 0
            ]


            # =================================================
            # FEATURES INCREASING FRAUD RISK
            # =================================================

            if len(positive_features) > 0:

                st.write(
                    "**Factors increasing fraud risk:**"
                )

                for _, row in positive_features.head(5).iterrows():

                    st.write(
                        f"🔴 **{row['Feature']}** "
                        f"(value: {row['Value']}) — "
                        f"contributed towards a higher "
                        f"fraud prediction."
                    )


            # =================================================
            # FEATURES REDUCING FRAUD RISK
            # =================================================

            if len(negative_features) > 0:

                st.write(
                    "**Factors reducing fraud risk:**"
                )

                for _, row in negative_features.head(5).iterrows():

                    st.write(
                        f"🟢 **{row['Feature']}** "
                        f"(value: {row['Value']}) — "
                        f"contributed towards a lower "
                        f"fraud prediction."
                    )


            # =================================================
            # LIME EXPLANATION
            # =================================================

            st.header(
                "🔬 LIME Explanation"
            )

            st.write(
                "LIME provides a second local explanation by "
                "approximating the model's behaviour around this "
                "particular transaction. This provides a "
                "complementary perspective to SHAP."
            )


            # =================================================
            # LIME PREDICTION FUNCTION
            # =================================================

            def lime_predict(input_array):

                lime_df = pd.DataFrame(
                    input_array,
                    columns=feature_names
                )

                return model.predict_proba(
                    lime_df
                )


            # =================================================
            # GENERATE LIME EXPLANATION
            # =================================================

            lime_explanation = (
                lime_explainer.explain_instance(
                    input_df.iloc[0].values,
                    lime_predict,
                    num_features=10
                )
            )


            # =================================================
            # CONVERT LIME EXPLANATION TO DATAFRAME
            # =================================================

            lime_list = (
                lime_explanation.as_list()
            )

            lime_df = pd.DataFrame(
                lime_list,
                columns=[
                    "Feature",
                    "LIME Weight"
                ]
            )


            # =================================================
            # DETERMINE LIME IMPACT
            # =================================================

            lime_df["Impact"] = np.where(
                lime_df["LIME Weight"] > 0,
                "Supports Fraud",
                "Supports Legitimate"
            )


            # =================================================
            # DISPLAY LIME RESULTS
            # =================================================

            st.dataframe(
                lime_df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # EXPLANATION SUMMARY
            # =================================================

            st.header(
                "📋 Explanation Summary"
            )

            st.write(
                "SHAP and LIME provide complementary local "
                "explanations of the model's decision."
            )

            st.write(
                "**SHAP** identifies how individual features "
                "contribute to the prediction and shows whether "
                "each feature pushes the model towards or away "
                "from the Fraud class."
            )

            st.write(
                "**LIME** approximates the behaviour of the "
                "machine-learning model around the submitted "
                "transaction and identifies local conditions "
                "that support either the Fraud or Legitimate "
                "class."
            )


            # =================================================
            # PLAIN-ENGLISH INTERPRETATION
            # =================================================

            st.subheader(
                "💡 What does this mean?"
            )

            st.write(
                f"The model assigned this transaction a fraud "
                f"probability of "
                f"**{fraud_probability * 100:.2f}%**. "
                f"The SHAP and LIME explanations show the "
                f"features that most influenced this decision."
            )


            # =================================================
            # IMPORTANT DISCLAIMER
            # =================================================

            st.info(
                "⚠️ These explanations describe the factors "
                "influencing the machine-learning model's "
                "prediction. They do not prove that a transaction "
                "is fraudulent or legitimate. SHAP and LIME should "
                "therefore be interpreted as model explanations "
                "rather than definitive evidence."
            )


        # ====================================================
        # API ERROR
        # ====================================================

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            try:

                error_details = response.json()

                st.json(error_details)

            except ValueError:

                st.write(response.text)


    # ========================================================
    # CONNECTION ERROR
    # ========================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the Fraud Detection API. "
            "Please check that the deployed Fraud Detection "
            "API is available."
        )


    # ========================================================
    # TIMEOUT ERROR
    # ========================================================

    except requests.exceptions.Timeout:

        st.error(
            "The request to the Fraud Detection API timed out. "
            "Please make sure FastAPI is running correctly."
        )


    # ========================================================
    # OTHER REQUEST ERRORS
    # ========================================================

    except requests.exceptions.RequestException as e:

        st.error(
            f"Request failed: {e}"
        )


    # ========================================================
    # XAI / GENERAL ERROR
    # ========================================================

    except Exception as e:

        st.error(
            "An error occurred while generating the "
            f"prediction or explanation: {e}"
        )