# dashboard/app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import requests

from eda import render_eda  # separate EDA module

st.set_page_config(page_title="Pulsar Star Prediction", layout="wide")
st.title("🌠 Pulsar Star Prediction Dashboard")

# ---- Constants ---------------------------------------------------------------
FEATURE_ORDER = [
    "Mean of the integrated profile",
    "Standard deviation of the integrated profile",
    "Excess kurtosis of the integrated profile",
    "Skewness of the integrated profile",
    "Mean of the DM-SNR curve",
    "Standard deviation of the DM-SNR curve",
    "Excess kurtosis of the DM-SNR curve",
    "Skewness of the DM-SNR curve",
]

DEFAULT_API = "http://localhost:8000"

# ---- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("Service Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API,
                            help="Point this to your FastAPI service (e.g., http://localhost:8000)")
    st.session_state["api_url"] = api_url

    st.header("Model Selection")
    selected_model = st.selectbox(
        "Choose model",
        ["lightgbm", "xgboost", "svm", "random_forest", "logreg"],
        index=0
    )

    st.header("Single Prediction Inputs")
    mean = st.number_input("Mean of the integrated profile", value=50.0)
    sd = st.number_input("Standard deviation of the integrated profile", value=35.0)
    kurt = st.number_input("Excess kurtosis of the integrated profile", value=3.0)
    skew = st.number_input("Skewness of the integrated profile", value=0.3)
    mean_dm = st.number_input("Mean of the DM-SNR curve", value=7.0)
    sd_dm = st.number_input("Standard deviation of the DM-SNR curve", value=8.0)
    kurt_dm = st.number_input("Excess kurtosis of the DM-SNR curve", value=1.0)
    skew_dm = st.number_input("Skewness of the DM-SNR curve", value=0.2)
    thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)


# ---- Single Prediction -------------------------------------------------------
st.subheader("Single Prediction")
if st.button("Predict"):
    payload = {
        "features": [mean, sd, kurt, skew, mean_dm, sd_dm, kurt_dm, skew_dm]
    }
    try:
        r = requests.post(
            f"{api_url}/predict_proba?model_name={selected_model}",
            json=payload,
            timeout=30
        )
        r.raise_for_status()
        p = float(r.json()["pulsar_probability"])
        st.metric("Pulsar Probability", f"{p:.3f}")
        st.success("Prediction: Pulsar" if p >= thr else "Prediction: Non-Pulsar")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.markdown("---")


# ---- Batch Scoring + EDA ----------------------------------------------------
st.subheader("Batch Scoring + EDA")

# STEP 1 — Upload file
up = st.file_uploader(
    "Upload CSV (must contain the 8 feature columns in any order; optional `target_class`)",
    type=["csv"],
    key="uploaded_file"
)

# STEP 2 — Store dataframe in session_state so model changes re-run prediction
if up is not None:
    df_uploaded = pd.read_csv(up)
    st.session_state["uploaded_df"] = df_uploaded

# Only proceed if a file was uploaded at least once
if "uploaded_df" in st.session_state:
    try:
        df = st.session_state["uploaded_df"].copy()
        df.columns = df.columns.str.strip()

        # Extract optional label
        has_label = "target_class" in df.columns
        y_series = df["target_class"].copy() if has_label else None

        # Prepare features
        X_df = df.copy()
        if has_label:
            X_df = X_df.drop(columns=["target_class"], errors="ignore")

        # enforce correct order
        X_df = X_df[FEATURE_ORDER]

        # Convert to numeric & impute missing values
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        missing_total = int(X_df.isna().sum().sum())
        if missing_total > 0:
            st.info(f"Found {missing_total} missing values; imputing with column means.")
            X_df = X_df.fillna(X_df.mean(numeric_only=True))

        # STEP 3 — Predict using the selected model
        rows = X_df.values.tolist()
        r = requests.post(
            f"{api_url}/predict_batch?model_name={selected_model}",
            json={"rows": rows},
            timeout=120
        )
        r.raise_for_status()
        probs = r.json()["pulsar_probabilities"]

        # STEP 4 — Build predictions dataframe
        pred_df = X_df.copy()
        pred_df["pulsar_prob"] = probs
        pred_df["prediction"] = (pred_df["pulsar_prob"] >= 0.5).astype(int)
        if has_label:
            pred_df["target_class"] = pd.to_numeric(y_series, errors="coerce")

        # STEP 5 — Render EDA (which now updates instantly!)
        render_eda(df_features=X_df, df_pred=pred_df, has_label=has_label)

        # STEP 6 — Download button
        st.download_button(
            "Download predictions.csv",
            pred_df.to_csv(index=False),
            "predictions.csv"
        )

    except Exception as e:
        st.error(f"Batch processing failed: {e}")
