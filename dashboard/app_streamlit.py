# ========== app_streamlit.py ==========
import streamlit as st
import pandas as pd
import numpy as np
import requests

from eda import render_eda
from chatbot import ask_question  # <-- Chatbot integration

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Pulsar Star Prediction", layout="wide")

st.title("🌠 Pulsar Star Prediction Dashboard")

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

ALL_MODELS = [
    "lightgbm", "lightgbm_no_smote",
    "xgboost", "xgboost_no_smote",
    "svm", "svm_no_smote",
    "decision_tree", "decision_tree_no_smote",
    "logreg", "logreg_no_smote"
]

DEFAULT_API = "http://localhost:8000"



# ========== CHATBOT SESSION SETUP ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =================== SIDEBAR SETTINGS ====================
with st.sidebar:
    st.header("⚙️ API Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API)
    st.session_state["api_url"] = api_url

    st.header("🤖 Choose Model")
    selected_model = st.selectbox("Model", ALL_MODELS, index=0)

    st.header("🔮 Single Prediction Input")
    inputs = []
    labels = FEATURE_ORDER
    default_values = [50, 35, 3, 0.3, 7, 8, 1, 0.2]

    for lbl, val in zip(labels, default_values):
        inputs.append(st.number_input(lbl, value=float(val)))

    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)



# =================== SINGLE PREDICTION ====================
st.subheader("✨ Single Prediction")

if st.button("Predict"):
    try:
        res = requests.post(
            f"{api_url}/predict_proba?model_name={selected_model}",
            json={"features": inputs},
            timeout=50
        )
        res.raise_for_status()

        prob = float(res.json()["pulsar_probability"])
        st.metric("Pulsar Probability", f"{prob:.4f}")

        st.success("PULSAR ⭐" if prob >= threshold else "Non-Pulsar")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")



# =================== BATCH SCORING + EDA ====================
st.subheader("📊 Batch Scoring + EDA")

upload = st.file_uploader("Upload CSV", type=["csv"])

if upload is not None:
    df = pd.read_csv(upload)
    st.session_state["uploaded_df"] = df

if "uploaded_df" in st.session_state:
    df = st.session_state["uploaded_df"].copy()
    df.columns = df.columns.str.strip()

    has_label = "target_class" in df.columns
    df_features = df.drop(columns=["target_class"], errors="ignore")

    # Clean + convert
    df_features = df_features[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce")
    df_features = df_features.fillna(df_features.mean())

    rows = df_features.values.tolist()

    try:
        api_response = requests.post(
            f"{api_url}/predict_batch?model_name={selected_model}",
            json={"rows": rows},
            timeout=120
        )
        api_response.raise_for_status()

        probs = api_response.json()["pulsar_probabilities"]

        df_pred = df_features.copy()
        df_pred["pulsar_prob"] = probs
        df_pred["prediction"] = (df_pred["pulsar_prob"] >= 0.5).astype(int)

        if has_label:
            df_pred["target_class"] = df["target_class"]

        # Render EDA
        render_eda(df_features, df_pred, has_label)

        # Download button
        st.download_button(
            "Download predictions.csv",
            df_pred.to_csv(index=False),
            "predictions.csv"
        )

    except Exception as e:
        st.error(f"Backend Error: {e}")

# ======================= FLOATING CHATBOT =======================

# CSS
st.markdown("""
<style>
#chatbot-box {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 320px;
    max-height: 70vh;
    padding: 15px;
    background: rgba(255,255,255,0.95);
    border-radius: 12px;
    border: 1px solid #ccc;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    overflow-y: auto;
    z-index: 99999;
}
#chatbot-header {
    font-weight: bold;
    font-size: 18px;
    margin-bottom: 10px;
}
.chat-user { font-weight: bold; margin-top: 8px; }
.chat-assistant { margin-left: 4px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)



# Show chat history
history_html = ""
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        history_html += f"<div class='chat-user'>You:</div><div>{msg['content']}</div>"
    else:
        history_html += f"<div class='chat-assistant'>Assistant:</div><div>{msg['content']}</div>"

st.markdown(history_html, unsafe_allow_html=True)

# Chat input
query = st.text_input("Ask something:", key="chat_query")

if query:
    answer = ask_question(query)

    # Store messages
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Clear ONLY by popping the key
    st.session_state.pop("chat_query", None)

    # Rerun cleanly
    st.rerun()

# close chatbot box
st.markdown("</div>", unsafe_allow_html=True)

