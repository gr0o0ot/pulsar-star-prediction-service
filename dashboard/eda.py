# dashboard/eda.py
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import requests
import time

def render_eda(df_features: pd.DataFrame, df_pred: pd.DataFrame, has_label: bool):
    """
    df_features: cleaned features-only DataFrame (8 columns)
    df_pred: df_features + ["pulsar_prob", "prediction"] (+ ["target_class"] if available)
    has_label: whether the uploaded CSV contained a target column
    """
    st.markdown("### Quick EDA")

    tab_overview, tab_hists, tab_corr, tab_threshold, tab_tops, tab_compare = st.tabs(
    ["Overview", "Feature histograms", "Correlations", "Threshold sweep", "Top/Bottom", "Model comparison"]
    )

    # ---------- Overview ----------
    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows (after cleaning)", len(df_features))
        c2.metric("Features", df_features.shape[1])
        c3.metric("Predicted positives @0.5", int((df_pred["pulsar_prob"] >= 0.5).sum()))

        st.write("**Summary (features):**")
        st.dataframe(df_features.describe().T)

        if has_label:
            st.write("**Label balance (uploaded file):**")
            st.table(df_pred["target_class"].value_counts(dropna=False).rename("count").to_frame())

    # ---------- Feature histograms ----------
    with tab_hists:
        feat_list = list(df_features.columns)
        feat = st.selectbox("Choose a feature", feat_list)
        chart = (
            alt.Chart(df_features.reset_index())
            .mark_bar()
            .encode(
                x=alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=40), title=feat),
                y=alt.Y("count()", title="count"),
                tooltip=[alt.Tooltip(f"{feat}:Q", title=feat), alt.Tooltip("count():Q", title="count")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    # ---------- Correlations ----------
    with tab_corr:
        corr = df_features.corr(numeric_only=True)
        corr_long = corr.reset_index().melt("index")
        corr_long.columns = ["feature_x", "feature_y", "corr"]
        heat = (
            alt.Chart(corr_long)
            .mark_rect()
            .encode(
                x="feature_x:O",
                y="feature_y:O",
                color=alt.Color("corr:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])),
                tooltip=["feature_x", "feature_y", alt.Tooltip("corr:Q", format=".2f")],
            )
            .properties(height=400)
        )
        st.altair_chart(heat, use_container_width=True)

    # ---------- Threshold sweep ----------
    with tab_threshold:
        st.caption("How predictions change with threshold")
        thr_min, thr_max = st.slider("Threshold range", 0.0, 1.0, (0.1, 0.9), 0.05)
        thr_values = np.linspace(thr_min, thr_max, num=25)

        summary = []
        y_true = df_pred["target_class"].astype(int).values if has_label else None
        y_prob = df_pred["pulsar_prob"].values

        for t in thr_values:
            y_hat = (y_prob >= t).astype(int)
            pos = int(y_hat.sum())
            row = {"threshold": float(t), "predicted_positives": pos}
            if has_label:
                tp = int(((y_true == 1) & (y_hat == 1)).sum())
                fp = int(((y_true == 0) & (y_hat == 1)).sum())
                fn = int(((y_true == 1) & (y_hat == 0)).sum())
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                row.update({"precision": precision, "recall": recall})
            summary.append(row)

        summary_df = pd.DataFrame(summary)
        c1, c2 = st.columns(2)

        chart_pos = (
            alt.Chart(summary_df)
            .mark_line(point=True)
            .encode(x="threshold:Q", y="predicted_positives:Q",
                    tooltip=["threshold", "predicted_positives"])
            .properties(title="Predicted positives vs threshold", height=300)
        )
        c1.altair_chart(chart_pos, use_container_width=True)

        if has_label:
            chart_pr = (
                alt.Chart(summary_df.melt("threshold", value_vars=["precision", "recall"]))
                .mark_line(point=True)
                .encode(
                    x="threshold:Q",
                    y="value:Q",
                    color="variable:N",
                    tooltip=["threshold", "variable", alt.Tooltip("value:Q", format=".3f")],
                )
                .properties(title="Precision/Recall vs threshold", height=300)
            )
            c2.altair_chart(chart_pr, use_container_width=True)

            # ROC
            try:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)
                roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                roc_chart = (
                    alt.Chart(roc_df)
                    .mark_line()
                    .encode(x="fpr:Q", y="tpr:Q")
                    .properties(title=f"ROC curve (AUC = {roc_auc:.3f})", height=300)
                )
                st.altair_chart(roc_chart, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not compute ROC: {e}")

    # ---------- Top/Bottom ----------
    with tab_tops:
        k = st.slider("How many to show", 5, min(50, len(df_pred)), 10)
        st.write("**Top-k highest probability**")
        st.dataframe(df_pred.sort_values("pulsar_prob", ascending=False).head(k))
        st.write("**Top-k lowest probability**")
        st.dataframe(df_pred.sort_values("pulsar_prob", ascending=True).head(k))
    
        # ---------- MODEL COMPARISON ----------
    with tab_compare:
        st.caption("Compare all trained models on this dataset")

        available_models = ["lightgbm", "xgboost", "svm", "random_forest", "logreg"]
        threshold = st.slider("Threshold for all models", 0.0, 1.0, 0.5, 0.01)

        compare_rows = []
        y_true = df_pred["target_class"].astype(int).values if has_label else None

        for model_name in available_models:
            try:
                t0 = time.time()

                # API call
                r = requests.post(
                    f"{st.session_state['api_url']}/predict_batch?model_name={model_name}",
                    json={"rows": df_features.values.tolist()},
                    timeout=120
                )
                r.raise_for_status()
                probs = np.array(r.json()["pulsar_probabilities"])

                # Predictions
                y_hat = (probs >= threshold).astype(int)

                # Base row
                row = {
                    "model": model_name,
                    "avg_prob": float(probs.mean()),
                    "predicted_positives": int(y_hat.sum()),
                    "runtime_ms": int((time.time() - t0) * 1000)
                }

                # Metrics if label exists
                if has_label:
                    from sklearn.metrics import (
                        accuracy_score, precision_score, recall_score,
                        f1_score, roc_auc_score
                    )

                    row.update({
                        "accuracy": accuracy_score(y_true, y_hat),
                        "precision": precision_score(y_true, y_hat, zero_division=0),
                        "recall": recall_score(y_true, y_hat),
                        "f1_score": f1_score(y_true, y_hat),
                        "auc": roc_auc_score(y_true, probs)
                    })

                compare_rows.append(row)

            except Exception as e:
                compare_rows.append({"model": model_name, "error": str(e)})

            comp_df = pd.DataFrame(compare_rows)

        # Reorder columns nicely
        col_order_with_labels = [
            "model", "auc", "accuracy", "precision", "recall", "f1_score",
            "avg_prob", "predicted_positives", "runtime_ms"
        ]
        col_order_without_labels = [
            "model", "avg_prob", "predicted_positives", "runtime_ms"
        ]

        if has_label:
            comp_df = comp_df[col_order_with_labels]
        else:
            comp_df = comp_df[col_order_without_labels]

        st.dataframe(comp_df, use_container_width=True)

        st.download_button(
            "Download model comparison CSV",
            comp_df.to_csv(index=False),
            "model_comparison.csv"
        )

