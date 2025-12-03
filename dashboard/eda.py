# dashboard/eda.py
import json
from pathlib import Path
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

    # TABS ---------------------------------------------------------------------
    (
        tab_overview,
        tab_hists,
        tab_corr,
        tab_threshold,
        tab_tops,
        tab_compare,
        tab_smote,
    ) = st.tabs(
        [
            "Overview",
            "Feature histograms",
            "Correlations",
            "Threshold sweep",
            "Top/Bottom",
            "Model comparison",
            "SMOTE analysis",
        ]
    )

    # -------------------------------------------------------------------------
    # OVERVIEW TAB
    # -------------------------------------------------------------------------
    with tab_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows (after cleaning)", len(df_features))
        c2.metric("Features", df_features.shape[1])
        c3.metric("Predicted positives @0.5", int((df_pred["pulsar_prob"] >= 0.5).sum()))

        st.subheader("Summary (features)")
        st.dataframe(df_features.describe().T)

        if has_label:
            st.subheader("Label distribution (uploaded file)")
            st.table(
                df_pred["target_class"]
                .value_counts(dropna=False)
                .rename("count")
                .to_frame()
            )

    # -------------------------------------------------------------------------
    # FEATURE HISTOGRAMS TAB
    # -------------------------------------------------------------------------
    with tab_hists:
        feat_list = list(df_features.columns)
        feat = st.selectbox("Choose a feature", feat_list)

        chart = (
            alt.Chart(df_features.reset_index())
            .mark_bar()
            .encode(
                x=alt.X(f"{feat}:Q", bin=alt.Bin(maxbins=40)),
                y="count()",
                tooltip=[feat, "count()"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    # -------------------------------------------------------------------------
    # CORRELATION TAB
    # -------------------------------------------------------------------------
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
                tooltip=["feature_x", "feature_y", "corr"],
            )
            .properties(height=400)
        )
        st.altair_chart(heat, use_container_width=True)

    # -------------------------------------------------------------------------
    # THRESHOLD SWEEP TAB
    # -------------------------------------------------------------------------
    with tab_threshold:
        st.caption("How predictions change with threshold")

        thr_min, thr_max = st.slider("Threshold range", 0.0, 1.0, (0.1, 0.9), 0.05)
        thr_values = np.linspace(thr_min, thr_max, 25)

        y_prob = df_pred["pulsar_prob"].values
        y_true = df_pred["target_class"].astype(int).values if has_label else None

        rows = []
        for t in thr_values:
            y_hat = (y_prob >= t).astype(int)
            row = {"threshold": float(t), "predicted_positives": int(y_hat.sum())}

            if has_label:
                tp = int(((y_true == 1) & (y_hat == 1)).sum())
                fp = int(((y_true == 0) & (y_hat == 1)).sum())
                fn = int(((y_true == 1) & (y_hat == 0)).sum())

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                row.update({"precision": precision, "recall": recall})

            rows.append(row)

        df_thr = pd.DataFrame(rows)

        c1, c2 = st.columns(2)

        # Positives vs threshold
        c1.altair_chart(
            alt.Chart(df_thr)
            .mark_line(point=True)
            .encode(x="threshold:Q", y="predicted_positives:Q")
            .properties(height=300),
            use_container_width=True,
        )

        # Precision/Recall curves
        if has_label:
            melted = df_thr.melt("threshold", value_vars=["precision", "recall"])
            c2.altair_chart(
                alt.Chart(melted)
                .mark_line(point=True)
                .encode(
                    x="threshold:Q",
                    y="value:Q",
                    color="variable:N",
                )
                .properties(height=300),
                use_container_width=True,
            )

    # -------------------------------------------------------------------------
    # TOP / BOTTOM TAB
    # -------------------------------------------------------------------------
    with tab_tops:
        k = st.slider("How many to show", 5, min(len(df_pred), 50), 10)

        st.subheader("Top-k highest probability")
        st.dataframe(df_pred.sort_values("pulsar_prob", ascending=False).head(k))

        st.subheader("Top-k lowest probability")
        st.dataframe(df_pred.sort_values("pulsar_prob", ascending=True).head(k))

    # -------------------------------------------------------------------------
    # MODEL COMPARISON TAB
    # -------------------------------------------------------------------------
    with tab_compare:
        st.caption("Compare all trained models on this dataset")

        available_models = ["lightgbm", "xgboost", "svm", "decision_tree", "logreg"]
        threshold = st.slider("Threshold for all models:", 0.0, 1.0, 0.5, 0.01)

        compare_rows = []
        y_true = df_pred["target_class"].astype(int).values if has_label else None

        for model_name in available_models:
            try:
                t0 = time.time()

                r = requests.post(
                    f"{st.session_state['api_url']}/predict_batch?model_name={model_name}",
                    json={"rows": df_features.values.tolist()},
                    timeout=120,
                )
                r.raise_for_status()

                probs = np.array(r.json()["pulsar_probabilities"])
                y_hat = (probs >= threshold).astype(int)

                row = {
                    "model": model_name,
                    "avg_prob": float(probs.mean()),
                    "predicted_positives": int(y_hat.sum()),
                    "runtime_ms": int((time.time() - t0) * 1000),
                }

                if has_label:
                    from sklearn.metrics import (
                        accuracy_score,
                        precision_score,
                        recall_score,
                        f1_score,
                        roc_auc_score,
                    )

                    row.update(
                        {
                            "accuracy": accuracy_score(y_true, y_hat),
                            "precision": precision_score(y_true, y_hat, zero_division=0),
                            "recall": recall_score(y_true, y_hat),
                            "f1_score": f1_score(y_true, y_hat),
                            "auc": roc_auc_score(y_true, probs),
                        }
                    )

                compare_rows.append(row)

            except Exception as e:
                compare_rows.append({"model": model_name, "error": str(e)})

        comp_df = pd.DataFrame(compare_rows)

        st.dataframe(comp_df, use_container_width=True)

        st.download_button(
            "Download model comparison CSV",
            comp_df.to_csv(index=False),
            "model_comparison.csv",
        )

    # -------------------------------------------------------------------------
    # SMOTE ANALYSIS TAB  (AUC + RECALL + ACCURACY)
    # -------------------------------------------------------------------------
    with tab_smote:
        st.subheader("SMOTE Performance Analysis")

        model_list = ["lightgbm", "xgboost", "svm", "decision_tree", "logreg"]
        results = []

        for model_name in model_list:
            meta_path = Path("models") / f"{model_name}.meta.json"
            if meta_path.exists():
                meta = json.load(open(meta_path))

                auc_before = meta.get("auc_before_smote")
                auc_after = meta.get("auc_after_smote")

                recall_before = meta.get("recall_before_smote")
                recall_after = meta.get("recall_after_smote")

                acc_before = meta.get("accuracy_before_smote")
                acc_after = meta.get("accuracy_after_smote")

                results.append(
                    {
                        "model": model_name,
                        "auc_before_smote": auc_before,
                        "auc_after_smote": auc_after,
                        "auc_improvement": (auc_after - auc_before)
                        if (auc_before is not None and auc_after is not None)
                        else None,
                        "recall_before_smote": recall_before,
                        "recall_after_smote": recall_after,
                        "recall_improvement": (recall_after - recall_before)
                        if (recall_before is not None and recall_after is not None)
                        else None,
                        "accuracy_before_smote": acc_before,
                        "accuracy_after_smote": acc_after,
                        "accuracy_improvement": (acc_after - acc_before)
                        if (acc_before is not None and acc_after is not None)
                        else None,
                    }
                )

        if len(results) == 0:
            st.warning("No SMOTE metadata found — did you run train.py recently?")
            return

        df_smote = pd.DataFrame(results)

        # ---------- AUC ----------
        st.write("### AUC Before vs After SMOTE")
        df_auc = df_smote[["model", "auc_before_smote", "auc_after_smote", "auc_improvement"]]
        st.dataframe(df_auc)

        st.write("### SMOTE AUC Improvement Chart")
        chart_auc = (
            alt.Chart(df_auc)
            .mark_bar()
            .encode(
                x="model:N",
                y="auc_improvement:Q",
                color="model:N",
                tooltip=["model", "auc_before_smote", "auc_after_smote", "auc_improvement"],
            )
        )
        st.altair_chart(chart_auc, use_container_width=True)

        # ---------- RECALL ----------
        st.write("### Recall Before vs After SMOTE")
        df_rec = df_smote[
            ["model", "recall_before_smote", "recall_after_smote", "recall_improvement"]
        ]
        st.dataframe(df_rec)

        st.write("### SMOTE Recall Improvement Chart")
        chart_rec = (
            alt.Chart(df_rec)
            .mark_bar()
            .encode(
                x="model:N",
                y="recall_improvement:Q",
                color="model:N",
                tooltip=[
                    "model",
                    "recall_before_smote",
                    "recall_after_smote",
                    "recall_improvement",
                ],
            )
        )
        st.altair_chart(chart_rec, use_container_width=True)

        # ---------- ACCURACY ----------
        st.write("### Accuracy Before vs After SMOTE")
        df_acc = df_smote[
            ["model", "accuracy_before_smote", "accuracy_after_smote", "accuracy_improvement"]
        ]
        st.dataframe(df_acc)

        st.write("### SMOTE Accuracy Improvement Chart")
        chart_acc = (
            alt.Chart(df_acc)
            .mark_bar()
            .encode(
                x="model:N",
                y="accuracy_improvement:Q",
                color="model:N",
                tooltip=[
                    "model",
                    "accuracy_before_smote",
                    "accuracy_after_smote",
                    "accuracy_improvement",
                ],
            )
        )
        st.altair_chart(chart_acc, use_container_width=True)
