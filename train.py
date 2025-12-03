# Updated train.py to match tuned models and preprocessing from notebook
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

# Paths and constants
DATA_PATH = Path("data/pulsar_data_train.csv")
MODEL_DIR = Path("models")
FEATURES = [
    "Mean of the integrated profile",
    "Standard deviation of the integrated profile",
    "Excess kurtosis of the integrated profile",
    "Skewness of the integrated profile",
    "Mean of the DM-SNR curve",
    "Standard deviation of the DM-SNR curve",
    "Excess kurtosis of the DM-SNR curve",
    "Skewness of the DM-SNR curve",
]
LABEL = "target_class"
ALL_MODELS = ["logreg", "svm", "decision_tree", "xgboost", "lightgbm"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    print("🔎 CSV Columns:", df.columns.tolist())
    df = df[FEATURES + [LABEL]].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    return df


def clip_outliers_iqr(df):
    for col in FEATURES:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df


def make_estimator(name: str, pos_weight=None):
    if name == "logreg":
        return LogisticRegression(penalty=None, C=1.6238, solver='lbfgs', max_iter=1000, random_state=1)
    elif name == "svm":
        return SVC(kernel='rbf', C=1.0, gamma=0.1, probability=True, random_state=1)
    elif name == "decision_tree":
        return DecisionTreeClassifier(
            criterion='entropy',
            max_depth=5,
            max_features=4,
            min_samples_leaf=525,
            min_samples_split=2593,
            random_state=1,
        )
    elif name == "xgboost":
        if not _HAS_XGB:
            raise ImportError("XGBoost not installed")
        return XGBClassifier(
            n_estimators=1400,
            max_depth=15,
            gamma=0.0947,
            colsample_bytree=1.0,
            subsample=1.0,
            learning_rate=0.05,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=1,
        )
    elif name == "lightgbm":
        if not _HAS_LGBM:
            raise ImportError("LightGBM not installed")
        return LGBMClassifier(
            n_estimators=500,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            objective="binary",
            scale_pos_weight=pos_weight if pos_weight else 1.0,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def build_pipeline(model_name, pos_weight, use_smote=True):
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    preprocessor = SklearnPipeline([
        ("imputer", imputer),
        ("scaler", scaler),
    ])

    ct = ColumnTransformer([
        ("num", preprocessor, list(range(len(FEATURES)))),
    ])

    estimator = make_estimator(model_name, pos_weight=pos_weight)

    steps = [("pre", ct)]

    # SMOTE toggle
    if use_smote:
        steps.append(("smote", SMOTE(random_state=42)))

    steps.append(("clf", estimator))

    return ImbPipeline(steps)


def train_and_save(model_name, df):
    print(f"\n=== Training: {model_name} ===")

    df = clip_outliers_iqr(df)
    X = df[FEATURES].values
    y = df[LABEL].values

    # scale_pos_weight for LGBM/XGB
    pos_weight = (y == 0).sum() / (y == 1).sum() if model_name in {"lightgbm", "xgboost"} else None

    # SPLIT ONCE, reuse for both variants
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    results = {}

    # ----------------------------------------------------
    # 1️⃣ TRAIN NON-SMOTE VERSION
    # ----------------------------------------------------
    print("→ Training NON-SMOTE model")
    model_no = build_pipeline(model_name, pos_weight, use_smote=False)
    model_no.fit(X_train, y_train)

    probs_no = model_no.predict_proba(X_test)[:, 1]
    y_pred_no = (probs_no >= 0.5).astype(int)

    auc_no = roc_auc_score(y_test, probs_no)
    acc_no = accuracy_score(y_test, y_pred_no)
    rec_no = recall_score(y_test, y_pred_no)

    results["auc_before_smote"] = float(auc_no)
    results["accuracy_before_smote"] = float(acc_no)
    results["recall_before_smote"] = float(rec_no)

    # Save
    no_path = MODEL_DIR / f"{model_name}_no_smote.joblib"
    joblib.dump(model_no, no_path)
    print(f"Saved: {no_path}")

    # ----------------------------------------------------
    # 2️⃣ TRAIN SMOTE VERSION
    # ----------------------------------------------------
    print("→ Training SMOTE model")
    model_sm = build_pipeline(model_name, pos_weight, use_smote=True)
    model_sm.fit(X_train, y_train)

    probs_sm = model_sm.predict_proba(X_test)[:, 1]
    y_pred_sm = (probs_sm >= 0.5).astype(int)

    auc_sm = roc_auc_score(y_test, probs_sm)
    acc_sm = accuracy_score(y_test, y_pred_sm)
    rec_sm = recall_score(y_test, y_pred_sm)

    results["auc_after_smote"] = float(auc_sm)
    results["accuracy_after_smote"] = float(acc_sm)
    results["recall_after_smote"] = float(rec_sm)

    # Save
    sm_path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model_sm, sm_path)
    print(f"Saved: {sm_path}")

    # ----------------------------------------------------
    # Save metadata JSON
    # ----------------------------------------------------
    meta_path = MODEL_DIR / f"{model_name}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metadata → {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm",
                        choices=ALL_MODELS + ["all"], help="Model to train")
    args = parser.parse_args()

    df = load_data()

    if args.model == "all":
        for model in ALL_MODELS:
            train_and_save(model, df)
    else:
        train_and_save(args.model, df)


if __name__ == "__main__":
    main()
