
import os
import joblib
import numpy as np

def test_model_artifact_exists():
    assert os.path.exists("models/pulsar_clf.joblib"), "Run train.py to generate the model first."

def test_single_row_prediction_path():
    model = joblib.load("models/pulsar_clf.joblib")
    x = np.array([[50.0, 35.0, 3.0, 0.3, 7.0, 8.0, 1.0, 0.2]], dtype=float)
    proba = float(model.predict_proba(x)[0,1])
    assert 0.0 <= proba <= 1.0
