
# Pulsar Star Prediction Service (FastAPI + Streamlit)

This repo contains:
- **FastAPI** inference service (`/api`)
- **Streamlit** dashboard (`/dashboard`)
- **Training script** (`train.py`) that exports a single sklearn Pipeline to `models/pulsar_clf.joblib`

## Quickstart (Windows / macOS / Linux)

```bash
# 1) Create & activate a virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train the model (reads data/pulsar_data_train.csv by default)
python train.py

# 4) Run the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 5) Run the dashboard (in a new terminal)
streamlit run dashboard/app_streamlit.py
```

Open the dashboard at http://localhost:8501 and set the API URL to `http://localhost:8000` in the sidebar.

## Data
Place your training CSV at `data/pulsar_data_train.csv` (HTRU2-style: 8 feature columns + `label`). A small example schema is shown in `train.py`.

## Testing
```bash
python -m pytest -q
```
This runs a **smoke test** that loads the model (after you've run `train.py`) and checks a one-row prediction path.
