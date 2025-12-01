#!/bin/bash

echo "Starting FastAPI..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

sleep 2

echo "Starting Streamlit..."
streamlit run dashboard/app_streamlit.py --server.port=8501 --server.address=0.0.0.0
