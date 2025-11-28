# ---------- Base Image ----------
FROM python:3.11-slim

# ---------- Set Working Directory ----------
WORKDIR /app

# ---------- Install system dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy files ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ---------- Expose ports ----------
EXPOSE 8000   
EXPOSE 8501   

# ---------- Run both API + Streamlit ----------
CMD ["bash", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run dashboard/app_streamlit.py --server.port=8501 --server.address=0.0.0.0"]
