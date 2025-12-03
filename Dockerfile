# -----------------------
# Base image
# -----------------------
FROM python:3.10-slim

# Avoid Python buffering
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# -----------------------
# Install OS dependencies
# -----------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Copy requirements
# -----------------------
COPY requirements.txt .

# -----------------------
# Install Python deps
# -----------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------
# Copy entire project
# -----------------------
COPY . .

# -----------------------
# Expose ports
# FastAPI = 8000
# Streamlit = 8501
# -----------------------
EXPOSE 8000
EXPOSE 8501

# -----------------------
# Start both services
# -----------------------
CMD ["bash", "start.sh"]
