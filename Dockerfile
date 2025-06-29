
FROM python:3.11-slim

# ---------- system packages (needed for scipy / implicit) -------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ git \
        libopenblas-dev liblapack-dev \
        && rm -rf /var/lib/apt/lists/*

# ---------- working directory in the container ---------------------------
WORKDIR /app

# ---------- copy dependencies and install them --------------------------
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---------- copy the entire application -----------------------------------
COPY . .

# ---------- (optional) recreate data and model inside the image ----------
# If you are loading data/* and models/* with the repo - hide this section
# RUN python prepare_data.py && python train_model.py

# ---------- port that listens to Streamlit ---------------------------
EXPOSE 8501

# ---------- launch command -------------------------------------------
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
