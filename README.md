<!-- ────────────────────────────────────────────────────────────────────────────── -->
# 📚 Book-Recommender

![render](https://img.shields.io/badge/Render-live-success?logo=render)
![python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![license](https://img.shields.io/badge/License-MIT-informational)

A tiny end-to-end MLOps project:  
**ALS** collaborative-filtering model + **Streamlit** UI + one-click **Docker** deploy on Render.

<br>

<p align="center">
  <a href="https://book-recommender-67zb.onrender.com">
    🔗 **Try the live demo** – book-recommender-67zb.onrender.com
  </a><br>
  <sub>(cold starts take ~30 s on the free tier)</sub>
</p>

---

## ✨ Features
* **Top-N recommendations** for any of the 53 k users in *Goodbooks-10k*  
  (978 k ratings → 2.7 MB sparse matrix).
* **Genre filter** – narrow suggestions down to your favourite tag.
* **“Books like this”** – type a title, get 10 similar books.
* **Dockerfile** &amp; **CI-less** deploy: push to `main` → Render auto-builds and restarts.
* 100 % open source, reproducible in ~10 min on a laptop.

---

## 🚀 Quick start (local)

(bash)
git clone https://github.com/<you>/book-recommender.git
cd book-recommender
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Streamlit + implicit + pandas …
streamlit run app/streamlit_app.py


## Training from scratch

python prepare_data.py    # build item-user matrix   (2–3 s)
python train_model.py     # fit ALS (64 factors)     (5–6 s on CPU)

Outputs go to:
models/als.pkl            – trained model
data/item_user_mapped.npz – 10 k × 53 k CSR matrix
data/id_map.pkl           – { als_idx ➜ real_book_id }


## Project Structure
book-recommender/
├── app/                  ← Streamlit UI (+utils)
├── data/                 ← raw & processed CSV / NPZ
├── models/               ← trained artefacts (.pkl)
├── notebooks/            ← 01_eda.ipynb (exploration)
├── prepare_data.py       ← build sparse matrix + id_map
├── train_model.py        ← train & save ALS
├── Dockerfile            ← slim python:3.11 image
└── requirements.txt



## 🐳 Docker / Render deploy

docker build -t book_rec .
docker run -p 8501:8501 -e PORT=8501 book_rec


##📝 Todo / ideas
Add author search + multi-select genres.
Replace missing covers with Open Library thumbnails.
Export REST endpoint via FastAPI (/recommend?user=1&k=10).
CI (GitHub Actions) to test data prep & training in PRs.






