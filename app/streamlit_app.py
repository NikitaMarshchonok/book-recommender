import streamlit as st, pandas as pd, pickle
from scipy.sparse import load_npz
from pathlib import Path

BASE  = Path(__file__).resolve().parent.parent
DATA  = BASE/"data"
MODELS= BASE/"models"

@st.cache_resource
def load_artifacts():
    books = pd.read_csv(DATA/"books.csv").set_index("book_id")
    id_map = pickle.load(open(DATA/"id_map.pkl","rb"))      # idx → book_id
    idx2book = pd.Series(id_map)
    model = pickle.load(open(MODELS/"als.pkl","rb"))
    item_user = load_npz(DATA/"item_user_mapped.npz").tocsr()
    user_item = item_user.T.tocsr()                         # users × items CSR
    return books, idx2book, model, user_item

books, idx2book, model, user_item = load_artifacts()

st.title("📚 Book Recommender")
uid_max = user_item.shape[0]
uid = st.number_input(f"User ID (1 — {uid_max}):", 1, uid_max, 1)

if st.button("Recommend"):
    hist = user_item[uid-1]                                 # 1×items CSR
    item_idx, _ = model.recommend(uid-1, hist, N=100)       # топ-100 индексов
    book_ids = [idx2book[i] for i in item_idx if idx2book[i] in books.index][:10]

    if not book_ids:
        st.warning("Для этого пользователя нет рекомендаций в каталоге книг.")
    else:
        for _, row in books.loc[book_ids][["title","authors","image_url"]].iterrows():
            img = row.image_url if isinstance(row.image_url, str) and row.image_url \
                  else "https://via.placeholder.com/90x135?text=No+Cover"
            st.image(img, width=90)
            st.write(f"**{row.title}** — {row.authors}")
