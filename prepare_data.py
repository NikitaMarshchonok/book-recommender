"""
Создаёт data/item_user_mapped.npz  (items × users)
и data/id_map.pkl                 (idx  → book_id)
Запуск:  python prepare_data.py
"""

from pathlib import Path
import pandas as pd, pickle
from scipy.sparse import coo_matrix, save_npz

DATA = Path("data")
ratings = pd.read_csv(DATA/"ratings.csv")
print("ratings rows:", len(ratings))            # ≈ 981 k

# ── карта индексов ───────────────────────────────────────────────
unique_ids = sorted(ratings.book_id.unique())    # 10 000 id в Goodbooks-10k
rev_map = {bid: idx for idx, bid in enumerate(unique_ids)}   # book_id → idx
id_map  = {idx: bid for bid, idx in rev_map.items()}         # idx → book_id

# ── матрица item × user ─────────────────────────────────────────
row  = ratings.book_id.map(rev_map).values
col  = ratings.user_id.values - 1
data = ratings.rating.astype("float32")

item_user = coo_matrix((data, (row, col))).tocsr()
save_npz(DATA/"item_user_mapped.npz", item_user)
pickle.dump(id_map, open(DATA/"id_map.pkl", "wb"))

print("✅ saved matrix", item_user.shape,
      "size", round((DATA/'item_user_mapped.npz').stat().st_size/1e6, 2), "MB")
