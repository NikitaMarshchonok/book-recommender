from pathlib import Path
import pickle, time
from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares

DATA   = Path("data")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)

# ── грузим items×users и ТРАНСПОНИРУЕМ ─────────────────────────────
#   после .T получаем users×items  (53424 × 10000)
user_item = load_npz(DATA / "item_user_mapped.npz").T.tocsr()

print("matrix shape :", user_item.shape)         # (53424, 10000)

# ── ALS ───────────────────────────────────────────────────────────
als = AlternatingLeastSquares(factors=64, iterations=20,
                              regularization=0.1, random_state=42)

t0 = time.perf_counter()
als.fit(user_item)
print(f"⏱ trained in {time.perf_counter()-t0:.1f}s")
print("item_factors.shape:", als.item_factors.shape)   # (10000, 64)  ← проверка

# ── save ──────────────────────────────────────────────────────────
pickle.dump(als, open(MODELS / "als.pkl", "wb"))
print("✅ model saved:", MODELS / "als.pkl")
