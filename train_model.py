from pathlib import Path
import pickle, time

from scipy.sparse import load_npz
from implicit.als import AlternatingLeastSquares

DATA   = Path("data")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)

# ─── Загружаем матрицу items × users ───────────────────────────────────
item_user = load_npz(DATA / "item_user_mapped.npz")

print("matrix shape :", item_user.shape)       # (10000, 53424)
print("non-zeros    :", item_user.nnz)

# ─── Обучаем ALS ───────────────────────────────────────────────────────
model = AlternatingLeastSquares(factors=64,
                                iterations=20,
                                regularization=0.1,
                                random_state=42)

t0 = time.perf_counter()
model.fit(item_user)
print(f"⏱  ALS trained in {time.perf_counter() - t0:.1f}s")

# ─── Сохраняем ─────────────────────────────────────────────────────────
pickle.dump(model, open(MODELS / "als.pkl", "wb"))
print("✅ model saved:", MODELS / "als.pkl")
