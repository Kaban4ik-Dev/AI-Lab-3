"""Пример применения сохранённой модели части Б.

Загружает preprocessor.joblib и best_solution.joblib,
применяет преобразования к новым строкам RFM и предсказывает кластер.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json

OUT = Path(__file__).resolve().parent.parent
prep = joblib.load(OUT / "outputs_b" / "preprocessor.joblib")
best = joblib.load(OUT / "outputs_b" / "best_solution.joblib")

space = best["space"]          # "rfm" или "ext"
method = best["method"]
model = best["model"]          # обученная модель

cols = prep["cols_rfm"] if space == "rfm" else prep["cols_ext"]
sc = prep["sc_rfm"] if space == "rfm" else prep["sc_ext"]
clip = prep["clip_bounds_rfm"] if space == "rfm" else prep["clip_bounds_ext"]
LOG_COLS = ("Frequency", "Monetary", "avg_basket", "n_products")

# новый клиент (все поля, как в rfm — лишние игнорируются)
new_df = pd.DataFrame([{
    "Recency": 10, "Frequency": 5, "Monetary": 1200.0,
    "n_products": 40, "avg_basket": 240.0,
}])

X = new_df[cols].astype(float).copy()
# winsorize по сохранённым порогам (а не по самой строке!)
for c in cols:
    X[c] = np.minimum(X[c], clip[c])
# log1p
for c in cols:
    if c in LOG_COLS:
        X[c] = np.log1p(X[c])
# масштабирование
Z = sc.transform(X)

prof_path = OUT / "outputs_b" / "segment_profile.csv"
cluster_id = int(model.predict(Z)[0]) if hasattr(model, "predict") else None

if cluster_id is not None and prof_path.exists():
    prof = pd.read_csv(prof_path)
    row = prof[prof["cluster"] == cluster_id]
    if len(row):
        print("Профиль сегмента:")
        print(row.to_string(index=False))

print("Модель:", method, best["params"])
print("Признаки:", cols)
if cluster_id is not None:
    print("Кластер:", cluster_id)
else:
    print("Метод без predict (agglomerative/dbscan) — "
          "используйте переобучение на полном наборе.")