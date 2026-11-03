"""
Пример предсказания для части А (Bank Marketing).

Загружает сохранённые артефакты из outputs_a/:
  - own_logreg_final.joblib   (собственная логистическая регрессия)
  - logreg_final.joblib        (sklearn LogisticRegression, Pipeline)
  - nn_preprocessor.joblib     (ColumnTransformer для NN)
  - nn_weights.pt              (веса MLP по 3 seed'ам)

Берёт первые 5 объектов из test.csv, считает вероятность отклика
тремя моделями, печатает таблицу и сохраняет prediction_example.csv.

Запуск:  python predict_example_a.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from config import OUT_DIR
from data_utils import column_types
from model_nn import MLP, predict_scores_nn


# ---------------------------------------------------------------- helpers
def load_nn_best_cfg() -> dict:
    """
    Достаём лучшую NN-конфигурацию из runs.csv (stage='search', model='nn',
    максимальный mean). Так не хардкодим hidden=[512,256] вручную.
    Если runs.csv нет — fallback на известную лучшую конфигурацию.
    """
    runs_path = OUT_DIR / "runs.csv"
    fallback = {"hidden": [512, 256], "dropout": 0.3,
                "lr": 5e-4, "batch": 128, "wd": 1e-4}
    if not runs_path.exists():
        print("[warn] runs.csv не найден — использую fallback NN-конфигурацию")
        return fallback
    df = pd.read_csv(runs_path)
    nn = df[(df["model"] == "nn") & (df["stage"] == "search")].copy()
    if nn.empty or "params" not in nn.columns or "mean" not in nn.columns:
        print("[warn] в runs.csv нет записей NN-search — fallback")
        return fallback
    nn["mean"] = pd.to_numeric(nn["mean"], errors="coerce")
    nn = nn.dropna(subset=["mean"]).sort_values("mean", ascending=False)
    if nn.empty:
        return fallback
    try:
        cfg = json.loads(nn.iloc[0]["params"])
    except Exception:
        print("[warn] не удалось распарсить params лучшей NN — fallback")
        return fallback
    # гарантируем нужные ключи
    for k, v in fallback.items():
        cfg.setdefault(k, v)
    return cfg


def predict_own_logreg(sample_df: pd.DataFrame, feats: list) -> np.ndarray:
    """Собственная логистическая регрессия: joblib-словарь {type, pre, model}."""
    obj = joblib.load(OUT_DIR / "own_logreg_final.joblib")
    pre = obj["pre"]
    model = obj["model"]
    X = pre.transform(sample_df[feats])
    return model.predict_proba(X)[:, 1]


def predict_logreg(sample_df: pd.DataFrame, feats: list) -> np.ndarray:
    """sklearn LogisticRegression в Pipeline: predict_proba сразу."""
    pipe = joblib.load(OUT_DIR / "logreg_final.joblib")
    return pipe.predict_proba(sample_df[feats])[:, 1]


def predict_nn(sample_df: pd.DataFrame, feats: list, cfg: dict,
               device: str = "cpu") -> np.ndarray:
    """
    NN: препроцессор + усреднение по всем сохранённым seed'ам
    (как в run_part_a.py на тесте).
    """
    pre_nn = joblib.load(OUT_DIR / "nn_preprocessor.joblib")
    states = torch.load(OUT_DIR / "nn_weights.pt", map_location="cpu")
    X = pre_nn.transform(sample_df[feats])

    preds = []
    for s in states:
        model = MLP(X.shape[1], cfg["hidden"], cfg.get("dropout", 0.0)).to(device)
        model.load_state_dict(s["state"])
        model.eval()
        preds.append(predict_scores_nn(model, X, device=device))
    return np.mean(preds, axis=0)


# ---------------------------------------------------------------- main
def main():
    print(f"OUT_DIR = {OUT_DIR.resolve()}")

    # 1) берём несколько реальных объектов из test.csv
    test_path = OUT_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Нет {test_path}. Сначала запустите run_part_a.py."
        )
    test_df = pd.read_csv(test_path)
    n_show = 5
    sample = test_df.head(n_show).copy()

    # 2) список признаков (y не входит)
    num_cols, cat_cols = column_types(test_df)
    feats = num_cols + cat_cols
    print(f"Признаков: {len(feats)} (num={len(num_cols)}, cat={len(cat_cols)})")
    print(f"Объектов для примера: {n_show}")

    # 3) предсказания тремя моделями
    nn_cfg = load_nn_best_cfg()
    print(f"NN-конфигурация: {nn_cfg}")

    p_own = predict_own_logreg(sample, feats)
    p_lib = predict_logreg(sample, feats)
    p_nn = predict_nn(sample, feats, nn_cfg)

    out = pd.DataFrame({
        "true_y": sample["y"].values,
        "p_own_logreg": p_own,
        "p_logreg": p_lib,
        "p_nn": p_nn,
    })

    pd.set_option("display.width", 120)
    print("\n=== ПРИМЕР ПРЕДСКАЗАНИЯ (5 объектов из test.csv) ===")
    print(out.to_string(index=False,
                        float_format=lambda x: f"{x:.4f}"))

    # 4) сохраняем
    out_path = OUT_DIR / "prediction_example.csv"
    out.to_csv(out_path, index=False)
    print(f"\nСохранено: {out_path}")

    # 5) краткая интерпретация
    print("\nИнтерпретация:")
    print("  - p_* — вероятность отклика (y=1) по каждой модели;")
    print("  - модели ранжируют клиентов; при пороге из валидации")
    print("    (например, thr≈0.044 для own_logreg) p>=thr => 'позвонить'.")
    thr_own = 0.044  # из final_leaderboard.csv (own_logreg)
    print(f"  - пример с thr_own={thr_own}: "
          f"{(p_own >= thr_own).sum()} из {n_show} объектов к обзвону.")


if __name__ == "__main__":
    main()