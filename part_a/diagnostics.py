"""
Диагностика для отчёта по части А:
1) временной дрейф доли положительного класса (train/test/folds)
2) кривые loss/AP нейросети по сохранённым nn_history_*.csv
3) парное сравнение own_logreg ↔ sklearn.LogisticRegression
   на одном разбиении при одинаковом C: AP, max|Δw|, |Δb|, время.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from config import DATA_PATH, OUT_DIR, RANDOM_STATE, TEST_FRACTION, N_EXPANDING_FOLDS
from data_utils import load_bank_marketing, clean, column_types
from splits import time_split, expanding_window_folds
from models_sklearn import make_preprocessor, OwnLogregAdapter
from own_logreg import OwnLogisticRegression

DIAG_DIR = OUT_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 1. ДРЕЙФ
def drift_report():
    df = clean(load_bank_marketing(DATA_PATH))
    train_df, test_df = time_split(df, TEST_FRACTION)
    folds = expanding_window_folds(train_df, N_EXPANDING_FOLDS)

    rows = [
        {"split": "all", "n": len(df), "n_pos": int(df["y"].sum()),
         "share_pos": float(df["y"].mean())},
        {"split": "train", "n": len(train_df), "n_pos": int(train_df["y"].sum()),
         "share_pos": float(train_df["y"].mean())},
        {"split": "test", "n": len(test_df), "n_pos": int(test_df["y"].sum()),
         "share_pos": float(test_df["y"].mean())},
    ]
    for i, (itr, iva) in enumerate(folds):
        sub_va = train_df.iloc[iva]
        rows.append({"split": f"fold{i}_val", "n": len(sub_va),
                     "n_pos": int(sub_va["y"].sum()),
                     "share_pos": float(sub_va["y"].mean())})
        sub_tr = train_df.iloc[itr]
        rows.append({"split": f"fold{i}_train", "n": len(sub_tr),
                     "n_pos": int(sub_tr["y"].sum()),
                     "share_pos": float(sub_tr["y"].mean())})
    tab = pd.DataFrame(rows)
    tab.to_csv(DIAG_DIR / "drift_share_pos.csv", index=False)
    print("\n=== ДРЕЙФ ДОЛИ ПОЛОЖИТЕЛЬНОГО КЛАССА ===")
    print(tab.to_string(index=False))

    window = max(200, len(df) // 100)
    roll = df["y"].rolling(window=window, min_periods=window // 2).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(roll.values, lw=1.2, label=f"rolling share y=1 (window={window})")
    ax.axvline(len(train_df), color="k", ls="--", label="train/test boundary")
    ax.set_xlabel("index (время наблюдения)")
    ax.set_ylabel("доля y=1")
    ax.set_title("Дрейф доли отклика во времени")
    ax.legend()
    plt.tight_layout()
    plt.savefig(DIAG_DIR / "drift_target_over_time.png", dpi=120)
    plt.close()
    print(f"  график: {DIAG_DIR / 'drift_target_over_time.png'}")
    return tab


# ------------------------------------------------------- 2. КРИВЫЕ NN
def nn_curves():
    files = sorted(OUT_DIR.glob("nn_history_*.csv"))
    if not files:
        print("\n[NN-кривые] нет файлов nn_history_*.csv — пропуск")
        return
    print(f"\n=== NN-КРИВЫЕ: {len(files)} файлов ===")
    groups = {}
    for f in files:
        df = pd.read_csv(f)
        groups[f.stem] = df

    finals = [k for k in groups if "final" in k]
    if finals:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for k in finals:
            d = groups[k]
            axes[0].plot(d["epoch"], d["train_loss_batch"], alpha=0.5, label=f"{k} train")
            axes[0].plot(d["epoch"], d["val_loss"], alpha=0.9, ls="--", label=f"{k} val")
            axes[1].plot(d["epoch"], d["val_AP"], label=k)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss")
        axes[0].set_title("Loss (train batch / val)"); axes[0].legend(fontsize=7)
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val AP")
        axes[1].set_title("val AP по эпохам"); axes[1].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(DIAG_DIR / "nn_loss_ap_final.png", dpi=120)
        plt.close()
        print(f"  график: {DIAG_DIR / 'nn_loss_ap_final.png'}")

    rows = []
    for k, d in groups.items():
        i_min_loss = int(d["val_loss"].idxmin())
        i_max_ap = int(d["val_AP"].idxmax())
        rows.append({
            "run": k,
            "epochs": int(d["epoch"].max()) + 1,
            "train_params": int(d["train_params"].iloc[0]),
            "best_val_AP": float(d["val_AP"].max()),
            "epoch_best_AP": int(d.loc[i_max_ap, "epoch"]),
            "min_val_loss": float(d["val_loss"].min()),
            "epoch_min_loss": int(d.loc[i_min_loss, "epoch"]),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(DIAG_DIR / "nn_curves_summary.csv", index=False)
    print(summary.to_string(index=False))


# ------------------------------- 3. ПАРНОЕ СРАВНЕНИЕ OWN ↔ LIBRARY
def own_vs_library(C_values=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """
    Одинаковые признаки, разбиение, C. Сравниваем AP на валидации
    последнего expanding-фолда, max|Δw|, |Δb|, время.
    sklearn — LBFGS; own — GD. Побитового совпадения не ждём.
    """
    df = clean(load_bank_marketing(DATA_PATH))
    train_df, _ = time_split(df, TEST_FRACTION)
    folds = expanding_window_folds(train_df, N_EXPANDING_FOLDS)
    itr, iva = folds[-1]  # последний фолд
    sub_tr = train_df.iloc[itr].reset_index(drop=True)
    sub_va = train_df.iloc[iva].reset_index(drop=True)

    num_cols, cat_cols = column_types(df)
    feats = num_cols + cat_cols
    X_tr_raw, y_tr = sub_tr[feats], sub_tr["y"].values
    X_va_raw, y_va = sub_va[feats], sub_va["y"].values

    rows = []
    for C in C_values:
        # --- own ---
        pre_own = make_preprocessor(num_cols, cat_cols)
        t0 = time.time()
        own = OwnLogregAdapter(C=C, lr=0.05, max_iter=1000).fit(pre_own, X_tr_raw, y_tr)
        t_own_fit = time.time() - t0
        t0 = time.time()
        s_own = own.score(X_va_raw)
        t_own_pred = time.time() - t0
        ap_own = float(average_precision_score(y_va, s_own))

        # --- library ---
        pre_lib = make_preprocessor(num_cols, cat_cols)
        Xtr = pre_lib.fit_transform(X_tr_raw)
        Xva = pre_lib.transform(X_va_raw)
        t0 = time.time()
        lib = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        lib.fit(Xtr, y_tr)
        t_lib_fit = time.time() - t0
        t0 = time.time()
        s_lib = lib.predict_proba(Xva)[:, 1]
        t_lib_pred = time.time() - t0
        ap_lib = float(average_precision_score(y_va, s_lib))

        # --- сравнение коэффициентов на одинаковом пространстве ---
        # own обучен на pre_own, lib — на pre_lib; пространства должны совпадать
        # (одинаковый ColumnTransformer + одинаковый порядок), но проверим формы
        w_own = own.model.w
        b_own = own.model.b
        w_lib = lib.coef_.ravel()
        b_lib = float(lib.intercept_[0])
        same_dim = (w_own.shape == w_lib.shape)
        if same_dim:
            dw = float(np.max(np.abs(w_own - w_lib)))
            db = float(abs(b_own - b_lib))
        else:
            dw, db = float("nan"), float("nan")

        rows.append({
            "C": C,
            "AP_own": ap_own,
            "AP_lib": ap_lib,
            "dAP": ap_own - ap_lib,
            "max_abs_dw": dw,
            "abs_db": db,
            "same_dim": same_dim,
            "dim": int(w_own.size),
            "fit_time_own_s": t_own_fit,
            "fit_time_lib_s": t_lib_fit,
            "pred_time_own_s": t_own_pred,
            "pred_time_lib_s": t_lib_pred,
            "own_loss_iters": int(len(own.model.loss_history)),
            "own_final_loss": float(own.model.loss_history[-1])
            if own.model.loss_history else float("nan"),
            "lib_n_iter": int(getattr(lib, "n_iter_", [ -1 ])[0])
            if hasattr(lib, "n_iter_") else None,
        })

    tab = pd.DataFrame(rows)
    tab.to_csv(DIAG_DIR / "own_vs_library.csv", index=False)
    print("\n=== OWN ↔ LIBRARY (последний expanding-фолд) ===")
    print(tab.to_string(index=False))
    print("\nПояснение: sklearn.LogisticRegression — LBFGS, own — GD;")
    print("критерии остановки разные, побитового совпадения w/b не ждём.")
    print("Сравнивать имеет смысл AP при одинаковом C и порядок max|Δw|.")
    return tab


# ---------------------------------------------------------------- MAIN
def main():
    print(f"OUT_DIR = {OUT_DIR.resolve()}")
    print(f"DIAG_DIR = {DIAG_DIR.resolve()}")
    drift_report()
    nn_curves()
    own_vs_library()
    print("\nГотово. Артефакты в:", DIAG_DIR.resolve())


if __name__ == "__main__":
    main()