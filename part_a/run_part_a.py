import json
import time
import warnings
import traceback

import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from config import (DATA_PATH, OUT_DIR, RANDOM_STATE, TEST_FRACTION,
                    N_EXPANDING_FOLDS, PARAM_GRIDS, NN_CONFIGS, NN_SEEDS,
                    NN_MAX_EPOCHS, NN_PATIENCE)
from data_utils import (sha256_of, load_bank_marketing, data_passport,
                        clean, column_types)
from eda import eda_report
from splits import time_split, expanding_window_folds
from models_sklearn import (make_preprocessor, build_model, get_scores,
                            search_family, OwnLogregAdapter)
from model_nn import train_nn, predict_scores_nn, count_params
from metrics_utils import (classification_metrics, find_best_threshold,
                           plot_roc_pr, bootstrap_diff_ap)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.impute")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

LOG_PATH = OUT_DIR / "runs.csv"
SEARCH_PATH = OUT_DIR / "search_summary.csv"
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
log_rows = []


def log_run(**kw):
    kw.setdefault("run_id", RUN_ID)
    kw.setdefault("ts", time.time())
    log_rows.append(kw)
    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)


def _params_key(p):
    return json.dumps(p, sort_keys=True, default=str)


def plot_roc_pr_named(y_true, scores, prefix_dir, name):
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "--")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC")
    axes[1].plot(rec, prec)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("PR")
    plt.tight_layout()
    plt.savefig(Path(prefix_dir) / f"roc_pr_{name}.png", dpi=120)
    plt.close()


def main():
    # ---------- 1. ПАСПОРТ ДАННЫХ ----------
    print("[1/9] Паспорт данных")
    checksum = sha256_of(DATA_PATH)
    df_raw = load_bank_marketing(DATA_PATH)
    passport_raw = data_passport(df_raw)
    (OUT_DIR / "data_passport_raw.json").write_text(
        json.dumps({"sha256": checksum, **passport_raw},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    df = clean(df_raw)
    passport_clean = data_passport(df)
    (OUT_DIR / "data_passport_clean.json").write_text(
        json.dumps(passport_clean, indent=2, ensure_ascii=False),
        encoding="utf-8")

    num_cols, cat_cols = column_types(df)
    print(f"  shape={df.shape}, num_cols={len(num_cols)}, cat_cols={len(cat_cols)}")

    # ---------- 2. РАЗБИЕНИЕ ----------
    print("[2/9] Разбиение train/test и expanding-window")
    train_df, test_df = time_split(df, TEST_FRACTION)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    folds = expanding_window_folds(train_df, N_EXPANDING_FOLDS)
    (OUT_DIR / "splits.json").write_text(json.dumps(
        {"n_train": len(train_df), "n_test": len(test_df),
         "fold_sizes": [(int(len(a)), int(len(b))) for a, b in folds]}),
        encoding="utf-8")
    print(f"  n_train={len(train_df)}, n_test={len(test_df)}, "
          f"folds={[(len(a), len(b)) for a, b in folds]}")

    # ---------- 3. EDA ----------
    print("[3/9] EDA на train")
    eda_report(train_df, str(OUT_DIR))

    # ---------- 4. ПОИСК ГИПЕРПАРАМЕТРОВ ----------
    print("[4/9] Поиск гиперпараметров на expanding-window")
    families = ["logreg", "knn", "svm", "tree", "rf", "gb", "nb", "own_logreg"]
    best_per_family = {}
    search_tables = []

    for fam in families:
        print(f"\n=== {fam} ===")
        all_tables = []
        for fi, (itr, iva) in enumerate(folds):
            sub_tr = train_df.iloc[itr].reset_index(drop=True)
            sub_va = train_df.iloc[iva].reset_index(drop=True)

            if fam == "svm" and "rbf" in PARAM_GRIDS["svm"].get("kernel", []):
                n_svm = min(8000, len(sub_tr))
                sub_tr_svm = sub_tr.sample(n=n_svm, random_state=RANDOM_STATE).reset_index(drop=True)
            else:
                sub_tr_svm = sub_tr

            try:
                best, table = search_family(
                    fam, sub_tr_svm, sub_va, num_cols, cat_cols,
                    PARAM_GRIDS[fam], log_rows)
            except Exception as e:
                print(f"  [ОШИБКА fold {fi}] {type(e).__name__}: {e}")
                traceback.print_exc()
                continue

            if len(table) == 0:
                continue
            table["fold"] = fi
            table["family"] = fam
            all_tables.append(table)
            print(f"  fold {fi}: best AP={best['AP']:.4f} "
                  f"params={best['params']} "
                  f"(перебрано {len(table)} конфигураций)")

        if not all_tables:
            print(f"  [пропуск] {fam}: не удалось получить результаты")
            continue

        df_fam = pd.concat(all_tables, ignore_index=True)
        df_fam["params_key"] = df_fam["params"].apply(_params_key)

        agg = (df_fam
               .groupby("params_key")
               .agg(mean_AP=("AP", "mean"),
                    std_AP=("AP", "std"),
                    mean_time_s=("time_s", "mean"),
                    n_folds=("AP", "count"))
               .reset_index()
               .sort_values("mean_AP", ascending=False))
        agg["family"] = fam
        search_tables.append(agg)

        best_row = agg.iloc[0]
        best_per_family[fam] = {
            "best_params": json.loads(best_row["params_key"]),
            "mean_AP": float(best_row["mean_AP"]),
            "std_AP": float(best_row["std_AP"] if pd.notna(best_row["std_AP"]) else 0.0),
            "mean_time_s": float(best_row["mean_time_s"]),
            "n_folds": int(best_row["n_folds"]),
        }
        print(f"  >> {fam}: mean AP={best_row['mean_AP']:.4f} "
              f"± {best_row['std_AP'] if pd.notna(best_row['std_AP']) else 0:.4f} "
              f"params={best_row['params_key']}")
        log_run(model=fam, stage="search",
                best_params=best_row["params_key"],
                mean_AP=float(best_row["mean_AP"]),
                std_AP=float(best_row["std_AP"] if pd.notna(best_row["std_AP"]) else 0.0),
                metric="AP_expanding")

    if search_tables:
        pd.concat(search_tables, ignore_index=True).to_csv(SEARCH_PATH, index=False)
        print(f"  сводка поиска сохранена: {SEARCH_PATH}")

    # ---------- 5. DUMMY ----------
    print("\n[5/9] Dummy-бейзлайн на тесте")
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy="prior").fit(
        train_df[num_cols + cat_cols], train_df["y"])
    dummy_scores = dummy.predict_proba(test_df[num_cols + cat_cols])[:, 1]
    dummy_metrics = classification_metrics(test_df["y"].values, dummy_scores, 0.5)
    log_run(model="dummy", stage="test", metrics=json.dumps(dummy_metrics))
    print(f"  DUMMY test AP: {dummy_metrics['AP']:.4f}")

    # ---------- 6. НЕЙРОСЕТЬ ----------
    print("\n[6/9] Обучение нейросети")
    pre_nn = make_preprocessor(num_cols, cat_cols)
    X_tr_full = pre_nn.fit_transform(train_df[num_cols + cat_cols])
    X_te_full = pre_nn.transform(test_df[num_cols + cat_cols])
    y_tr_full = train_df["y"].values
    y_te_full = test_df["y"].values

    n_inner = int(0.15 * len(train_df))
    X_fit, y_fit = X_tr_full[:-n_inner], y_tr_full[:-n_inner]
    X_inner, y_inner = X_tr_full[-n_inner:], y_tr_full[-n_inner:]

    nn_results = []
    for ci, cfg in enumerate(NN_CONFIGS):
        seed_aps = []
        for seed in NN_SEEDS:
            model, hist, best = train_nn(
                X_fit, y_fit, X_inner, y_inner, cfg,
                max_epochs=NN_MAX_EPOCHS, patience=NN_PATIENCE, seed=seed)
            pd.DataFrame(hist).to_csv(
                OUT_DIR / f"nn_history_cfg{ci}_seed{seed}.csv", index=False)
            ap_inner = float(max(h["val_AP"] for h in hist)) if hist else float("nan")
            seed_aps.append(ap_inner)
            nn_results.append({"cfg": cfg, "seed": seed,
                               "inner_AP": ap_inner,
                               "epochs_used": best["epoch"] + 1,
                               "n_params": count_params(model)})
        log_run(model="nn", stage="search",
                params=json.dumps(cfg, sort_keys=True),
                metric="AP_inner",
                mean=float(np.mean(seed_aps)),
                std=float(np.std(seed_aps)))

    cfg_groups = {}
    for r in nn_results:
        k = json.dumps(r["cfg"], sort_keys=True)
        cfg_groups.setdefault(k, []).append(r["inner_AP"])
    cfg_summary = [
        {"cfg_key": k, "cfg": json.loads(k),
         "mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in cfg_groups.items()
    ]
    cfg_summary.sort(key=lambda r: -r["mean"])
    nn_best_cfg = cfg_summary[0]
    print(f"  лучшая NN-конфигурация: mean inner AP={nn_best_cfg['mean']:.4f} "
          f"± {nn_best_cfg['std']:.4f} cfg={nn_best_cfg['cfg']}")

    nn_test_scores = []
    nn_states = []
    nn_n_params = None
    for seed in NN_SEEDS:
        model, hist, best = train_nn(
            X_tr_full, y_tr_full, X_inner, y_inner,
            nn_best_cfg["cfg"],
            max_epochs=NN_MAX_EPOCHS, patience=NN_PATIENCE, seed=seed)
        pd.DataFrame(hist).to_csv(
            OUT_DIR / f"nn_history_final_seed{seed}.csv", index=False)
        nn_n_params = count_params(model)
        s = predict_scores_nn(model, X_te_full)
        nn_test_scores.append(s)
        nn_states.append({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                          "seed": seed,
                          "n_params": nn_n_params})
        log_run(model="nn", stage="test_final_seed",
                params=json.dumps(nn_best_cfg["cfg"], sort_keys=True),
                seed=seed,
                AP=classification_metrics(y_te_full, s)["AP"],
                n_params=nn_n_params)

    nn_scores_avg = np.mean(nn_test_scores, axis=0)
    nn_metrics = classification_metrics(y_te_full, nn_scores_avg)
    nn_metrics["n_params"] = int(nn_n_params) if nn_n_params is not None else None
    log_run(model="nn", stage="test",
            params=json.dumps(nn_best_cfg["cfg"], sort_keys=True),
            metrics=json.dumps(nn_metrics), metric="AP")
    torch.save(nn_states, OUT_DIR / "nn_weights.pt")
    joblib.dump(pre_nn, OUT_DIR / "nn_preprocessor.joblib")
    plot_roc_pr_named(y_te_full, nn_scores_avg, str(OUT_DIR), name="nn")
    print(f"  NN test AP: {nn_metrics['AP']:.4f}  n_params={nn_n_params}")

    # ---------- 7. ФИНАЛЬНЫЕ КЛАССИЧЕСКИЕ МОДЕЛИ ----------
    print("\n[7/9] Финальное обучение классических моделей на всём train")
    test_scores = {}
    predict_times = {}
    n_params_per_family = {}
    for fam, info in best_per_family.items():
        params = info["best_params"]
        print(f"  {fam}: {params}")
        try:
            t0 = time.time()
            if fam == "own_logreg":
                pre = make_preprocessor(num_cols, cat_cols)
                adapter = OwnLogregAdapter(**params).fit(
                    pre, train_df[num_cols + cat_cols], train_df["y"])
                train_time = time.time() - t0
                tp0 = time.time()
                scores = adapter.score(test_df[num_cols + cat_cols])
                predict_times[fam] = time.time() - tp0
                n_params_per_family[fam] = int(adapter.model.w.size + 1)
                joblib.dump({"type": "own", "pre": pre,
                             "model": adapter.model},
                            OUT_DIR / "own_logreg_final.joblib")
            else:
                pre = make_preprocessor(num_cols, cat_cols)
                model = build_model(fam, params)
                from sklearn.pipeline import Pipeline
                pipe = Pipeline([("pre", pre), ("model", model)])
                pipe.fit(train_df[num_cols + cat_cols], train_df["y"])
                train_time = time.time() - t0
                tp0 = time.time()
                scores = get_scores(pipe, test_df[num_cols + cat_cols])
                predict_times[fam] = time.time() - tp0
                # число обучаемых параметров, где применимо
                n_params_per_family[fam] = _count_model_params(pipe.named_steps["model"])
                joblib.dump(pipe, OUT_DIR / f"{fam}_final.joblib")
        except Exception as e:
            print(f"  [ОШИБКА {fam}] {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # порог по последнему expanding-фолду
        itr, iva = folds[-1]
        sub_tr = train_df.iloc[itr].reset_index(drop=True)
        sub_va = train_df.iloc[iva].reset_index(drop=True)
        try:
            if fam == "own_logreg":
                pre_v = make_preprocessor(num_cols, cat_cols)
                tmp = OwnLogregAdapter(**params).fit(
                    pre_v, sub_tr[num_cols + cat_cols], sub_tr["y"])
                val_scores = tmp.score(sub_va[num_cols + cat_cols])
            else:
                from sklearn.pipeline import Pipeline
                pre_v = make_preprocessor(num_cols, cat_cols)
                p_v = Pipeline([("pre", pre_v),
                                ("model", build_model(fam, params))])
                p_v.fit(sub_tr[num_cols + cat_cols], sub_tr["y"])
                val_scores = get_scores(p_v, sub_va[num_cols + cat_cols])
            thr, _ = find_best_threshold(sub_va["y"].values, val_scores, metric="f1")
        except Exception as e:
            print(f"  [порог {fam}] {type(e).__name__}: {e} — беру 0.5")
            thr = 0.5

        metrics = classification_metrics(test_df["y"].values, scores, thr)
        metrics["train_time_s"] = float(train_time)
        metrics["predict_time_s"] = float(predict_times[fam])
        metrics["threshold_source"] = "val (last expanding fold)"
        metrics["n_params"] = n_params_per_family.get(fam)
        test_scores[fam] = scores
        log_run(model=fam, stage="test", params=json.dumps(params, default=str),
                metrics=json.dumps(metrics), metric="AP")
        plot_roc_pr_named(test_df["y"].values, scores, str(OUT_DIR), name=fam)
        print(f"  {fam:12s} test AP={metrics['AP']:.4f} "
              f"F1={metrics['f1']:.4f} thr={thr:.3f} "
              f"train={train_time:.1f}s pred={predict_times[fam]:.2f}s "
              f"n_params={metrics['n_params']}")

    # ---------- 8. СРАВНЕНИЕ ЛУЧШИХ ДВУХ ----------
    print("\n[8/9] Сравнение двух лучших (bootstrap AP)")
    family_ap = {fam: classification_metrics(test_df["y"].values, s)["AP"]
                 for fam, s in test_scores.items()}
    family_ap["nn"] = nn_metrics["AP"]
    all_scores = dict(test_scores); all_scores["nn"] = nn_scores_avg

    sorted_by_ap = sorted(family_ap.items(), key=lambda kv: -kv[1])
    print("  Итог по AP:")
    for name, ap in sorted_by_ap:
        print(f"    {name:12s} AP={ap:.4f}")

    if len(sorted_by_ap) >= 2:
        top2 = sorted_by_ap[:2]
        diff = bootstrap_diff_ap(test_df["y"].values,
                                 all_scores[top2[0][0]],
                                 all_scores[top2[1][0]])
        print(f"  Bootstrap AP({top2[0][0]}) - AP({top2[1][0]}): "
              f"mean={diff['mean']:+.4f} "
              f"95% CI [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]")
        (OUT_DIR / "bootstrap_top2.json").write_text(json.dumps(
            {"model_a": top2[0][0], "model_b": top2[1][0], **diff},
            indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- 9. ИТОГОВАЯ ТАБЛИЦА ----------
    print("\n[9/9] Итоговая таблица")
    rows = []
    for fam, ap in family_ap.items():
        if fam == "nn":
            n_par = nn_n_params
        else:
            n_par = n_params_per_family.get(fam)
        rows.append({
            "model": fam,
            "AP_test": ap,
            "n_params": n_par,
            "train_time_s": (float(best_per_family[fam]["mean_time_s"])
                             if fam in best_per_family else None),
            "predict_time_s": predict_times.get(fam, None),
        })
    final_table = pd.DataFrame(rows).sort_values("AP_test", ascending=False)
    final_table.to_csv(OUT_DIR / "final_leaderboard.csv", index=False)
    print(final_table.to_string(index=False))

    print(f"\nЖурнал:      {LOG_PATH}")
    print(f"Сводка:      {SEARCH_PATH}")
    print(f"Артефакты:   {OUT_DIR.resolve()}")


def _count_model_params(model):
    """Оценка числа обучаемых параметров для классических моделей (где есть)."""
    try:
        if hasattr(model, "coef_"):
            n = int(np.sum(model.coef_.size))
            if getattr(model, "intercept_", None) is not None:
                n += int(np.sum(np.asarray(model.intercept_).size))
            return n
        if hasattr(model, "tree_"):
            t = model.tree_
            return int(t.node_count)
        if hasattr(model, "estimators_"):
            # ансамбли деревьев: суммарное число узлов
            total = 0
            for est in np.ravel(model.estimators_):
                if hasattr(est, "tree_"):
                    total += int(est.tree_.node_count)
            return int(total) if total > 0 else None
        if hasattr(model, "classes_"):
            return None  # NB/Dummy — параметры не сопоставимы напрямую
    except Exception:
        return None
    return None


if __name__ == "__main__":
    import time as _time
    _t0 = _time.perf_counter()
    main()
    _total = _time.perf_counter() - _t0
    print(f"\nСуммарное время прогона: {_total:.2f} с")