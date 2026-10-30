import json
import time
import warnings
import traceback

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from config import (DATA_PATH, OUT_DIR, RANDOM_STATE,
                    WINSORIZE_QUANTILE, WHOLESALE_QUANTILE,
                    KMEANS_KS, KMEANS_N_INIT,
                    AGG_KS, AGG_LINKAGES,
                    DBSCAN_MIN_SAMPLES, DBSCAN_EPS_QUANTILES,
                    GMM_KS, GMM_COV_TYPES,
                    PCA_VARIANCE_TARGETS,
                    TSNE_PERPLEXITIES, TSNE_INITS,
                    STABILITY_N_SUBSAMPLES, STABILITY_FRACTION,
                    STABILITY_SEED, OWN_VS_SKLEARN_N_INIT)
from data_utils_b import (sha256_of, load_online_retail, data_passport_raw,
                          clean_transactions, build_customer_table,
                          customer_passport)
from eda_b import eda_report
from preprocessing_b import build_feature_matrix, scale, compute_clip_bounds
from own_pca import OwnPCA
from own_kmeans import OwnKMeans
from clustering_b import (run_kmeans, run_agglomerative, run_dbscan,
                          run_gmm, run_own_kmeans,
                          compare_own_vs_sklearn_kmeans)
from metrics_b import (internal_metrics, internal_metrics_on_common_subset,
                       adjusted_rand, cluster_sizes)
from viz_b import (plot_pca_variance, plot_kdistance, plot_tsne)

warnings.filterwarnings("ignore")

LOG_PATH = OUT_DIR / "runs_b.csv"
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
log_rows: list[dict] = []


def log(**kw):
    kw["run_id"] = RUN_ID
    log_rows.append(kw)
    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)


# ---------- загрузка / кэш ----------
def load_or_build_customers():
    cache_parquet = OUT_DIR / "customers.parquet"
    cache_csv = OUT_DIR / "customers.csv"
    for cache in (cache_parquet, cache_csv):
        if cache.exists():
            try:
                dfc = (pd.read_parquet(cache) if cache.suffix == ".parquet"
                       else pd.read_csv(cache))
                if "is_wholesale" in dfc.columns:
                    print(f"  [cache] {cache}")
                    return dfc
                else:
                    print(f"  [cache stale, no is_wholesale] {cache} — пересоздаю")
                    cache.unlink()
            except Exception as e:
                print(f"  [cache error] {e} — пересоздаю")
                try:
                    cache.unlink()
                except Exception:
                    pass

    checksum = sha256_of(DATA_PATH)
    df = load_online_retail(DATA_PATH)
    raw_pass = data_passport_raw(df)
    (OUT_DIR / "passport_raw.json").write_text(
        json.dumps({"sha256": checksum, **raw_pass},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  raw: {df.shape}, отмен={raw_pass['canceled_invoices']}, "
          f"без CustomerID={raw_pass['missing_customer']}")

    dfc = clean_transactions(df)
    print(f"  clean: {dfc.shape}")

    rfm, snapshot = build_customer_table(dfc,
                                         wholesale_quantile=WHOLESALE_QUANTILE)
    print(f"  клиентов: {len(rfm)}, VIP/опт: "
          f"{int(rfm['is_wholesale'].sum())}, дата среза: {snapshot.date()}")

    (OUT_DIR / "passport_customers.json").write_text(
        json.dumps(customer_passport(rfm), indent=2, ensure_ascii=False),
        encoding="utf-8")

    try:
        rfm.to_parquet(cache_parquet, index=False)
        print(f"  кэш сохранён: {cache_parquet}")
    except Exception:
        rfm.to_csv(cache_csv, index=False)
        print(f"  кэш сохранён: {cache_csv} (parquet недоступен)")
    return rfm


def pick_best_for_metric(results, metric="silhouette"):
    scored = []
    for r in results:
        try:
            m = internal_metrics(r["_Z"], r["labels"])
        except Exception:
            continue
        if m.get(metric) is None:
            continue
        scored.append((r, m))
    if not scored:
        return None, None
    if metric == "davies_bouldin":
        scored.sort(key=lambda x: x[1][metric])
    else:
        scored.sort(key=lambda x: -x[1][metric])
    return scored[0]


# ---------- PCA: сравнение own vs sklearn ----------
def compare_own_pca_with_sklearn(Z, ks=(2, 3, 5), seed=RANDOM_STATE):
    """
    Сравнивает OwnPCA и sklearn PCA по:
    - explained_variance_ratio (L1),
    - reconstruction MSE,
    - |cos| между компонентами (с учётом неопределённости знака).
    """
    from sklearn.decomposition import PCA as SkPCA
    rows = []
    pca_own_full = OwnPCA().fit(Z)
    pca_sk_full = SkPCA().fit(Z)
    ev_own = pca_own_full.explained_variance_ratio_
    ev_sk = pca_sk_full.explained_variance_ratio_
    m = min(len(ev_own), len(ev_sk))
    ev_l1 = float(np.abs(ev_own[:m] - ev_sk[:m]).sum())

    # PCA ограничен min(n_samples, n_features)
    max_k = min(Z.shape)
    valid_ks = sorted({k for k in ks if 1 <= k <= max_k})
    if not valid_ks:
        valid_ks = [max_k]

    for k in valid_ks:
        pca_own = OwnPCA(n_components=k).fit(Z)
        pca_sk = SkPCA(n_components=k).fit(Z)
        err_own = pca_own.reconstruction_error(Z)
        Xr_sk = pca_sk.inverse_transform(pca_sk.transform(Z))
        err_sk = float(((Z - Xr_sk) ** 2).mean())
        A = pca_own.components_
        B = pca_sk.components_
        cos = np.abs(np.sum(A * B, axis=1)) / (
            np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
        rows.append({
            "k": int(k),
            "reconstruction_mse_own": err_own,
            "reconstruction_mse_sklearn": err_sk,
            "abs_diff_mse": float(abs(err_own - err_sk)),
            "mean_abs_cos_components": float(cos.mean()),
            "min_abs_cos_components": float(cos.min()),
        })
    return {
        "n_samples": int(Z.shape[0]),
        "n_features": int(Z.shape[1]),
        "valid_ks": valid_ks,
        "explained_variance_ratio_l1_diff_full": ev_l1,
        "per_k": rows,
    }


# ---------- устойчивость: добавлена ветка own_kmeans ----------
def stability_for_top2(top2, Z_by_space, rfm, n, seed):
    stab_rows = []
    rng = np.random.default_rng(seed)
    for r, m in top2:
        Z = r["_Z"]
        labels_full = r["labels"]
        aris = []
        noise_transitions = []   # только для DBSCAN
        for i in range(STABILITY_N_SUBSAMPLES):
            idx = rng.choice(n, size=int(STABILITY_FRACTION * n),
                             replace=False)
            Zsub = Z[idx]
            method = r["method"]
            try:
                if method == "kmeans":
                    from sklearn.cluster import KMeans
                    lsub = KMeans(n_clusters=r["k"], n_init=10,
                                  random_state=seed + i).fit_predict(Zsub)
                elif method == "own_kmeans":
                    km = OwnKMeans(n_clusters=r["k"], n_init=5,
                                   random_state=seed + i).fit(Zsub)
                    lsub = km.labels_
                elif method.startswith("agg_"):
                    from sklearn.cluster import AgglomerativeClustering
                    link = method.split("_", 1)[1]
                    lsub = AgglomerativeClustering(
                        n_clusters=r["k"], linkage=link,
                        metric="euclidean").fit_predict(Zsub)
                elif method.startswith("gmm_"):
                    from sklearn.mixture import GaussianMixture
                    cov = method.split("_", 1)[1]
                    lsub = GaussianMixture(n_components=r["k"],
                                           covariance_type=cov,
                                           random_state=seed + i,
                                           n_init=3).fit_predict(Zsub)
                elif method == "dbscan":
                    from sklearn.cluster import DBSCAN
                    lsub = DBSCAN(eps=r["eps"],
                                  min_samples=r["min_samples"]
                                  ).fit_predict(Zsub)
                    # --- учёт переходов в шум (требование 4.3) ---
                    full_noise = labels_full[idx] == -1
                    sub_noise = lsub == -1
                    noise_transitions.append(
                        float((full_noise != sub_noise).mean()))
                else:
                    continue
                ari = adjusted_rand(labels_full[idx], lsub)
                aris.append(ari)
            except Exception as e:
                print(f"  [stab {method}] {type(e).__name__}: {e}")
                continue
        if aris:
            row = {
                "method": r["method"], "k": r.get("k"),
                "space": r["space"],
                "mean_ARI": float(np.mean(aris)),
                "std_ARI": float(np.std(aris)),
                "n_runs": len(aris),
                "note": "фиксированные преобразования (устойчивость алгоритма)",
            }
            if noise_transitions:
                row["mean_noise_transition_rate"] = float(
                    np.mean(noise_transitions))
                row["std_noise_transition_rate"] = float(
                    np.std(noise_transitions))
            stab_rows.append(row)
    return stab_rows

# Вместо прямого вызова run_* — оборачиваем и проставляем time_sec
def timed(fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    dt = time.perf_counter() - t0
    if isinstance(out, list):
        for r in out:
            r["time_sec"] = dt / max(len(out), 1)  # оценка на 1 конфиг
    return out


def main():
    print("[1/9] Загрузка и агрегация")
    rfm_all = load_or_build_customers()
    print(f"  форма rfm: {rfm_all.shape}")

    rfm = rfm_all[rfm_all["is_wholesale"] == 0].reset_index(drop=True)
    rfm_wholesale = rfm_all[rfm_all["is_wholesale"] == 1].reset_index(drop=True)
    print(f"  основная масса: {len(rfm)}, VIP/опт: {len(rfm_wholesale)}")

    print("[2/9] EDA по RFM (основная масса)")
    eda_report(rfm, str(OUT_DIR))
    if len(rfm_wholesale):
        rfm_wholesale.to_csv(OUT_DIR / "wholesale_customers.csv", index=False)

    print("[3/9] Признаки: RFM vs расширенный (winsorize + log1p)")
    # считаем пороги на обучающей части (здесь — на всей основной массе,
    # т.к. в части Б нет временного разбиения; это явно оговорить в отчёте)
    X_rfm_raw, cols_rfm = build_feature_matrix(
        rfm, which="rfm", log_transform=False,
        winsorize_quantile=WINSORIZE_QUANTILE)
    X_ext_raw, cols_ext = build_feature_matrix(
        rfm, which="ext", log_transform=False,
        winsorize_quantile=WINSORIZE_QUANTILE)

    clip_rfm = compute_clip_bounds(X_rfm_raw, WINSORIZE_QUANTILE)
    clip_ext = compute_clip_bounds(X_ext_raw, WINSORIZE_QUANTILE)

    # применяем log1p уже после winsorize (внутри build_feature_matrix)
    X_rfm, cols_rfm = build_feature_matrix(
        rfm, which="rfm", log_transform=True, clip_bounds=clip_rfm)
    X_ext, cols_ext = build_feature_matrix(
        rfm, which="ext", log_transform=True, clip_bounds=clip_ext)

    Z_rfm, _, sc_rfm = scale(X_rfm)
    Z_ext, _, sc_ext = scale(X_ext)
    np.save(OUT_DIR / "Z_rfm.npy", Z_rfm)
    np.save(OUT_DIR / "Z_ext.npy", Z_ext)
    joblib.dump({
        "sc_rfm": sc_rfm, "sc_ext": sc_ext,
        "cols_rfm": cols_rfm, "cols_ext": cols_ext,
        "winsorize_quantile": WINSORIZE_QUANTILE,
        "clip_bounds_rfm": clip_rfm,
        "clip_bounds_ext": clip_ext,
    }, OUT_DIR / "preprocessor.joblib")

    print("[4/9] Кластеризация на двух пространствах")
    all_results = []
    for which, Z in [("rfm", Z_rfm), ("ext", Z_ext)]:
        for r in timed(run_kmeans, Z, KMEANS_KS, KMEANS_N_INIT, RANDOM_STATE):
            r["_Z"] = Z; r["space"] = which; all_results.append(r)
        for r in timed(run_own_kmeans, Z, KMEANS_KS, RANDOM_STATE):
            r["_Z"] = Z; r["space"] = which; all_results.append(r)
        for r in timed(run_agglomerative, Z, AGG_KS, AGG_LINKAGES):
            r["_Z"] = Z; r["space"] = which; all_results.append(r)
        for r in timed(run_dbscan, Z, DBSCAN_MIN_SAMPLES, DBSCAN_EPS_QUANTILES):
            r["_Z"] = Z; r["space"] = which; all_results.append(r)
        for r in timed(run_gmm, Z, GMM_KS, GMM_COV_TYPES):
            r["_Z"] = Z; r["space"] = which; all_results.append(r)

    # -- Метрики + машиночитаемый журнал --
    metric_rows = []
    for r in all_results:
        m = internal_metrics(r["_Z"], r["labels"])
        row = {"method": r["method"], "space": r["space"],
               "k": r.get("k"), "min_samples": r.get("min_samples"),
               "eps": r.get("eps"), "cov_type": r.get("cov_type"),
               "bic": r.get("bic"), "aic": r.get("aic"),
               "inertia": r.get("inertia")}
        row.update(m)
        row["sizes"] = json.dumps(cluster_sizes(r["labels"]))
        metric_rows.append(row)
        log(model=r["method"], space=r["space"],
            params=json.dumps({k: r.get(k) for k in
                               ("k", "min_samples", "eps", "cov_type",
                                "seed")}),
            seed=r.get("seed"), n_clusters=m["n_clusters"],
            noise_frac=m["noise_frac"], silhouette=m["silhouette"],
            davies_bouldin=m["davies_bouldin"],
            calinski_harabasz=m["calinski_harabasz"],
            time_sec=r.get("time_sec"),   # время, замеренное при fit
            status="ok")
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "clustering_metrics.csv",
                                     index=False)
    print("  результаты в clustering_metrics.csv + runs_b.csv")

    # -- Общее подмножество для сопоставления DBSCAN с другими --
    # Честное сравнение: для каждой пары (ref, r) берём объекты,
    # которые не шум у ОБОИХ. ref = лучший kmeans в том же пространстве.
    common_rows = []
    for space in ("rfm", "ext"):
        ref = next((r for r in all_results
                    if r["method"] == "kmeans" and r["space"] == space
                    and r.get("k") == 3), None)
        if ref is None:
            continue
        ref_labels = ref["labels"]
        for r in all_results:
            if r["space"] != space or r is ref:
                continue
            keep = (ref_labels != -1) & (r["labels"] != -1)
            m = internal_metrics_on_common_subset(
                r["_Z"], r["labels"], keep_mask=keep)
            common_rows.append({
                "method": r["method"], "space": space,
                "k": r.get("k"),
                "n_common": int(keep.sum()),
                "frac_common": float(keep.mean()),
                **m,
            })
    pd.DataFrame(common_rows).to_csv(
        OUT_DIR / "clustering_metrics_common.csv", index=False)
    print("  общие индексы (попарно с kmeans k=3) в clustering_metrics_common.csv")

    # -- Отбор лучших --
    valid = []
    for r in all_results:
        uniq = np.unique(r["labels"][r["labels"] != -1])
        if len(uniq) < 2:
            continue
        sizes = pd.Series(r["labels"]).value_counts(normalize=True)
        if sizes.iloc[0] > 0.90:
            continue
        valid.append(r)

    if not valid:
        print("  [!] нет результатов с осмысленным разделением")
        return

    best_r, best_m = pick_best_for_metric(valid, "silhouette")
    if best_r is None:
        print("  [!] не удалось выбрать лучшую модель")
        return
    print(f"  лучший по silhouette: {best_r['method']} k={best_r.get('k')} "
          f"sil={best_m['silhouette']:.4f} на space={best_r['space']}")

    print("[5/9] Сравнение own vs sklearn (KMeans и PCA)")
    # --- KMeans: при n_init=1 и одинаковых стартовых центрах ---
    kmeans_cmp_rows = []
    for which, Z in [("rfm", Z_rfm), ("ext", Z_ext)]:
        for k in (3, 4, 5):
            try:
                row = compare_own_vs_sklearn_kmeans(
                    Z, k=k, seed=RANDOM_STATE,
                    n_init=OWN_VS_SKLEARN_N_INIT)
                row["space"] = which
                kmeans_cmp_rows.append(row)
                print(f"  kmeans own vs sk [{which}, k={k}]: "
                      f"ARI={row['ari_own_vs_sklearn']:.4f}, "
                      f"Δinertia={row['abs_inertia_diff']:.3e}")
            except Exception as e:
                print(f"  [kmeans cmp {which}, k={k}] "
                      f"{type(e).__name__}: {e}")
    pd.DataFrame(kmeans_cmp_rows).to_csv(
        OUT_DIR / "own_vs_sklearn_kmeans.csv", index=False)

    # --- PCA: сравнение own vs sklearn ---
    pca_cmp = {}
    for which, Z in [("rfm", Z_rfm), ("ext", Z_ext)]:
        pca_cmp[which] = compare_own_pca_with_sklearn(Z, ks=(2, 3, 5))
    (OUT_DIR / "own_vs_sklearn_pca.json").write_text(
        json.dumps(pca_cmp, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  PCA own vs sklearn сохранено в own_vs_sklearn_pca.json")

    print("[6/9] Устойчивость двух лучших (5 × 80%)")
    seen = set(); top2 = []
    for r, m in sorted(
            [(r, internal_metrics(r["_Z"], r["labels"])) for r in valid],
            key=lambda x: -(x[1].get("silhouette") or -1)):
        fam = r["method"]
        if fam in seen:
            continue
        seen.add(fam); top2.append((r, m))
        if len(top2) == 2:
            break

    stab_rows = stability_for_top2(
        top2, {"rfm": Z_rfm, "ext": Z_ext}, rfm, len(rfm), STABILITY_SEED)
    pd.DataFrame(stab_rows).to_csv(OUT_DIR / "stability.csv", index=False)
    for row in stab_rows:
        print(f"  {row['method']}: mean ARI={row['mean_ARI']:.3f} "
              f"± {row['std_ARI']:.3f}")

    print("[7/9] PCA (own) и t-SNE")
    Z = Z_ext
    pca_full = OwnPCA().fit(Z)
    assert pca_full.explained_variance_ratio_ is not None
    plot_pca_variance(pca_full.explained_variance_ratio_,
                      OUT_DIR / "pca_variance.png")
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    k95 = int(np.searchsorted(cum, 0.95) + 1)

    pca90 = OwnPCA(n_components=k90).fit(Z)
    pca95 = OwnPCA(n_components=k95).fit(Z)
    err90 = pca90.reconstruction_error(Z)
    err95 = pca95.reconstruction_error(Z)
    err_full = pca_full.reconstruction_error(Z)
    print(f"  PCA: k90={k90} err={err90:.4f}, k95={k95} err={err95:.4f}, "
          f"full-rank err={err_full:.6f}")

    (OUT_DIR / "pca_summary.json").write_text(json.dumps({
        "k90": k90, "k95": k95,
        "reconstruction_mse_k90": err90,
        "reconstruction_mse_k95": err95,
        "reconstruction_mse_full": err_full,
        "explained_ratio": pca_full.explained_variance_ratio_.tolist(),
    }, indent=2), encoding="utf-8")

    Z_pca2 = pca_full.transform(Z)[:, :2]
    np.save(OUT_DIR / "Z_pca2.npy", Z_pca2)
    plt.figure(figsize=(7, 6))
    plt.scatter(Z_pca2[:, 0], Z_pca2[:, 1], s=4, alpha=0.5)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA (own), проекция 2D")
    plt.tight_layout(); plt.savefig(OUT_DIR / "pca_2d.png", dpi=120); plt.close()

    plot_kdistance(Z, min_samples=10, path=OUT_DIR / "kdistance.png")

    labels_for_tsne = best_r["labels"]
    for perp in TSNE_PERPLEXITIES:
        for init in TSNE_INITS:
            try:
                if perp >= len(Z):
                    # perplexity должна быть < n
                    continue
                title = f"t-SNE: {best_r['method']} (perp={perp}, init={init})"
                fn = f"tsne_{best_r['method']}_p{perp}_{init}.png"
                plot_tsne(Z, labels_for_tsne, title, OUT_DIR / fn,
                          perplexity=perp, init=init)
            except Exception as e:
                print(f"  [t-SNE {perp}/{init}] {type(e).__name__}: {e}")

    print("[8/9] Профили сегментов лучшей модели")
    labels = best_r["labels"]
    prof = rfm.copy()
    prof["cluster"] = labels
    prof_main = prof[prof["cluster"] != -1]
    profile = prof_main.groupby("cluster").agg(
        n=("CustomerID", "size"),
        Recency_median=("Recency", "median"),
        Frequency_median=("Frequency", "median"),
        Monetary_median=("Monetary", "median"),
        n_products_median=("n_products", "median"),
        avg_basket_median=("avg_basket", "median"),
    ).reset_index()
    profile["share"] = profile["n"] / profile["n"].sum()
    profile.to_csv(OUT_DIR / "segment_profile.csv", index=False)
    print(profile.to_string(index=False))

    # проверка «не выбросы ли кластеры»: сравнение медиан с общей выборкой
    overall_med = rfm[["Recency", "Frequency", "Monetary",
                       "n_products", "avg_basket"]].median()
    outlier_check = profile.copy()
    for c in ("Recency_median", "Frequency_median", "Monetary_median",
              "n_products_median", "avg_basket_median"):
        base = c.replace("_median", "")
        outlier_check[f"{base}_ratio_vs_all"] = (
            profile[c] / overall_med[base]).round(3)
    outlier_check.to_csv(OUT_DIR / "segment_outlier_check.csv", index=False)
    print("  проверка на выбросы: segment_outlier_check.csv")

    if len(rfm_wholesale):
        ws = rfm_wholesale[["CustomerID", "Recency", "Frequency",
                            "Monetary", "n_products", "avg_basket"]]
        ws.describe().to_csv(OUT_DIR / "wholesale_profile.csv")
        print(f"\n  VIP/опт: {len(ws)} клиентов, "
              f"Monetary_median={ws['Monetary'].median():.0f}, "
              f"Frequency_median={ws['Frequency'].median():.0f}")

    print("[9/9] Сохранение моделей и пример предсказания")
    Z_best = Z_ext if best_r["space"] == "ext" else Z_rfm
    method = best_r["method"]

    # --- обучаем финальную модель (для сохранения и тайминга) ---
    t0 = time.perf_counter()
    if method == "kmeans":
        from sklearn.cluster import KMeans
        mdl = KMeans(n_clusters=best_r["k"], n_init=10,
                    random_state=best_r["seed"]).fit(Z_best)
    elif method == "own_kmeans":
        mdl = OwnKMeans(n_clusters=best_r["k"], n_init=5,
                        random_state=best_r["seed"]).fit(Z_best)
    elif method.startswith("agg_"):
        from sklearn.cluster import AgglomerativeClustering
        link = method.split("_", 1)[1]
        mdl = AgglomerativeClustering(
            n_clusters=best_r["k"], linkage=link,
            metric="euclidean").fit(Z_best)
    elif method.startswith("gmm_"):
        from sklearn.mixture import GaussianMixture
        cov = method.split("_", 1)[1]
        mdl = GaussianMixture(n_components=best_r["k"],
                              covariance_type=cov,
                              random_state=RANDOM_STATE,
                              n_init=3).fit(Z_best)
    elif method == "dbscan":
        from sklearn.cluster import DBSCAN
        mdl = DBSCAN(eps=best_r["eps"],
                     min_samples=best_r["min_samples"]).fit(Z_best)
    else:
        raise RuntimeError(f"unknown method: {method}")
    t_fit = time.perf_counter() - t0

    # --- predict (если есть) ---
    t_pred = float("nan")
    if hasattr(mdl, "predict"):
        t0 = time.perf_counter()
        _ = mdl.predict(Z_best)
        t_pred = time.perf_counter() - t0

    # --- сохраняем модель + конвейер-метаданные ---
    joblib.dump({
        "model": mdl,
        "method": method,
        "space": best_r["space"],
        "params": {k: v for k, v in best_r.items()
                   if k in ("k", "min_samples", "eps", "cov_type", "seed")},
        "labels": best_r["labels"],
    }, OUT_DIR / "best_solution.joblib")

    # --- тайминги ---
    pd.DataFrame([{
        "method": method, "space": best_r["space"],
        "time_fit_sec": t_fit, "time_predict_sec": t_pred,
        "n_objects": int(len(Z_best)),
    }]).to_csv(OUT_DIR / "timing_best.csv", index=False)
    print(f"  время: fit={t_fit:.4f}s, "
          f"predict={t_pred if t_pred == t_pred else 'n/a'}")

    # --- пример предсказания (использует сохранённую модель) ---
    print(f"Используйте пример предсказания с помощью: predict_example_b.py")

    print(f"\nЖурнал:        {LOG_PATH}")
    print(f"Метрики:       {OUT_DIR / 'clustering_metrics.csv'}")
    print(f"Общие метрики: {OUT_DIR / 'clustering_metrics_common.csv'}")
    print(f"own vs sklearn kmeans: {OUT_DIR / 'own_vs_sklearn_kmeans.csv'}")
    print(f"own vs sklearn pca:    {OUT_DIR / 'own_vs_sklearn_pca.json'}")
    print(f"Устойчивость:  {OUT_DIR / 'stability.csv'}")
    print(f"Профили:       {OUT_DIR / 'segment_profile.csv'}")
    print(f"Проверка выбросов: {OUT_DIR / 'segment_outlier_check.csv'}")
    print(f"PCA:           {OUT_DIR / 'pca_summary.json'}")
    print(f"Тайминги:      {OUT_DIR / 'timing_best.csv'}")
    print(f"VIP/опт:       {OUT_DIR / 'wholesale_customers.csv'}")


if __name__ == "__main__":
    main()