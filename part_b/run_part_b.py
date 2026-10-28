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
                    STABILITY_N_SUBSAMPLES, STABILITY_FRACTION, STABILITY_SEED)
from data_utils_b import (sha256_of, load_online_retail, data_passport_raw,
                          clean_transactions, build_customer_table, customer_passport)
from eda_b import eda_report
from preprocessing_b import build_feature_matrix, scale
from own_pca import OwnPCA
from clustering_b import (run_kmeans, run_agglomerative, run_dbscan,
                          run_gmm, run_own_kmeans)
from metrics_b import internal_metrics, adjusted_rand, cluster_sizes
from viz_b import (plot_pca_variance, plot_kdistance, plot_tsne)

warnings.filterwarnings("ignore")

LOG_PATH = OUT_DIR / "runs_b.csv"
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
log_rows = []


def log(**kw):
    kw["run_id"] = RUN_ID
    log_rows.append(kw)
    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)


def load_or_build_customers():
    cache_parquet = OUT_DIR / "customers.parquet"
    cache_csv = OUT_DIR / "customers.csv"
    # кэш считаем валидным, только если в нём есть is_wholesale
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
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"  raw: {df.shape}, отмен={raw_pass['canceled_invoices']}, "
          f"без CustomerID={raw_pass['missing_customer']}")

    dfc = clean_transactions(df)
    print(f"  clean: {dfc.shape}")

    rfm, snapshot = build_customer_table(dfc, wholesale_quantile=WHOLESALE_QUANTILE)
    print(f"  клиентов: {len(rfm)}, VIP/опт: {int(rfm['is_wholesale'].sum())}, "
          f"дата среза: {snapshot.date()}")

    (OUT_DIR / "passport_customers.json").write_text(
        json.dumps(customer_passport(rfm), indent=2, ensure_ascii=False),
        encoding="utf-8")

    # пытаемся parquet, если нет движка — падаем в csv
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


def main():
    print("[1/8] Загрузка и агрегация")
    rfm_all = load_or_build_customers()
    print(f"  форма rfm: {rfm_all.shape}")

    # отделяем VIP/опт от основной массы
    rfm = rfm_all[rfm_all["is_wholesale"] == 0].reset_index(drop=True)
    rfm_wholesale = rfm_all[rfm_all["is_wholesale"] == 1].reset_index(drop=True)
    print(f"  основная масса: {len(rfm)}, VIP/опт: {len(rfm_wholesale)}")

    print("[2/8] EDA по RFM (основная масса)")
    eda_report(rfm, str(OUT_DIR))

    # VIP/опт — отдельная таблица в артефактах
    if len(rfm_wholesale):
        rfm_wholesale.to_csv(OUT_DIR / "wholesale_customers.csv", index=False)

    print("[3/8] Признаки: RFM vs расширенный (winsorize + log1p)")
    X_rfm, cols_rfm = build_feature_matrix(rfm, which="rfm",
                                           log_transform=True,
                                           winsorize_quantile=WINSORIZE_QUANTILE)
    X_ext, cols_ext = build_feature_matrix(rfm, which="ext",
                                           log_transform=True,
                                           winsorize_quantile=WINSORIZE_QUANTILE)

    Z_rfm, _, sc_rfm = scale(X_rfm)
    Z_ext, _, sc_ext = scale(X_ext)
    np.save(OUT_DIR / "Z_rfm.npy", Z_rfm)
    np.save(OUT_DIR / "Z_ext.npy", Z_ext)
    joblib.dump({"sc_rfm": sc_rfm, "sc_ext": sc_ext,
                 "cols_rfm": cols_rfm, "cols_ext": cols_ext,
                 "winsorize_quantile": WINSORIZE_QUANTILE},
                OUT_DIR / "preprocessor.joblib")

    print("[4/8] Кластеризация на двух пространствах")
    all_results = []
    for which, Z in [("rfm", Z_rfm), ("ext", Z_ext)]:
        for r in run_kmeans(Z, KMEANS_KS, KMEANS_N_INIT, RANDOM_STATE):
            r["_Z"] = Z; r["space"] = which; all_results.append(r)
        for r in run_own_kmeans(Z, KMEANS_KS, RANDOM_STATE):
            r["_Z"] = Z; r["space"] = which; all_results.append(r)
        for r in run_agglomerative(Z, AGG_KS, AGG_LINKAGES):
            r["_Z"] = Z; r["space"] = which; all_results.append(r)
        for r in run_dbscan(Z, DBSCAN_MIN_SAMPLES, DBSCAN_EPS_QUANTILES):
            r["_Z"] = Z; r["space"] = which; all_results.append(r)
        for r in run_gmm(Z, GMM_KS, GMM_COV_TYPES):
            r["_Z"] = Z; r["space"] = which; all_results.append(r)

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
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "clustering_metrics.csv", index=False)
    print("  все результаты сохранены в clustering_metrics.csv")

    # -- Отбор лучших по silhouette среди «многокластерных»,
    #    но не считаем таковыми результаты, где >90% в одном кластере.
    valid = []
    for r in all_results:
        uniq = np.unique(r["labels"][r["labels"] != -1])
        if len(uniq) < 2:
            continue
        sizes = pd.Series(r["labels"]).value_counts(normalize=True)
        if sizes.iloc[0] > 0.90:   # один кластер > 90% — не сегментация
            continue
        valid.append(r)

    if not valid:
        print("  [!] нет результатов с осмысленным разделением (>2 кластеров, без >90% доминанты)")
        return

    best_r, best_m = pick_best_for_metric(valid, "silhouette")
    if best_r is None or best_m is None:
        print("  [!] не удалось выбрать лучшую модель по silhouette")
        return
    print(f"  лучший по silhouette: {best_r['method']} k={best_r.get('k')} "
          f"sil={best_m['silhouette']:.4f} на space={best_r['space']}")

    print("[5/8] Устойчивость двух лучших (5 × 80%)")
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

    stab_rows = []
    rng = np.random.default_rng(STABILITY_SEED)
    n = len(rfm)
    for r, m in top2:
        Z = r["_Z"]; labels_full = r["labels"]
        aris = []
        for i in range(STABILITY_N_SUBSAMPLES):
            idx = rng.choice(n, size=int(STABILITY_FRACTION * n), replace=False)
            Zsub = Z[idx]
            method = r["method"]
            try:
                if method == "kmeans":
                    from sklearn.cluster import KMeans
                    lsub = KMeans(n_clusters=r["k"], n_init=10,
                                  random_state=STABILITY_SEED + i).fit_predict(Zsub)
                elif method.startswith("agg_"):
                    from sklearn.cluster import AgglomerativeClustering
                    link = method.split("_", 1)[1]
                    lsub = AgglomerativeClustering(n_clusters=r["k"],
                                                   linkage=link).fit_predict(Zsub)
                elif method.startswith("gmm_"):
                    from sklearn.mixture import GaussianMixture
                    cov = method.split("_", 1)[1]
                    lsub = GaussianMixture(n_components=r["k"],
                                           covariance_type=cov,
                                           random_state=STABILITY_SEED + i,
                                           n_init=3).fit_predict(Zsub)
                elif method == "dbscan":
                    from sklearn.cluster import DBSCAN
                    lsub = DBSCAN(eps=r["eps"], min_samples=r["min_samples"]
                                  ).fit_predict(Zsub)
                else:
                    continue
                ari = adjusted_rand(labels_full[idx], lsub)
                aris.append(ari)
            except Exception:
                continue
        if aris:
            stab_rows.append({"method": r["method"], "k": r.get("k"),
                              "space": r["space"],
                              "mean_ARI": float(np.mean(aris)),
                              "std_ARI": float(np.std(aris)),
                              "n_runs": len(aris)})
    pd.DataFrame(stab_rows).to_csv(OUT_DIR / "stability.csv", index=False)
    for row in stab_rows:
        print(f"  {row['method']}: mean ARI={row['mean_ARI']:.3f} ± {row['std_ARI']:.3f}")

    print("[6/8] PCA (own) и t-SNE")
    Z = Z_ext
    # полный PCA — для кривой объяснённой дисперсии
    pca_full = OwnPCA().fit(Z)
    assert pca_full.explained_variance_ratio_ is not None
    plot_pca_variance(pca_full.explained_variance_ratio_, OUT_DIR / "pca_variance.png")
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    k95 = int(np.searchsorted(cum, 0.95) + 1)

    # корректная ошибка восстановления для k90 и k95
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
                title = f"t-SNE: {best_r['method']} (perp={perp}, init={init})"
                fn = f"tsne_{best_r['method']}_p{perp}_{init}.png"
                plot_tsne(Z, labels_for_tsne, title, OUT_DIR / fn,
                          perplexity=perp, init=init)
            except Exception as e:
                print(f"  [t-SNE {perp}/{init}] {type(e).__name__}: {e}")

    print("[7/8] Профили сегментов лучшей модели")
    labels = best_r["labels"]
    prof = rfm.copy()
    prof["cluster"] = labels
    # исключаем шум (-1), если это DBSCAN
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

    # отдельно сохраняем описание VIP/опт
    if len(rfm_wholesale):
        ws = rfm_wholesale[["CustomerID", "Recency", "Frequency",
                            "Monetary", "n_products", "avg_basket"]]
        ws.describe().to_csv(OUT_DIR / "wholesale_profile.csv")
        print(f"\n  VIP/опт: {len(ws)} клиентов, "
              f"Monetary_median={ws['Monetary'].median():.0f}, "
              f"Frequency_median={ws['Frequency'].median():.0f}")

    print("[8/8] Сохранение моделей")
    joblib.dump({"labels": best_r["labels"],
                 "method": best_r["method"],
                 "space": best_r["space"],
                 "params": {k: v for k, v in best_r.items()
                            if k in ("k", "min_samples", "eps")}},
                OUT_DIR / "best_solution.joblib")

    print(f"\nЖурнал:        {LOG_PATH}")
    print(f"Метрики:       {OUT_DIR / 'clustering_metrics.csv'}")
    print(f"Устойчивость:  {OUT_DIR / 'stability.csv'}")
    print(f"Профили:       {OUT_DIR / 'segment_profile.csv'}")
    print(f"PCA:           {OUT_DIR / 'pca_summary.json'}")
    print(f"VIP/опт:       {OUT_DIR / 'wholesale_customers.csv'}")


if __name__ == "__main__":
    main()