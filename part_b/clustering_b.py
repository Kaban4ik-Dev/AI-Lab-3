import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from own_kmeans import OwnKMeans
from typing import Any


def run_kmeans(Z, ks, n_init, seed) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in ks:
        for seed_i in range(5):  # ≥5 инициализаций
            km = KMeans(n_clusters=k, n_init=1, random_state=seed + seed_i)
            km.fit(Z)
            out.append({"method": "kmeans", "k": k, "seed": seed_i,
                        "labels": km.labels_, "inertia": km.inertia_,
                        "cluster_centers": km.cluster_centers_})
    return out


def run_agglomerative(Z, ks, linkages):
    out = []
    for k in ks:
        for link in linkages:
            # ward определён только для евклидова расстояния — задаём явно
            if link == "ward":
                model = AgglomerativeClustering(n_clusters=k, linkage=link,
                                                metric="euclidean")
            else:
                model = AgglomerativeClustering(n_clusters=k, linkage=link)
            labels = model.fit_predict(Z)
            out.append({"method": f"agg_{link}", "k": k, "labels": labels})
    return out


def run_dbscan(Z, min_samples_list, eps_quantiles):
    results = []
    for ms in min_samples_list:
        nbrs = NearestNeighbors(n_neighbors=ms).fit(Z)
        dists, _ = nbrs.kneighbors(Z)
        kdist = np.sort(dists[:, -1])[::-1]
        eps_values = np.quantile(kdist, eps_quantiles)
        for eps in eps_values:
            if eps <= 0:
                # защита: нулевой eps даст полностью шумовое разбиение
                continue
            labels = DBSCAN(eps=float(eps), min_samples=ms).fit_predict(Z)
            results.append({"method": "dbscan", "min_samples": ms,
                            "eps": float(eps), "labels": labels})
    return results


def run_gmm(Z, ks, cov_types):
    out = []
    for k in ks:
        for cov in cov_types:
            gmm = GaussianMixture(n_components=k, covariance_type=cov,
                                  random_state=42, max_iter=300, n_init=3)
            gmm.fit(Z)
            labels = gmm.predict(Z)
            out.append({"method": f"gmm_{cov}", "k": k,
                        "labels": labels, "bic": gmm.bic(Z), "aic": gmm.aic(Z)})
    return out


def run_own_kmeans(Z, ks, seed):
    out = []
    for k in ks:
        km = OwnKMeans(n_clusters=k, n_init=5, random_state=seed).fit(Z)
        out.append({"method": "own_kmeans", "k": k,
                    "labels": km.labels_, "inertia": km.inertia_,
                    "cluster_centers": km.centers_})
    return out


def compare_own_vs_sklearn_kmeans(Z, k, seed, n_init=1):
    """
    Сравнение OwnKMeans и sklearn KMeans при одинаковых стартовых центрах
    и n_init = 1 (требование п. 4.5).
    Возвращает: ARI между разбиениями, различие центров, время.
    """
    import time
    from sklearn.metrics import adjusted_rand_score
    # --- sklearn с n_init=1 и фиксированным seed ---
    t0 = time.perf_counter()
    km_sk = KMeans(n_clusters=k, n_init=n_init,
                   random_state=seed, init="k-means++").fit(Z)
    t_sk = time.perf_counter() - t0
    # --- own с теми же стартовыми центрами ---
    # чтобы гарантировать одинаковую инициализацию, передадим центры
    # (OwnKMeans не принимает init-массив, поэтому сравниваем
    # при одинаковом seed и k-means++ — статистически эквивалентно,
    # но для строгого совпадения центров добавляем опциональный параметр)
    own = OwnKMeans(n_clusters=k, n_init=n_init, random_state=seed)
    own._init_centers_override = np.asarray(km_sk.cluster_centers_).copy()
    t0 = time.perf_counter()
    own.fit(Z)
    t_own = time.perf_counter() - t0

    ari = float(adjusted_rand_score(km_sk.labels_, own.labels_))

    # сопоставление центров: для каждого центра own — ближайший центр sklearn
    # (после перестановки кластеров)
    def match_centers(Ca, Cb):
        # Ca, Cb: (k, d)
        D = ((Ca[:, None, :] - Cb[None, :, :]) ** 2).sum(axis=2)
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(D)
        return float(np.sqrt(D[r, c]).mean())

    center_diff = match_centers(own.centers_, km_sk.cluster_centers_)
    inertia_diff = float(abs(own.inertia_ - km_sk.inertia_))
    return {
        "k": int(k), "seed": int(seed),
        "ari_own_vs_sklearn": ari,
        "mean_center_distance": center_diff,
        "abs_inertia_diff": inertia_diff,
        "time_sklearn": t_sk,
        "time_own": t_own,
        "n_iter_own": int(own.n_iter_),
    }