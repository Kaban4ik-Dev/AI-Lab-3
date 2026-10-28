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
                        "labels": km.labels_, "inertia": km.inertia_})
    return out

def run_agglomerative(Z, ks, linkages):
    out = []
    for k in ks:
        for link in linkages:
            if link == "ward":
                # ward только с euclidean — sklearn подразумевает euclidean
                model = AgglomerativeClustering(n_clusters=k, linkage=link)
            else:
                model = AgglomerativeClustering(n_clusters=k, linkage=link)
            labels = model.fit_predict(Z)
            out.append({"method": f"agg_{link}", "k": k, "labels": labels})
    return out

def run_dbscan(Z, min_samples_list, eps_quantiles):
    # eps задаём через квантили k-расстояний (k = min_samples)
    results = []
    for ms in min_samples_list:
        nbrs = NearestNeighbors(n_neighbors=ms).fit(Z)
        dists, _ = nbrs.kneighbors(Z)
        kdist = np.sort(dists[:, -1])[::-1]
        eps_values = np.quantile(kdist, eps_quantiles)
        for eps in eps_values:
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
                    "labels": km.labels_, "inertia": km.inertia_})
    return out