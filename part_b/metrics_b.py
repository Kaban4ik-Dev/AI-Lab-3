import numpy as np
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

def internal_metrics(Z, labels):
    mask = labels != -1   # для DBSCAN шум исключаем
    Zf = Z[mask]; lf = labels[mask]
    n_labels = len(np.unique(lf))
    res = {"noise_frac": float(1 - mask.mean()),
           "n_clusters": int(n_labels)}
    if n_labels < 2 or n_labels >= len(Zf):
        res.update({"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None})
        return res
    res["silhouette"] = float(silhouette_score(Zf, lf))
    res["davies_bouldin"] = float(davies_bouldin_score(Zf, lf))
    res["calinski_harabasz"] = float(calinski_harabasz_score(Zf, lf))
    return res

def adjusted_rand(labels_a, labels_b):
    return float(adjusted_rand_score(labels_a, labels_b))

def cluster_sizes(labels):
    vals, counts = np.unique(labels, return_counts=True)
    return dict(zip(map(int, vals), map(int, counts)))