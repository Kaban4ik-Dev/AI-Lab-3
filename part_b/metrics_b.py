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
        res.update({"silhouette": None, "davies_bouldin": None,
                    "calinski_harabasz": None})
        return res
    res["silhouette"] = float(silhouette_score(Zf, lf))
    res["davies_bouldin"] = float(davies_bouldin_score(Zf, lf))
    res["calinski_harabasz"] = float(calinski_harabasz_score(Zf, lf))
    return res


def internal_metrics_on_common_subset(Z, labels, keep_mask):
    """
    Внутренние индексы на общем подмножестве объектов.
    keep_mask: bool-массив длины n; True — объект входит в сравнение
    (для DBSCAN — не шум; для остальных — True везде).
    Так DBSCAN и k-means сравниваются на одних и тех же объектах.
    """
    Zf = Z[keep_mask]
    lf = labels[keep_mask]
    # внутри подмножества могут остаться шумовые метки для DBSCAN
    sub_mask = lf != -1
    Zf2 = Zf[sub_mask]; lf2 = lf[sub_mask]
    n_labels = len(np.unique(lf2))
    res = {"n_clusters_common": int(n_labels)}
    if n_labels < 2 or n_labels >= len(Zf2):
        res.update({"silhouette_common": None,
                    "davies_bouldin_common": None,
                    "calinski_harabasz_common": None})
        return res
    res["silhouette_common"] = float(silhouette_score(Zf2, lf2))
    res["davies_bouldin_common"] = float(davies_bouldin_score(Zf2, lf2))
    res["calinski_harabasz_common"] = float(calinski_harabasz_score(Zf2, lf2))
    return res


def adjusted_rand(labels_a, labels_b):
    return float(adjusted_rand_score(labels_a, labels_b))


def cluster_sizes(labels):
    vals, counts = np.unique(labels, return_counts=True)
    return dict(zip(map(int, vals), map(int, counts)))