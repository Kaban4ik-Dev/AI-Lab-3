import numpy as np


class OwnKMeans:
    """
    K-means (Lloyd) с инициализацией k-means++.
    Поддерживает внешние стартовые центры (для строгого сравнения с sklearn
    при одинаковой инициализации и n_init = 1).
    """

    def __init__(self, n_clusters=3, n_init=1, max_iter=300, tol=1e-6,
                 random_state=42, init="kmeans++"):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        # внешние стартовые центры (опционально)
        self._init_centers_override = None

    @staticmethod
    def _sq_dist(X, C):
        return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)

    def _init_centers(self, X, rng):
        # если заданы внешние стартовые центры — используем их
        if self._init_centers_override is not None:
            C = np.asarray(self._init_centers_override, dtype=np.float64)
            if C.shape[0] != min(self.n_clusters, len(X)):
                raise ValueError(
                    "init_centers_override shape mismatch with n_clusters")
            return C.copy()

        n, d = X.shape
        k = min(self.n_clusters, n)

        if self.init == "random":
            idx = rng.choice(n, size=k, replace=False)
            return X[idx].copy()

        # k-means++
        centers = np.empty((k, d), dtype=X.dtype)
        first = int(rng.integers(0, n))
        centers[0] = X[first]
        closest = self._sq_dist(X, centers[0:1]).ravel()

        for j in range(1, k):
            s = closest.sum()
            if s <= 0:
                unchosen = np.setdiff1d(
                    np.arange(n),
                    np.where(np.any(np.all(
                        X[:, None, :] == centers[None, :j, :], axis=2),
                        axis=1))[0])
                if len(unchosen) == 0:
                    unchosen = np.arange(n)
                idx = int(rng.choice(unchosen))
            else:
                probs = closest / s
                idx = int(rng.choice(n, p=probs))
            centers[j] = X[idx]
            d_new = self._sq_dist(X, centers[j:j+1]).ravel()
            closest = np.minimum(closest, d_new)
        return centers

    def _lloyd(self, X, rng):
        n = len(X)
        centers = self._init_centers(X, rng)
        k = len(centers)
        labels = np.zeros(n, dtype=int)
        prev_inertia = np.inf
        n_iter = 0
        inertia = float(np.inf)

        for it in range(self.max_iter):
            n_iter = it + 1
            d = self._sq_dist(X, centers)
            labels = d.argmin(axis=1)

            new_centers = centers.copy()
            for c in range(k):
                mask = labels == c
                if not mask.any():
                    # пустой кластер: точка, максимально удалённая от центра
                    min_d = d[np.arange(n), labels]
                    far_idx = int(min_d.argmax())
                    new_centers[c] = X[far_idx]
                    labels[far_idx] = c
                else:
                    new_centers[c] = X[mask].mean(axis=0)

            inertia = float(d[np.arange(n), labels].sum())
            centers = new_centers
            if np.isfinite(prev_inertia) and abs(prev_inertia - inertia) < self.tol:
                break
            prev_inertia = inertia

        return centers, labels, inertia, n_iter

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf")
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            c, l, i, nit = self._lloyd(X, rng)
            if best is None or i < best[2]:
                best = (c, l, i, nit)
        assert best is not None
        self.centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._sq_dist(X, self.centers_).argmin(axis=1)