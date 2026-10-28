import numpy as np


class OwnPCA:
    """
    PCA через SVD на центрированных данных.
    Самостоятельно: центрирование, SVD, выбор компонент,
    преобразование, восстановление, расчёт объяснённой дисперсии.
    """
    def __init__(self, n_components=None, variance_target=None):
        self.n_components = n_components
        self.variance_target = variance_target
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self._d = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self._d = d
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        var = (s ** 2) / (n - 1)
        ratio = var / var.sum()

        if self.variance_target is not None:
            cum = np.cumsum(ratio)
            k = int(np.searchsorted(cum, self.variance_target) + 1)
            self.n_components = min(k, d)
        elif self.n_components is None:
            self.n_components = d

        self.components_ = Vt[: self.n_components]
        self.explained_variance_ = var[: self.n_components]
        self.explained_variance_ratio_ = ratio[: self.n_components]
        return self

    def fit_at_k(self, X, k):
        """Явно обучить модель на k компонент — нужно для корректной
        оценки ошибки восстановления при выбранном k."""
        self.n_components = int(k)
        return self.fit(X)

    def transform(self, X):
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Call fit before transform")
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def inverse_transform(self, Z):
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Call fit before inverse_transform")
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xr = self.inverse_transform(self.transform(X))
        return float(((X - Xr) ** 2).mean())