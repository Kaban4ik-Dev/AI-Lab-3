import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_SETS = {
    "rfm": ["Recency", "Frequency", "Monetary"],
    "ext": ["Recency", "Frequency", "Monetary", "n_products", "avg_basket"],
}

# признаки, к которым применяется log1p (неотрицательные, тяжёлый хвост)
LOG_COLS = ("Frequency", "Monetary", "avg_basket", "n_products")


def build_feature_matrix(rfm, which="ext", log_transform=True,
                         winsorize_quantile=None):
    """
    Возвращает (X, cols).
    - winsorize_quantile: если задан, обрезаем сверху по квантилю (например, 0.99)
    - log_transform: применяем log1p к LOG_COLS
    """
    cols = FEATURE_SETS[which]
    X = rfm[cols].astype(float).copy()
    if winsorize_quantile is not None:
        for c in cols:
            hi = X[c].quantile(winsorize_quantile)
            X[c] = np.minimum(X[c], hi)
    if log_transform:
        for c in cols:
            if c in LOG_COLS:
                X[c] = np.log1p(X[c])
    return X, cols


def scale(X_train, X_val=None):
    sc = StandardScaler()
    Z_train = sc.fit_transform(X_train)
    if X_val is not None:
        return Z_train, sc.transform(X_val), sc
    return Z_train, None, sc