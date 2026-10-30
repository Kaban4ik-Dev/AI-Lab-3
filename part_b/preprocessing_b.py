import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_SETS = {
    "rfm": ["Recency", "Frequency", "Monetary"],
    "ext": ["Recency", "Frequency", "Monetary", "n_products", "avg_basket"],
}

# признаки, к которым применяется log1p (неотрицательные, тяжёлый хвост)
LOG_COLS = ("Frequency", "Monetary", "avg_basket", "n_products")


def compute_clip_bounds(X, quantile):
    """
    Пороги winsorize по каждому столбцу.
    Сохраняются в preprocessor.joblib, чтобы применять к новым данным
    без переобучения (требование воспроизводимости 5.1.3/5.1.5).
    """
    if quantile is None:
        return {c: float("inf") for c in X.columns}
    return {c: float(X[c].quantile(quantile)) for c in X.columns}


def build_feature_matrix(rfm, which="ext", log_transform=True,
                         winsorize_quantile=None, clip_bounds=None):
    """
    Возвращает (X, cols).
    - clip_bounds: dict {col: hi}; если задан — используем готовые пороги
      (для применения к новым данным без переобучения).
    - winsorize_quantile: если задан и clip_bounds=None — считаем пороги
      на переданной таблице (только для обучающей части!).
    - log_transform: log1p к LOG_COLS.
    """
    cols = FEATURE_SETS[which]
    X = rfm[cols].astype(float).copy()

    if clip_bounds is not None:
        for c in cols:
            X[c] = np.minimum(X[c], clip_bounds[c])
    elif winsorize_quantile is not None:
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