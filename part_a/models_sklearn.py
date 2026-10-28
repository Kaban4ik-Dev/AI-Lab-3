import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import average_precision_score
from own_logreg import OwnLogisticRegression
from typing import Optional, Any
from sklearn.compose import ColumnTransformer

def make_preprocessor(num_cols, cat_cols):
    """Все преобразования обучаются ТОЛЬКО на train внутри Pipeline."""
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

def build_model(name, params):
    if name == "logreg":
        return LogisticRegression(max_iter=2000, **params)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=-1, **params)
    if name == "svm":
        return SVC(**params)  # decision_function для ранжирующих метрик
    if name == "tree":
        return DecisionTreeClassifier(random_state=42, **params)
    if name == "rf":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    if name == "gb":
        return HistGradientBoostingClassifier(random_state=42, **params)
    if name == "nb":
        return GaussianNB(**params)
    if name == "own_logreg":
        return OwnLogisticRegression(**params)
    raise ValueError(name)

def get_scores(model, X):
    # ранжирующий скор: probability, decision_function или собственный proba
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)

class OwnLogregAdapter:
    def __init__(self, **kw):
        self.kw = kw
        self.pre: Optional[ColumnTransformer] = None
        self.model: Optional[OwnLogisticRegression] = None

    def fit(self, pre, X, y):
        self.pre = pre
        assert self.pre is not None
        Xt = self.pre.fit_transform(X)
        self.model = OwnLogisticRegression(**self.kw).fit(Xt, y)
        return self

    def score(self, X):
        assert self.model is not None and self.pre is not None
        return self.model.predict_proba(self.pre.transform(X))[:, 1]

def search_family(name, df_train, df_val, num_cols, cat_cols, grid, log):
    """Перебор по сетке для одной семьи на одном фолде/одной паре train/val."""
    best = {"AP": -np.inf, "params": None, "model": None}
    results = []
    for params in ParameterGrid(grid):
        t0 = time.time()
        if name == "own_logreg":
            pre = make_preprocessor(num_cols, cat_cols)
            adapter = OwnLogregAdapter(**params).fit(pre, df_train[num_cols + cat_cols], df_train["y"])
            scores = adapter.score(df_val[num_cols + cat_cols])
            model_obj = adapter
        else:
            pre = make_preprocessor(num_cols, cat_cols)
            model = build_model(name, params)
            pipe = Pipeline([("pre", pre), ("model", model)])
            pipe.fit(df_train[num_cols + cat_cols], df_train["y"])
            scores = get_scores(pipe, df_val[num_cols + cat_cols])
            model_obj = pipe
        elapsed = time.time() - t0
        ap = average_precision_score(df_val["y"], scores)
        results.append({"family": name, "params": params, "AP": ap, "time_s": elapsed})
        log.append({**results[-1]})
        if ap > best["AP"]:
            best = {"AP": ap, "params": params, "model": model_obj}
    return best, pd.DataFrame(results)