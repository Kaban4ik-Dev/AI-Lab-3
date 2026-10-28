import numpy as np
from own_logreg import OwnLogisticRegression

def finite_difference_check(seed=0):
    rng = np.random.default_rng(seed)
    n, d = 8, 5
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 2, size=n).astype(float)
    model = OwnLogisticRegression(C=1.0)
    model.w = rng.normal(size=d)
    model.b = rng.normal()
    loss, gw, gb = model._loss_and_grads(X, y)

    eps = 1e-6
    # численные градиенты по w
    num_gw = np.zeros_like(model.w)
    for i in range(d):
        old = model.w[i]
        model.w[i] = old + eps
        l1, _, _ = model._loss_and_grads(X, y)
        model.w[i] = old - eps
        l2, _, _ = model._loss_and_grads(X, y)
        model.w[i] = old
        num_gw[i] = (l1 - l2) / (2 * eps)
    # численный градиент по b
    old = model.b
    model.b = old + eps; l1, _, _ = model._loss_and_grads(X, y)
    model.b = old - eps; l2, _, _ = model._loss_and_grads(X, y)
    model.b = old
    num_gb = (l1 - l2) / (2 * eps)

    err_w = np.max(np.abs(gw - num_gw))
    err_b = abs(gb - num_gb)
    print(f"grad check: max|Δw|={err_w:.3e}, |Δb|={err_b:.3e}")
    assert err_w < 1e-5 and err_b < 1e-5, "Градиенты не совпали!"

def edge_cases():
    # одинаковые объекты
    X = np.ones((4, 3)); y = np.array([1, 1, 0, 0])
    m = OwnLogisticRegression(C=1.0, lr=0.1, max_iter=200).fit(X, y)
    # постоянный признак
    Xc = np.column_stack([np.ones(20), np.zeros(20)])
    yc = np.r_[np.ones(10), np.zeros(10)]
    mc = OwnLogisticRegression(C=10.0, lr=0.5, max_iter=500).fit(Xc, yc)
    p = mc.predict_proba(Xc)[:, 1]
    print("edge cases OK; prob range:", p.min(), p.max())

def sanity_known_answer():
    """
    На тривиально разделимых данных доля правильных ответов должна быть 1.0.
    """
    X = np.vstack([np.random.randn(50, 2) + 2,
                   np.random.randn(50, 2) - 2])
    y = np.r_[np.ones(50), np.zeros(50)]
    m = OwnLogisticRegression(C=10.0, lr=0.5, max_iter=2000).fit(X, y)
    acc = (m.predict(X) == y).mean()
    print(f"sanity accuracy (linear separable): {acc:.3f}")
    assert acc > 0.98

if __name__ == "__main__":
    finite_difference_check()
    edge_cases()
    sanity_known_answer()
    print("Все проверки собственной реализации пройдены.")