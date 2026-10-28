import numpy as np

class OwnLogisticRegression:
    """
    Логистическая регрессия, обученная градиентным спуском с L2-регуляризацией.
    Параметры: w (D,), b (скаляр). Сигмоида + бинарная кросс-энтропия.
    L2-регуляризация на весах, НЕ на смещении.
    Критерий остановки: |Δ loss| < tol или max_iter.
    """

    def __init__(self, C=1.0, lr=0.1, max_iter=2000, tol=1e-6, verbose=False):
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.w = None
        self.b = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        # численно устойчивая сигмоида
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        expz = np.exp(z[~pos])
        out[~pos] = expz / (1.0 + expz)
        return out

    def _loss_and_grads(self, X, y):
        assert self.w is not None
        n, d = X.shape
        z = X @ self.w + self.b
        p = self._sigmoid(z)
        eps = 1e-12
        # бинарная кросс-энтропия + L2
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        loss += 0.5 / self.C * np.sum(self.w ** 2)
        # градиенты (вывод: dL/dz = (p - y)/n, dL/dw = X^T (p-y)/n + w/C)
        diff = (p - y) / n
        grad_w = X.T @ diff + self.w / self.C
        grad_b = np.sum(diff)
        return loss, grad_w, grad_b

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        prev = np.inf
        self.loss_history = []
        last_good_w = self.w.copy()
        last_good_b = self.b

        for it in range(self.max_iter):
            loss, gw, gb = self._loss_and_grads(X, y)
            if not np.isfinite(loss):
                # расходимся — откатываемся
                self.w = last_good_w
                self.b = last_good_b
                break
            self.loss_history.append(loss)
            if abs(prev - loss) < self.tol:
                break
            self.w -= self.lr * gw
            self.b -= self.lr * gb
            # запоминаем «хорошие» веса
            if np.all(np.isfinite(self.w)) and np.isfinite(self.b):
                last_good_w = self.w.copy()
                last_good_b = self.b
            prev = loss

        return self

    def predict_proba(self, X):
        assert self.w is not None and self.b is not None
        z = np.asarray(X) @ self.w + self.b
        p = self._sigmoid(z)
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)