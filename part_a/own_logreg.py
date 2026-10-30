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
        self.loss_history = []
        loss, gw, gb = self._loss_and_grads(X, y)
        self.loss_history.append(loss)

        for it in range(self.max_iter):
            # backtracking: уменьшаем lr, пока loss не уменьшится
            lr = self.lr
            improved = False
            for _ in range(30):
                w_new = self.w - lr * gw
                b_new = self.b - lr * gb
                if not (np.all(np.isfinite(w_new)) and np.isfinite(b_new)):
                    lr *= 0.5
                    continue
                # временно подставляем и считаем loss
                old_w, old_b = self.w, self.b
                self.w, self.b = w_new, b_new
                new_loss, gw_new, gb_new = self._loss_and_grads(X, y)
                self.w, self.b = old_w, old_b
                if np.isfinite(new_loss) and new_loss < loss:
                    self.w, self.b = w_new, b_new
                    gw, gb = gw_new, gb_new
                    improved = True
                    break
                lr *= 0.5
            if not improved:
                break
            prev_loss = loss
            loss = new_loss
            self.loss_history.append(loss)
            if abs(prev_loss - loss) < self.tol:
                break
        return self

    def predict_proba(self, X):
        assert self.w is not None and self.b is not None
        z = np.asarray(X) @ self.w + self.b
        p = self._sigmoid(z)
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)