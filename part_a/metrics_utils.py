import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support,
                             confusion_matrix, f1_score,
                             roc_curve, precision_recall_curve)

def classification_metrics(y_true, scores, threshold=0.5):
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    out = {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "macro_f1": float(macro_f1),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        # ЯВНО: AP = average_precision_score (не площадь под PR!)
        "AP": float(average_precision_score(y_true, scores)),
        "threshold": float(threshold),
    }
    return out

def find_best_threshold(y_true, scores, metric="f1"):
    """Подбор порога по валидации, НЕ по тесту."""
    if metric == "f1":
        prec, rec, thr = precision_recall_curve(y_true, scores)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        best = np.argmax(f1[:-1])
        return float(thr[best]), float(f1[best])
    raise ValueError(metric)

def plot_roc_pr(y_true, scores, prefix):
    import matplotlib.pyplot as plt
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr); axes[0].plot([0, 1], [0, 1], "--")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC")
    axes[1].plot(rec, prec)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("PR")
    plt.tight_layout(); plt.savefig(f"{prefix}/roc_pr.png", dpi=120); plt.close()

def bootstrap_diff_ap(y_true, scores_a, scores_b, n_boot=500, seed=0):
    """
    Парный бутстрап по объектам: разность AP(A) - AP(B).
    Возвращает mean и 95% CI.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        a = average_precision_score(yt, scores_a[idx])
        b = average_precision_score(yt, scores_b[idx])
        diffs.append(a - b)
    diffs = np.array(diffs)
    return {
        "mean": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_boot_used": int(len(diffs)),
    }