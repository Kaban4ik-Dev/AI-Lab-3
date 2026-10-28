import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

def describe_numeric(df, cols, path_prefix):
    rows = []
    for c in cols:
        s = df[c].astype(float)
        rows.append({
            "feature": c, "n": int(s.notna().sum()), "missing": int(s.isna().sum()),
            "min": s.min(), "max": s.max(), "mean": s.mean(), "median": s.median(),
            "var": s.var(), "std": s.std(),
            "q1": s.quantile(.25), "q3": s.quantile(.75),
            "iqr": s.quantile(.75) - s.quantile(.25),
            "skew": skew(s.dropna()),
            "kurtosis_fisher": kurtosis(s.dropna(), fisher=True),
        })
    pd.DataFrame(rows).to_csv(f"{path_prefix}/numeric_stats_b.csv", index=False)
    return rows

def plot_histograms(df, cols, path_prefix):
    n = len(cols)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(15, 4 * ((n + 2) // 3)))
    for ax, c in zip(axes.flat, cols):
        ax.hist(df[c].dropna(), bins=60)
        ax.set_title(c); ax.set_xlabel(c); ax.set_ylabel("частота")
    for ax in axes.flat[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{path_prefix}/hist_b.png", dpi=120); plt.close()

def plot_correlations(df, cols, path_prefix):
    pear = df[cols].corr("pearson")
    spear = df[cols].corr("spearman")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(pear, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0])
    axes[0].set_title("Pearson")
    sns.heatmap(spear, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1])
    axes[1].set_title("Spearman")
    plt.tight_layout(); plt.savefig(f"{path_prefix}/corr_b.png", dpi=120); plt.close()
    return pear, spear

def eda_report(rfm, path_prefix):
    cols = ["Recency", "Frequency", "Monetary", "n_products", "avg_basket"]
    describe_numeric(rfm, cols, path_prefix)
    plot_histograms(rfm, cols, path_prefix)
    plot_correlations(rfm, cols, path_prefix)