import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

def describe_numeric(df, num_cols, path_prefix):
    rows = []
    for c in num_cols:
        s = df[c]
        rows.append({
            "feature": c,
            "n": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "min": s.min(), "max": s.max(),
            "mean": s.mean(), "median": s.median(),
            "var": s.var(), "std": s.std(),
            "q1": s.quantile(.25), "q3": s.quantile(.75),
            "iqr": s.quantile(.75) - s.quantile(.25),
            "skew": skew(s.dropna()),
            "kurtosis_fisher": kurtosis(s.dropna(), fisher=True),  # явно указано определение
        })
    out = pd.DataFrame(rows)
    out.to_csv(f"{path_prefix}/numeric_stats.csv", index=False)
    return out

def describe_categorical(df, cat_cols, path_prefix):
    rows = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False)
        for val, cnt in vc.items():
            rows.append({"feature": c, "value": val, "count": int(cnt),
                         "fraction": cnt / len(df)})
    out = pd.DataFrame(rows)
    out.to_csv(f"{path_prefix}/categorical_stats.csv", index=False)
    return out

def plot_histograms(df, num_cols, path_prefix):
    n = len(num_cols)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(15, 4 * ((n + 2) // 3)))
    for ax, c in zip(axes.flat, num_cols):
        ax.hist(df[c].dropna(), bins=50)
        ax.set_title(c); ax.set_xlabel(c); ax.set_ylabel("частота")
    for ax in axes.flat[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{path_prefix}/histograms.png", dpi=120); plt.close()

def plot_boxplots(df, num_cols, path_prefix):
    n = len(num_cols)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(15, 4 * ((n + 2) // 3)))
    for ax, c in zip(axes.flat, num_cols):
        ax.boxplot(df[c].dropna(), vert=True)
        ax.set_title(c); ax.set_ylabel(c)
    for ax in axes.flat[n:]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(f"{path_prefix}/boxplots.png", dpi=120); plt.close()

def plot_categorical(df, cat_cols, path_prefix):
    fig, axes = plt.subplots(len(cat_cols), 1, figsize=(10, 3 * len(cat_cols)))
    for ax, c in zip(axes, cat_cols):
        df[c].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(c); ax.set_ylabel("частота")
    plt.tight_layout(); plt.savefig(f"{path_prefix}/categorical_bars.png", dpi=120); plt.close()

def plot_correlations(df, num_cols, path_prefix):
    pear = df[num_cols + ["y"]].corr(method="pearson")
    spear = df[num_cols + ["y"]].corr(method="spearman")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(pear, ax=axes[0], cmap="coolwarm", center=0)
    axes[0].set_title("Pearson")
    sns.heatmap(spear, ax=axes[1], cmap="coolwarm", center=0)
    axes[1].set_title("Spearman")
    plt.tight_layout(); plt.savefig(f"{path_prefix}/correlations.png", dpi=120); plt.close()
    return pear, spear

def target_by_class(df, num_cols, path_prefix):
    """Сравнение признаков между классами."""
    fig, axes = plt.subplots((len(num_cols) + 2) // 3, 3,
                             figsize=(15, 4 * ((len(num_cols) + 2) // 3)))
    for ax, c in zip(axes.flat, num_cols):
        df.boxplot(column=c, by="y", ax=ax)
        ax.set_title(c); ax.set_ylabel(c)
    for ax in axes.flat[len(num_cols):]:
        ax.axis("off")
    plt.suptitle("")
    plt.tight_layout(); plt.savefig(f"{path_prefix}/target_by_class.png", dpi=120); plt.close()

def eda_report(df, path_prefix):
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "y"]
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    describe_numeric(df, num_cols, path_prefix)
    describe_categorical(df, cat_cols, path_prefix)
    plot_histograms(df, num_cols, path_prefix)
    plot_boxplots(df, num_cols, path_prefix)
    plot_categorical(df, cat_cols, path_prefix)
    pear, spear = plot_correlations(df, num_cols, path_prefix)
    target_by_class(df, num_cols, path_prefix)
    return pear, spear