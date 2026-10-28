import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as SkPCA

def plot_pca_variance(explained_ratio, path):
    cum = np.cumsum(explained_ratio)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cum) + 1), cum, marker="o")
    plt.axhline(0.90, ls="--", c="gray", label="90%")
    plt.axhline(0.95, ls="--", c="red", label="95%")
    plt.xlabel("число компонент"); plt.ylabel("накопленная объяснённая дисперсия")
    plt.title("PCA: explained variance")
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def plot_kdistance(Z, min_samples, path):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(Z)
    dists, _ = nbrs.kneighbors(Z)
    kdist = np.sort(dists[:, -1])[::-1]
    plt.figure(figsize=(7, 4))
    plt.plot(kdist)
    plt.xlabel("точки (отсортированы по убыванию)"); plt.ylabel(f"{min_samples}-NN расстояние")
    plt.title("k-distance graph для выбора eps")
    plt.grid(True); plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()

def plot_tsne(Z, labels, title, path, perplexity=30, init="pca", seed=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, init=init,
                random_state=seed, learning_rate="auto")
    emb = tsne.fit_transform(Z)
    plt.figure(figsize=(8, 6))
    for lab in np.unique(labels):
        m = labels == lab
        plt.scatter(emb[m, 0], emb[m, 1], s=6, label=f"cluster {lab}" if lab != -1 else "noise")
    plt.legend(markerscale=2, fontsize=8)
    plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    return emb