import numpy as np
from own_kmeans import OwnKMeans
from own_pca import OwnPCA

def test_kmeans_known():
    # три явно разделимых блоба
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal((0, 0), 0.3, size=(50, 2)),
        rng.normal((5, 5), 0.3, size=(50, 2)),
        rng.normal((-5, 5), 0.3, size=(50, 2)),
    ])
    km = OwnKMeans(n_clusters=3, n_init=1, random_state=0).fit(X)
    # центры — перестановочно эквивалентны (0,0), (5,5), (-5,5)
    assert km.centers_ is not None
    centers = np.array(sorted([tuple(np.round(c, 0)) for c in km.centers_]))
    expected = np.array(sorted([(0., 0.), (5., 5.), (-5., 5.)]))
    assert np.allclose(centers, expected, atol=0.5), centers
    print("kmeans: known-answer test passed")

def test_kmeans_empty_cluster():
    # две точки → 3 кластера; должен сработать fallback для пустого
    X = np.array([[0., 0.], [1., 1.]])
    km = OwnKMeans(n_clusters=3, n_init=1, random_state=0).fit(X)
    assert km.labels_ is not None
    assert len(np.unique(km.labels_)) >= 1
    print("kmeans: empty-cluster fallback passed")

def test_pca_known():
    # 2D с сильной линейной связью → 1-я компонента ~ (1, 1)/sqrt(2)
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=1000), rng.normal(size=1000)])
    X[:, 1] = X[:, 0] * 3 + 0.01 * rng.normal(size=1000)
    pca = OwnPCA(n_components=1).fit(X)
    comp = pca.components_
    assert comp is not None
    c = comp[0]
    # допускаем ±знак
    if c[0] < 0:
        c = -c
    expected = np.array([1, 3]) / np.linalg.norm([1, 3])
    assert np.allclose(np.abs(c), np.abs(expected), atol=1e-2), c
    print("pca: first component matches expected direction")

def test_pca_reconstruction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 5))
    pca = OwnPCA(n_components=5).fit(X)
    err = pca.reconstruction_error(X)
    assert err < 1e-10, err
    print("pca: full-rank reconstruction is exact")

if __name__ == "__main__":
    test_kmeans_known()
    test_kmeans_empty_cluster()
    test_pca_known()
    test_pca_reconstruction()
    print("Все проверки части Б пройдены.")