import numpy as np
from own_kmeans import OwnKMeans
from own_pca import OwnPCA


def test_kmeans_known():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal((0, 0), 0.3, size=(50, 2)),
        rng.normal((5, 5), 0.3, size=(50, 2)),
        rng.normal((-5, 5), 0.3, size=(50, 2)),
    ])
    km = OwnKMeans(n_clusters=3, n_init=1, random_state=0).fit(X)
    assert km.centers_ is not None
    centers = np.array(sorted([tuple(np.round(c, 0)) for c in km.centers_]))
    expected = np.array(sorted([(0., 0.), (5., 5.), (-5., 5.)]))
    assert np.allclose(centers, expected, atol=0.5), centers
    print("kmeans: known-answer test passed")


def test_kmeans_empty_cluster():
    # 3 точки, но 2 из них совпадают → после argmin один кластер пуст,
    # должен сработать fallback: центр пустого кластера переставляется
    # в точку, максимально удалённую от своего центра
    X = np.array([[0., 0.], [0., 0.], [10., 10.]])
    km = OwnKMeans(n_clusters=3, n_init=1, random_state=0).fit(X)
    assert km.labels_ is not None
    # все объекты распределены
    assert len(km.labels_) == len(X)
    # эффективное число центров = min(k, n) = 3
    assert km.centers_.shape == (3, 2)
    # каждый центр — это либо (0,0), либо (10,10), т.к. других точек нет
    for c in km.centers_:
        assert (np.allclose(c, [0., 0.]) or np.allclose(c, [10., 10.])), c
    # хотя бы один объект должен быть отнесён к кластеру с центром (10,10)
    # (fallback не должен «потерять» дальнюю точку)
    labels_at_far = km.labels_[np.where(np.all(X == [10., 10.], axis=1))[0]]
    assert len(np.unique(labels_at_far)) == 1
    print("kmeans: empty-cluster fallback passed")


def test_kmeans_override_centers():
    # проверка, что внешние стартовые центры принимаются и используются
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal((0, 0), 0.3, size=(50, 2)),
        rng.normal((5, 5), 0.3, size=(50, 2)),
    ])
    init = np.array([[0.1, 0.1], [4.9, 4.9]])
    km = OwnKMeans(n_clusters=2, n_init=1, random_state=0)
    km._init_centers_override = init.copy()
    km.fit(X)
    # после сходимости центры должны быть близки к истинным
    centers = np.array(sorted([tuple(np.round(c, 0)) for c in km.centers_]))
    expected = np.array(sorted([(0., 0.), (5., 5.)]))
    assert np.allclose(centers, expected, atol=0.5), centers
    print("kmeans: override init centers passed")


def test_kmeans_k_greater_than_n():
    # k > n: эффективное число центров = n
    X = np.array([[0., 0.], [1., 0.], [2., 0.]])
    km = OwnKMeans(n_clusters=5, n_init=1, random_state=0).fit(X)
    assert len(km.centers_) == 3
    print("kmeans: k > n passed")


def test_pca_known():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=1000), rng.normal(size=1000)])
    X[:, 1] = X[:, 0] * 3 + 0.01 * rng.normal(size=1000)
    pca = OwnPCA(n_components=1).fit(X)
    comp = pca.components_
    assert comp is not None
    c = comp[0]
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


def test_pca_vs_sklearn():
    # own PCA должна совпадать со sklearn по explained_variance_ratio
    # (с точностью до знака/базиса) и reconstruction MSE
    from sklearn.decomposition import PCA as SkPCA
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 6)) @ rng.normal(size=(6, 6))
    for k in (1, 3, 6):
        own = OwnPCA(n_components=k).fit(X)
        sk = SkPCA(n_components=k).fit(X)
        # explained_variance_ratio
        assert np.allclose(own.explained_variance_ratio_,
                           sk.explained_variance_ratio_, atol=1e-8)
        # reconstruction MSE
        err_own = own.reconstruction_error(X)
        Xr_sk = sk.inverse_transform(sk.transform(X))
        err_sk = float(((X - Xr_sk) ** 2).mean())
        assert abs(err_own - err_sk) < 1e-8, (err_own, err_sk)
        # |cos| между компонентами
        A = own.components_; B = sk.components_
        cos = np.abs(np.sum(A * B, axis=1)) / (
            np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)
        assert np.all(cos > 1 - 1e-8), cos
    print("pca: matches sklearn (explained var, reconstruction, subspace)")


if __name__ == "__main__":
    test_kmeans_known()
    test_kmeans_empty_cluster()
    test_kmeans_override_centers()
    test_kmeans_k_greater_than_n()
    test_pca_known()
    test_pca_reconstruction()
    test_pca_vs_sklearn()
    print("Все проверки части Б пройдены.")