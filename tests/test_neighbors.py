import numpy as np
from sklearn.datasets import make_regression
from sklearn.utils._testing import assert_allclose
from bipartite_learn.neighbors import WeightedNeighborsRegressor


def test_weighted_neighbors_euclidean():
    X, y = make_regression(n_samples=1000, n_features=500, n_targets=100)
    X.setflags(write=False)

    rng = np.random.default_rng()
    X_test = rng.random((X.shape[0]//2, X.shape[1]))

    wnr = WeightedNeighborsRegressor()
    wnr.fit(X, y)

    assert_allclose(wnr.predict(X), y)

    y_pred = wnr.predict(X_test)
    euclidean = np.sqrt(((X_test[:, np.newaxis, :] - X) ** 2).sum(axis=-1))
    w = 1 / euclidean
    w /= w.sum(axis=1, keepdims=True)

    assert_allclose(y_pred, (w @ y))


def test_weighted_neighbors_precomputed():
    X, y = make_regression(n_samples=5000, n_features=5000, n_targets=100)
    X = np.abs(X)
    X.setflags(write=False)

    wnr = WeightedNeighborsRegressor(metric="precomputed")
    wnr.fit(X, y)
    wnr.predict(X[:X.shape[0]//2])

    y_pred = wnr.predict(X)

    w = (1/X)
    w /= w.sum(axis=1, keepdims=True)
    assert (y_pred == w @ y).all()


def main():
    test_weighted_neighbors_euclidean()
    test_weighted_neighbors_precomputed()


if __name__ == '__main__':
    main()