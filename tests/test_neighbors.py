import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.utils._testing import assert_allclose
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from bipartite_learn.neighbors import WeightedNeighborsRegressor


@pytest.fixture
def data():
    X, y = make_regression(n_samples=1000, n_features=500, n_targets=100)
    X.setflags(write=False)
    return X, y


def test_weighted_neighbors_euclidean(data):
    X, y = data

    rng = np.random.default_rng()
    X_test = rng.random((X.shape[0] // 2, X.shape[1]))

    wnr = WeightedNeighborsRegressor()
    wnr.fit(X, y)

    assert_allclose(wnr.predict(X), y)

    y_pred = wnr.predict(X_test)
    euclidean = np.sqrt(((X_test[:, np.newaxis, :] - X) ** 2).sum(axis=-1))
    w = 1 / euclidean
    w /= w.sum(axis=1, keepdims=True)

    assert_allclose(y_pred, (w @ y))


def test_weighted_neighbors_precomputed(data):
    X, y = data
    X = euclidean_distances(X) + 1e-6  # to avoid zero distances

    wnr = WeightedNeighborsRegressor(metric="precomputed")
    wnr.fit(X, y)
    wnr.predict(X[: X.shape[0] // 2])

    y_pred = wnr.predict(X)

    w = 1 / X
    w /= w.sum(axis=1, keepdims=True)
    assert (y_pred == w @ y).all()


def test_weighted_neighbors_precomputed_similarity(data):
    X, y = data
    X = rbf_kernel(X)

    wnr = WeightedNeighborsRegressor(
        metric="precomputed",
        weights="similarity",
    )
    wnr.fit(X, y)
    y_pred = wnr.predict(X)

    w = X / X.sum(axis=1, keepdims=True)

    assert np.allclose(y_pred, (w @ y), rtol=1e-5, atol=1e-5)


def test_weighted_neighbors_regressor_validate_params(data):
    X, y = data
    X = rbf_kernel(X)
    wnr = WeightedNeighborsRegressor(metric="euclidean", weights="similarity")

    with pytest.raises(
        ValueError,
        match="weights='similarity' is not supported for string-valued.*",
    ):
        wnr.fit(X, y)
