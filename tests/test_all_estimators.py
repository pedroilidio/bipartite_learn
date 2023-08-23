import logging
import warnings
import functools

import pytest
from sklearn.utils.validation import check_random_state
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.utils._testing import assert_allclose
from sklearn.metrics import r2_score, f1_score
from sklearn.metrics.pairwise import rbf_kernel
from bipartite_learn.base import BaseMultipartiteEstimator
from bipartite_learn.utils import all_estimators
from bipartite_learn.melter import row_cartesian_product
from .utils.make_examples import (
    make_interaction_blobs,
    make_interaction_regression,
)


def apply_rbf_kernel(regression_data):
    X, Y, x, y = regression_data
    X = [rbf_kernel(X[0]), rbf_kernel(X[1])]
    x = row_cartesian_product(X)
    return X, Y, x, y


@pytest.fixture(params=range(3))
def random_state(request):
    return check_random_state(request.param)


@pytest.fixture
def n_samples(request):
    return (11, 13)


@pytest.fixture
def n_features(request):
    return (5, 7)


@pytest.fixture
def classification_data(n_samples, n_features, random_state):
    X, Y, x, y = make_interaction_blobs(
        return_molten=True,
        n_features=n_features,
        n_samples=n_samples,
        random_state=random_state,
        noise=0.0,
        centers=2,
        row_kwargs={'center_box': (.3, .7), 'cluster_std': .1},
        col_kwargs={'center_box': (.3, .7), 'cluster_std': .1},
    )
    # XXX
    Y = (Y == 0).astype(int)
    y = Y.reshape(-1)
    return X, Y, x, y


@pytest.fixture
def regression_data(n_samples, n_features, random_state):
    X, Y, x, y = make_interaction_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=None,
        min_target=0.0,
        max_target=1.0,
        noise=0.0,
        return_molten=True,
        return_tree=False,
        random_state=random_state,
        max_depth=None,
    )
    return X, Y, x, y


@pytest.fixture
def pairwise_regression_data(regression_data):
    return apply_rbf_kernel(regression_data)


@pytest.fixture
def pairwise_classification_data(classification_data):
    return apply_rbf_kernel(classification_data)


@pytest.fixture
def regression_scorer():
    return r2_score


@pytest.fixture
def classification_scorer():
    return functools.partial(f1_score, average='macro')


@pytest.fixture
def regression_min_score(regression_data, regression_scorer):
    _, _, x, y = regression_data
    pred = DummyRegressor().fit(x, y).predict(x)
    return regression_scorer(y, pred.reshape(-1, 1))


@pytest.fixture
def classification_min_score(classification_data, classification_scorer):
    _, _, x, y = classification_data
    pred = DummyClassifier().fit(x, y).predict(x)
    return classification_scorer(y, pred.reshape(-1, 1))


@pytest.mark.parametrize(
    'estimator_name, estimator',
    (
        e for e in all_estimators("regressor")
        if not issubclass(e[1], BaseMultipartiteEstimator)
    )
)
def test_all_monopartite_regressors(
    regression_data,
    pairwise_regression_data,
    estimator_name,
    estimator,
    regression_scorer,
    regression_min_score,
):
    estimator = estimator()

    if estimator._get_tags()['pairwise']:
        X, Y, x, y = pairwise_regression_data
    else:
        X, Y, x, y = regression_data

    estimator.fit(X[0], Y)
    estimator.score(X[0], Y)  # Test own scorer
    score = regression_scorer(y, estimator.predict(X[0]).reshape(-1, 1))

    logging.info(f"{estimator_name}'s score: {score}")
    assert score > regression_min_score


@pytest.mark.parametrize(
    'estimator_name, estimator', all_estimators("regressor", "multipartite")
)
def test_all_multipartite_regressors(
    regression_data,
    pairwise_regression_data,
    estimator_name,
    estimator,
    regression_scorer,
    regression_min_score,
):
    estimator = estimator()

    if estimator._get_tags()['pairwise']:
        X, Y, x, y = pairwise_regression_data
    else:
        X, Y, x, y = regression_data

    estimator.fit(X, Y)

    # The two formats must be supported:
    pred1 = estimator.predict(X)
    try:
        pred2 = estimator.predict(x)
        assert_allclose(pred1, pred2)
    except ValueError:
        warnings.warn(f"{estimator_name} does not accept melted input.")

    estimator.score(X, y)  # Test own scorer
    score = regression_scorer(y, estimator.predict(X).reshape(-1, 1))

    logging.info(f"{estimator_name}'s score: {score}")
    assert score > regression_min_score


@pytest.mark.parametrize(
    'estimator_name, estimator', all_estimators("classifier", "multipartite")
)
def test_all_multipartite_classifiers(
    classification_data,
    pairwise_classification_data,
    estimator_name,
    estimator,
    classification_scorer,
    classification_min_score,
):
    estimator = estimator()

    if estimator._get_tags()['pairwise']:
        X, Y, x, y = pairwise_classification_data
    else:
        X, Y, x, y = classification_data

    estimator.fit(X, Y)

    # The two formats must be supported:
    pred1 = estimator.predict(X)
    try:
        pred2 = estimator.predict(x)
        assert_allclose(pred1, pred2)
    except ValueError:
        warnings.warn(f"{estimator_name} does not accept melted input.")

    estimator.score(X, y)  # Test own scorer
    score = classification_scorer(y, estimator.predict(X).reshape(-1, 1))

    logging.info(f"{estimator_name}'s score: {score}")
    assert score > classification_min_score
