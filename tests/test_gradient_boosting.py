import logging
import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose
from sklearn.ensemble._gb import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from bipartite_learn.ensemble._gb import (
    BipartiteGradientBoostingClassifier,
    BipartiteGradientBoostingRegressor,
)
from bipartite_learn.tree._bipartite_classes import BIPARTITE_CRITERIA
from .utils.make_examples import make_interaction_regression
from .utils.test_utils import stopwatch


@pytest.fixture(params=range(5))
def random_state(request):
    return request.param


@pytest.fixture(
    params=[
        ("squared_error", "squared_error_gso"),
        ("friedman_mse", "friedman_gso"),
    ],
    ids=["mse", "friedman"],
)
def regression_criteria_pair(request):
    return request.param


@pytest.fixture(params=[1.0, 0.1])
def subsample(request):
    return request.param


@pytest.fixture
def mono_bi_estimators(random_state, subsample, regression_criteria_pair):
    mono_criterion, bi_criterion = regression_criteria_pair
    mono_estimator = GradientBoostingRegressor(
        criterion=mono_criterion,
        random_state=random_state,
        subsample=subsample,
    )
    bi_estimator = BipartiteGradientBoostingRegressor(
        criterion=bi_criterion,
        random_state=random_state,
        subsample=subsample,
        bipartite_adapter="gmosa",
    )
    return mono_estimator, bi_estimator


@pytest.fixture
def data(random_state):
    XX, Y, X, y = make_interaction_regression(
        n_samples=(23, 37),
        random_state=random_state,
        return_molten=True,
    )
    Y_bin = (Y > Y.mean()).astype("float64")
    y_bin = (y > y.mean()).astype("float64")
    return XX, Y, X, y, Y_bin, y_bin


@pytest.fixture
def fitted_estimators(mono_bi_estimators, data):
    mono_estimator, bi_estimator = mono_bi_estimators
    XX, Y, X, y, Y_bin, y_bin = data
    with stopwatch("Fitting bipartite..."):
        bi_estimator.fit(XX, Y)
    with stopwatch("Fitting monopartite..."):
        mono_estimator.fit(X, y)
    return mono_estimator, bi_estimator


# NOTE: GBM Classifier uses a regressor tree as a base estimator
@pytest.fixture(params=list(BIPARTITE_CRITERIA["gmosa"]["regression"].keys()))
def classifier(request, random_state, subsample):
    estimator = BipartiteGradientBoostingClassifier(
        random_state=random_state,
        subsample=subsample,
        criterion=request.param,
        bipartite_adapter="gmosa",
    )
    return estimator


@pytest.fixture
def fitted_classifier(classifier, data):
    XX, Y, X, y, Y_bin, y_bin = data
    with stopwatch("Fitting classifier..."):
        classifier.fit(XX, Y_bin)
    return classifier


def test_predict(fitted_estimators, subsample, data):
    mono_estimator, bi_estimator = fitted_estimators
    XX, Y, X, y, *_ = data

    pred1 = mono_estimator.predict(X)
    pred2 = bi_estimator.predict(XX)
    mse1 = np.mean((y - pred1) ** 2)
    mse2 = np.mean((y - pred2) ** 2)

    logging.info(f"Base MSE (var):  {y.var()}")
    logging.info(f"Monopartite MSE: {mse1}")
    logging.info(f"Bipartite MSE:   {mse2}")

    if subsample == 1.0:
        assert_allclose(mse1, mse2)
        assert_allclose(pred1, pred2)


def test_staged_predict(fitted_estimators, subsample, data):
    mono_estimator, bi_estimator = fitted_estimators
    XX, Y, X, *_ = data

    stage_iter1 = mono_estimator.staged_predict(X)
    stage_iter2 = bi_estimator.staged_predict(XX)

    for pred1, pred2 in zip(stage_iter1, stage_iter2):
        assert pred1.shape == pred2.shape == (X.shape[0],)
        if subsample == 1.0:
            assert_allclose(pred1, pred2)


@pytest.mark.parametrize(
    "method",
    [
        "predict",
        "predict_proba",
        "predict_log_proba",
        "decision_function",
    ],
)
def test_classifier_inference(fitted_classifier, data, method):
    estimator = fitted_classifier
    XX, Y, X, y, *_ = data
    out1 = getattr(estimator, method)(XX)
    out2 = getattr(estimator, method)(X)
    assert out1.shape[0] == X.shape[0]

    if method in ["predict_proba", "predict_log_proba"]:
        assert out1.ndim == 2
        assert out1.shape[1] == estimator.n_classes_

    assert (out1 == out2).all()


@pytest.mark.parametrize(
    "method",
    [
        "staged_predict",
        "staged_predict_proba",
        "staged_decision_function",
    ],
)
def test_classifier_staged_inference(fitted_classifier, data, method):
    estimator = fitted_classifier
    XX, Y, X, y, *_ = data
    out1_iter = getattr(estimator, method)(XX)
    out2_iter = getattr(estimator, method)(X)

    for out1, out2 in zip(out1_iter, out2_iter):
        assert out1.shape[0] == X.shape[0]

        if method == "staged_predict_proba":
            assert out1.ndim == 2
            assert out1.shape[1] == estimator.n_classes_

        assert (out1 == out2).all()
