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
from bipartite_learn.wrappers import GlobalSingleOutputWrapper
from .utils.make_examples import make_interaction_regression
from .utils.test_utils import stopwatch


@pytest.fixture(params=range(5))
def random_state(request):
    return request.param


@pytest.mark.parametrize("subsample", (1.0, 0.1))
@pytest.mark.parametrize(
    "mono_criterion, bi_criterion",
    [
        ("squared_error", "squared_error_gso"),
        ("friedman_mse", "friedman_gso"),
    ],
    ids=['mse', 'friedman'],
)
def test_gradient_boosting(
    random_state,
    subsample,
    mono_criterion,
    bi_criterion,
):
    estimator1 = GradientBoostingRegressor(
        criterion=mono_criterion,
        random_state=random_state,
        subsample=subsample,
    )
    estimator = BipartiteGradientBoostingRegressor(
        criterion=bi_criterion,
        random_state=random_state,
        subsample=subsample,
        bipartite_adapter="gmosa",
    )

    XX, Y, X, y = make_interaction_regression(
        n_samples=(40, 50),
        random_state=random_state,
        return_molten=True,
    )
    y = y.reshape(-1, 1)

    logging.info(f"{random_state=}, {subsample=}")
    with stopwatch("Fitting bipartite..."):
        estimator.fit(XX, Y)
    with stopwatch("Fitting monopartite..."):
        estimator1.fit(X, y)

    pred1 = estimator1.predict(X)
    pred2 = estimator.predict(XX)
    mse1 = np.mean((y - pred1)**2)
    mse2 = np.mean((y - pred2)**2)

    logging.info(f"Base MSE (var):  {y.var()}")
    logging.info(f"Monopartite MSE: {mse1}")
    logging.info(f"Bipartite MSE:   {mse2}")

    if subsample == 1.0:
        assert_allclose(mse1, mse2)
        assert_allclose(pred1, pred2)
