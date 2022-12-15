import logging
import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose
from sklearn.ensemble._gb import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from hypertrees.ensemble._gb import (
    BipartiteGradientBoostingClassifier,
    BipartiteGradientBoostingRegressor,
)
from hypertrees.wrappers import GlobalSingleOutputWrapper
from make_examples import make_interaction_regression
from test_utils import stopwatch


@pytest.fixture(params=range(5))
def random_state(request):
    return request.param


@pytest.mark.parametrize("subsample", (1.0, 0.1))
@pytest.mark.parametrize("criterion", ("squared_error", "friedman_mse"))
def test_gradient_boosting(random_state, subsample, criterion):
    estimator1 = GradientBoostingRegressor(
        criterion=criterion,
        random_state=random_state,
        subsample=subsample,
    )
    estimator = BipartiteGradientBoostingRegressor(
        criterion=criterion,
        random_state=random_state,
        subsample=subsample,
    )

    XX, Y, X, y = make_interaction_regression(
        n_samples=(40, 50),
        random_state=random_state,
        return_molten=True,
    )

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
