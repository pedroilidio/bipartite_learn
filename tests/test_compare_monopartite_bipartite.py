import logging
import numpy as np
import pytest
from scipy.stats import ttest_ind, ttest_rel
from sklearn.utils._testing import assert_allclose
from sklearn.utils.validation import check_random_state
from sklearn.dummy import DummyRegressor
from sklearn.ensemble._gb import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from bipartite_learn.ensemble._gb import (
    BipartiteGradientBoostingClassifier,
    BipartiteGradientBoostingRegressor,
)
from bipartite_learn.ensemble._forest import (
    BipartiteRandomForestRegressor,
    BipartiteExtraTreesRegressor,
)
from bipartite_learn.wrappers import GlobalSingleOutputWrapper
from .utils.make_examples import make_interaction_regression
from .utils.test_utils import assert_equal_dicts, stopwatch
from sklearn import metrics


@pytest.mark.parametrize(
    "monopartite_estimator,bipartite_estimator,common_estimator_params",
    [
        (
            RandomForestRegressor,
            BipartiteRandomForestRegressor,
            dict(
                criterion='squared_error',
                n_estimators=10,
                bootstrap=False,  # Bootstrapping in bipartite data is different.
            ),
        ),
        (
            ExtraTreesRegressor,
            BipartiteExtraTreesRegressor,
            dict(
                criterion='squared_error',
                n_estimators=30,
            ),
        ),
        (
            GradientBoostingRegressor,
            BipartiteGradientBoostingRegressor,
            dict(
                criterion='friedman_mse',
                subsample=0.2,
                n_estimators=10,
            ),
        ),
    ],
)
def test_compare_estimators(
    monopartite_estimator,
    bipartite_estimator,
    common_estimator_params,
    scorer=metrics.mean_squared_error,
    random_state=0,
    n_seeds=30,
):
    logging.info(f"{random_state=}")
    seeds = check_random_state(random_state).randint(
        np.iinfo(np.uint32).max,
        size=n_seeds,
        dtype=np.uint32,
    )
    monopartite_scores = []
    bipartite_scores = []
    dummy_scores = []

    for seed in seeds:
        logging.info(f"{seed=}")
        common_estimator_params.update({'random_state': seed})
        dummy_estimator = DummyRegressor()
        monopartite_estimator_ = monopartite_estimator(**common_estimator_params)
        bipartite_estimator_ = bipartite_estimator(**common_estimator_params)

        XX, Y, X, y = make_interaction_regression(
            n_samples=(40, 50),
            n_features=(9, 10),
            return_molten=True,
            random_state=seed,
            # max_target=2,
        )
        # Classification
        # Y = Y.astype(int)
        # y = y.astype(int)
        bipartite_estimator_.fit(XX, Y)
        monopartite_estimator_.fit(X, y)
        dummy_estimator.fit(X, y)
        
        assert_equal_dicts(
            monopartite_estimator_.get_params(),
            bipartite_estimator_.get_params(),
        )

        dummy_pred = dummy_estimator.predict(X)
        pred1 = monopartite_estimator_.predict(X)
        pred2 = bipartite_estimator_.predict(XX)
        # score1 = np.mean((y - pred1)**2)
        # score2 = np.mean((y - pred2)**2)
        # score1 = monopartite_estimator_.score(X, y)
        # score2 = bipartite_estimator_.score(XX, y)
        dummy_score = scorer(dummy_pred, y)
        score1 = scorer(pred1, y)
        score2 = scorer(pred2, y)

        logging.info(f"Monopartite score: {score1}")
        logging.info(f"Bipartite score:   {score2}")

        dummy_scores.append(dummy_score)
        monopartite_scores.append(score1)
        bipartite_scores.append(score2)
    
    # FIXME: paired test seems more adequate, but does not yield significant results.
    assert ttest_ind(dummy_scores, monopartite_scores).pvalue < 0.05
    assert ttest_ind(dummy_scores, bipartite_scores).pvalue < 0.05
    assert ttest_ind(monopartite_scores, bipartite_scores).pvalue > 0.05
