from pprint import pprint
from typing import Sequence
import pytest
from sklearn.utils._tags import _safe_tags

from bipartite_learn.model_selection import multipartite_cross_validate
from bipartite_learn.tree import BipartiteExtraTreeRegressor
from bipartite_learn.ensemble import BipartiteExtraTreesRegressor
import bipartite_learn.ensemble

from .utils.test_utils import stopwatch, parse_args
from .utils.make_examples import make_interaction_regression


# Default test params
DEF_PARAMS = dict(
    n_samples=(50, 40),
    n_features=(10, 9),
    noise=0.1,
)

CV_DEF_PARAMS = dict(
    groups=None,
    scoring="average_precision",
    cv=3,
    n_jobs=None,
    verbose=10,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    error_score='raise',
    diagonal=False,
    train_test_combinations=None,
    pairwise=False,
)


@pytest.fixture(params=[6, .1])
def msl(request):  # min_samples_leaf parameter
    return request.param


@pytest.fixture(params=range(10))
def random_state(request):
    return request.param


def _test_cv(estimator, random_state=None, cv_params=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    PARAMS['noise'] = 0.
    pprint(PARAMS)
    cv_params = CV_DEF_PARAMS | (cv_params or {})
    cv_params |= dict(return_estimator=True)
    pprint(cv_params)

    is_pairwise = cv_params["pairwise"] or _safe_tags(estimator, "pairwise")
    cv = cv_params["cv"]
    n_dim = len(PARAMS['n_samples'])

    if isinstance(cv, Sequence):
        if not isinstance(cv[0], int):
            raise TypeError("cv_params['cv'] must be (sequence of) int")

    elif not isinstance(cv, int):
        raise TypeError("cv_params['cv'] must be (sequence of) int")

    else:  # isinstance(cv_params["cv"], int)
        cv = (cv, cv)

    if is_pairwise:
        if PARAMS["n_features"] != PARAMS["n_samples"]:
            raise ValueError(
                "Pairwise estimators must receive square Xs satisfying"
                f"attrs == n_samples (PARAMS['n_features'] != PARAMS['n_features'])"
            )
        n_samples = PARAMS["n_samples"]
        # For pairwise data, each fold must be square
        # Sum Xi.n_samples where Xi is a training fold of X
        fold_n_samples = sum(s-(s//cvi) for s, cvi in zip(n_samples, cv))
    else:
        # Sum Yi.n_samples where Yi is a training fold of Y
        n_features = sum(PARAMS["n_features"])

    with stopwatch('Generating data...'):
        X, y = make_interaction_regression(random_state=random_state, **PARAMS)
        y = y > y.mean()
    
    # TODO: random_state
    cv_res = multipartite_cross_validate(estimator, X=X, y=y, **cv_params)
    pprint(cv_res)

    for estimator in cv_res["estimator"]:
        if is_pairwise:
            # When cv does not divide n_samples perfectly, it distributes the
            # remainders across folds (see np.array_split), so that each fold
            # size can vary on one unit. Naturally, this happens for the n_dim
            # X matrices, so n_features_in_ can be up to n_dim smaller.
            assert (estimator.n_features_in_ - fold_n_samples) <= n_dim
        else:
            assert estimator.n_features_in_ == n_features
    
    return cv_res


def test_cv_tree(msl, random_state, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    return _test_cv(
        estimator=BipartiteExtraTreeRegressor(
            min_samples_leaf=msl,
            random_state=random_state,
        ),
        random_state=random_state,
        **PARAMS,
    )


def test_cv_ensemble(msl, random_state, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    return _test_cv(
        estimator=BipartiteExtraTreesRegressor(
            n_estimators=10,
            min_samples_leaf=msl,
            random_state=random_state,
        ),
        random_state=random_state,
        **PARAMS,
    )


def test_cv_semisupervised_ensemble(msl, random_state, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    return _test_cv(
        estimator=bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
            n_estimators=10,
            min_samples_leaf=msl,
            random_state=random_state,
            bipartite_adapter='gmosa',
            criterion='squared_error',
        ),
        random_state=random_state,
        **PARAMS,
    )


def test_cv_pairwise_tag(msl, random_state, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    PARAMS["n_features"] = PARAMS["n_samples"]  # Make Xs square
    cv_params = dict(pairwise=False)  # tag will be gotten from the estimator

    estimator = BipartiteExtraTreeRegressor(
        min_samples_leaf=msl,
        random_state=random_state,
    )

    # Monkey-patch pairwise tag
    old_tags = _safe_tags(estimator)
    estimator.__class__._more_tags = (
        lambda self: old_tags | dict(pairwise=True)
    )
    # Revert patch even if an exception is raised
    try:
        cv_res = _test_cv(
            estimator,
            random_state=random_state,
            cv_params=cv_params,
            **PARAMS,
        )
    finally:
        estimator.__class__._more_tags = lambda self: old_tags

    assert not _safe_tags(estimator, "pairwise")
    return cv_res


def test_cv_pairwise_parameter(msl, random_state, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    cv_params = dict(pairwise=True)
    estimator = BipartiteExtraTreeRegressor(
        min_samples_leaf=msl,
        random_state=random_state,
    )
    PARAMS["n_features"] = PARAMS["n_samples"]  # Make Xs square

    cv_res = _test_cv(
        estimator, cv_params=cv_params, random_state=random_state, **PARAMS,
    )

    assert not _safe_tags(estimator, "pairwise")
    return cv_res