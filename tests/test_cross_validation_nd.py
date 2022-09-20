from pprint import pprint
from typing import Sequence
from sklearn.utils._tags import _safe_tags

from hypertrees.model_selection import cross_validate_nd
from hypertrees.tree import ExtraTreeRegressor2D
from hypertrees.ensemble import ExtraTreesRegressor2D

from test_utils import gen_mock_data, stopwatch, DEF_PARAMS, parse_args


# Default test params
LOCAL_DEF_PARAMS = dict(
    seed=7,
    shape=(50, 40),
    nrules=3,
    min_samples_leaf=100,
    transpose_test=False,
    noise=0.1,
    inspect=False,
    plot=False,
)

CV_DEF_PARAMS = dict(
    groups=None,
    scoring="roc_auc",
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

DEF_PARAMS |= LOCAL_DEF_PARAMS


def _test_cv(estimator, cv_params=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    PARAMS['noise'] = 0.
    pprint(PARAMS)
    cv_params = CV_DEF_PARAMS | (cv_params or {})
    cv_params |= dict(return_estimator=True)
    pprint(cv_params)

    is_pairwise = cv_params["pairwise"] or _safe_tags(estimator, "pairwise")
    cv = cv_params["cv"]
    n_dim = len(PARAMS['shape'])

    if isinstance(cv, Sequence):
        if not isinstance(cv[0], int):
            raise TypeError("cv_params['cv'] must be (sequence of) int")

    elif not isinstance(cv, int):
        raise TypeError("cv_params['cv'] must be (sequence of) int")

    else:  # isinstance(cv_params["cv"], int)
        cv = (cv, cv)

    if is_pairwise:
        if PARAMS["nattrs"] != PARAMS["shape"]:
            raise ValueError(
                "Pairwise estimators must receive square Xs satisfying"
                f"attrs == shape (PARAMS['nattrs'] != PARAMS['nattrs'])"
            )
        n_samples = PARAMS["shape"]
        # For pairwise data, each fold must be square
        # Sum Xi.shape where Xi is a training fold of X
        fold_n_samples = sum(s-(s//cvi) for s, cvi in zip(n_samples, cv))
    else:
        # Sum Yi.shape where Yi is a training fold of Y
        n_features = sum(PARAMS["nattrs"])

    with stopwatch('Generating data...'):
        X, y, _ = gen_mock_data(**PARAMS)
    
    if y.var() == 0:
        raise RuntimeError(f'y is homogeneous, try another seed.')

    cv_res = cross_validate_nd(estimator, X=X, y=y, **cv_params)
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
    
    if PARAMS['inspect']:
        breakpoint()

    return cv_res


def test_cv_tree(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    return _test_cv(
        estimator=ExtraTreeRegressor2D(
            min_samples_leaf=PARAMS["min_samples_leaf"],
        ),
        **PARAMS,
    )


def test_cv_ensemble(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    return _test_cv(
        estimator=ExtraTreesRegressor2D(
            min_samples_leaf=PARAMS["min_samples_leaf"],
        ),
        **PARAMS,
    )


def test_cv_pairwise_tag(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    PARAMS["nattrs"] = PARAMS["shape"]  # Make Xs square
    cv_params = dict(pairwise=False)  # tag will be gotten from the estimator

    estimator = ExtraTreeRegressor2D(
        min_samples_leaf=PARAMS["min_samples_leaf"],
    )

    # Monkey-patch pairwise tag
    old_tags = _safe_tags(estimator)
    estimator.__class__._more_tags = (
        lambda self: old_tags | dict(pairwise=True)
    )
    # Revert patch even if an exception is raised
    try:
        cv_res = _test_cv(estimator, cv_params=cv_params, **PARAMS)
    finally:
        estimator.__class__._more_tags = lambda self: old_tags

    assert not _safe_tags(estimator, "pairwise")
    return cv_res


def test_cv_pairwise_parameter(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    cv_params = dict(pairwise=True)
    estimator = ExtraTreeRegressor2D(
        min_samples_leaf=PARAMS["min_samples_leaf"],
    )
    PARAMS["nattrs"] = PARAMS["shape"]  # Make Xs square

    cv_res = _test_cv(estimator, cv_params=cv_params, **PARAMS)

    assert not _safe_tags(estimator, "pairwise")
    return cv_res


def main(**PARAMS):
    test_cv_tree(**PARAMS)
    test_cv_ensemble(**PARAMS)
    test_cv_pairwise_tag(**PARAMS)
    test_cv_pairwise_parameter(**PARAMS)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))