import pytest
from pprint import pprint
from typing import Sequence
import numpy as np
from sklearn.utils._tags import _safe_tags
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import rbf_kernel

from bipartite_learn.model_selection import (
    multipartite_cross_validate,
    check_multipartite_cv,
    MultipartiteCrossValidator,
)
from bipartite_learn.tree import BipartiteExtraTreeRegressor
from bipartite_learn.ensemble import BipartiteExtraTreesRegressor

import bipartite_learn.ensemble

from .utils.test_utils import stopwatch, parse_args
from .utils.make_examples import make_interaction_regression


CV_DEF_PARAMS = dict(
    groups=None,
    scoring="average_precision",
    cv=3,
    n_jobs=None,
    verbose=10,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    error_score="raise",
    diagonal=False,
    train_test_combinations=None,
    pairwise=False,
)


@pytest.fixture(params=range(5))
def random_state(request):
    return request.param


@pytest.fixture
def data(random_state):
    XX, Y = make_interaction_regression(
        n_samples=(50, 40),
        n_features=(10, 9),
        return_molten=False,
        noise=0.0,
    )
    XX = [rbf_kernel(Xi, gamma=1) for Xi in XX]
    Y = (Y > Y.mean()).astype("float64")
    return XX, Y


def _test_cv(estimator, data, cv_params=None):
    X, y = data

    cv_params = CV_DEF_PARAMS | (cv_params or {})
    cv_params |= dict(return_estimator=True)

    is_pairwise = cv_params["pairwise"] or _safe_tags(estimator, "pairwise")
    cv = cv_params["cv"]

    if isinstance(cv, Sequence):
        if not isinstance(cv[0], int):
            raise TypeError("cv_params['cv'] must be (sequence of) int")

    elif not isinstance(cv, int):
        raise TypeError("cv_params['cv'] must be (sequence of) int")

    else:  # isinstance(cv_params["cv"], int)
        cv = (cv, cv)

    cv_res = multipartite_cross_validate(
        estimator,
        X=X,
        y=y,
        **cv_params,
    )
    print("Cross-validation results:")
    pprint(cv_res)

    for estimator in cv_res["estimator"]:
        if is_pairwise:
            # For pairwise data, each fold must be square
            # Sum Xi.n_samples where Xi is a training fold of X
            fold_n_samples = sum(s - (s // cvi) for s, cvi in zip(y.shape, cv))
            # When cv does not divide n_samples perfectly, it distributes the
            # remainders across folds (see np.array_split), so that each fold
            # size can vary on one unit. Naturally, this happens for the n_dim
            # X matrices, so n_features_in_ can be up to n_dim smaller.
            assert (estimator.n_features_in_ - fold_n_samples) <= y.ndim
        else:
            # Sum Yi.n_samples where Yi is a training fold of Y
            n_features = sum(Xi.shape[1] for Xi in X)
            assert estimator.n_features_in_ == n_features

    return cv_res


# parametrize estimator:
@pytest.mark.parametrize(
    "estimator",
    [
        BipartiteExtraTreeRegressor(
            min_samples_leaf=25,
        ),
        BipartiteExtraTreesRegressor(
            min_samples_leaf=25,
        ),
        bipartite_learn.ensemble.BipartiteExtraTreesRegressorSS(
            n_estimators=10,
            min_samples_leaf=25,
            bipartite_adapter="gmosa",
            criterion="squared_error",
        ),
    ],
)
def test_cv_on_estimators(estimator, data, random_state, **PARAMS):
    return _test_cv(
        estimator=estimator.set_params(random_state=random_state),
        data=data,
    )


def test_cv_pairwise_tag(data, random_state):
    estimator = BipartiteExtraTreeRegressor(
        min_samples_leaf=25,
        random_state=random_state,
    )

    # Monkey-patch pairwise tag
    old_tags = _safe_tags(estimator)
    type(estimator)._more_tags = lambda self: old_tags | dict(pairwise=True)
    # Revert patch even if an exception is raised
    try:
        _test_cv(
            data=data,
            estimator=estimator,
            cv_params=dict(pairwise=False),  # get tag from the estimator
        )
    finally:
        type(estimator)._more_tags = lambda self: old_tags

    assert not _safe_tags(estimator, "pairwise")


def test_cv_pairwise_parameter(data, random_state):
    estimator = BipartiteExtraTreeRegressor(
        min_samples_leaf=25,
        random_state=random_state,
    )
    _test_cv(
        estimator=estimator,
        data=data,
        cv_params=dict(pairwise=True),
    )

    assert not _safe_tags(estimator, "pairwise")


def test_check_multipartite_cv_with_multipartite_cv():
    # Test that if cv is already a MultipartiteCrossValidator object,
    # the function returns it directly without any modifications
    cv = MultipartiteCrossValidator(KFold(n_splits=2), n_parts=5)
    assert check_multipartite_cv(cv) == cv


def test_check_multipartite_cv_with_n_parts():
    # Test that the function creates a MultipartiteCrossValidator object with
    # n_parts equal to the specified value when cv is an integer
    cv = check_multipartite_cv(3, n_parts=3)
    assert isinstance(cv, MultipartiteCrossValidator)
    assert len(cv.cross_validators) == 3
    assert cv.n_parts == 3


def test_check_multipartite_cv_with_list_of_cv():
    # Test that the function creates a MultipartiteCrossValidator object with
    # the specified cross-validation objects when cv is a list/tuple
    cv1 = KFold(n_splits=2)
    cv2 = KFold(n_splits=3)
    cv = check_multipartite_cv([cv1, cv2])
    assert isinstance(cv, MultipartiteCrossValidator)
    assert len(cv.cross_validators) == 2
    assert cv.cross_validators[0] == cv1
    assert cv.cross_validators[1] == cv2
    assert cv.n_parts == 2


@pytest.mark.parametrize("shuffle", [True, False])
def test_check_multipartite_cv_shuffle(data, shuffle, random_state):
    X, y = data
    n_parts = 2
    n_splits = (3, 4)

    cv = check_multipartite_cv(
        n_splits,
        y=y,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
        n_parts=n_parts,
    )

    assert isinstance(cv, MultipartiteCrossValidator)
    assert len(cv.cross_validators) == n_parts

    split = list(cv.split(X))

    assert len(split) == np.prod(n_splits)

    for ax_split in split:
        for i in zip(ax_split, y.shape, n_splits):
            (ax_train_idx, ax_test_idx), n_samples, ax_n_splits = i

            is_consecutive = (np.diff(np.array(ax_test_idx)) == 1).all()
            if shuffle:
                assert not is_consecutive
            else:
                assert is_consecutive
            assert (len(ax_test_idx) - n_samples // ax_n_splits) <= 1
            assert len(ax_train_idx) + len(ax_test_idx) == n_samples
