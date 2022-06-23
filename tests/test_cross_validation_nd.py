from pprint import pprint

from hypertrees.model_selection import cross_validate_nd
from hypertrees.tree import DecisionTreeRegressor2D
from hypertrees.ensemble import ExtraTreesRegressor2D, RandomForestRegressor2D

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

DEF_PARAMS |= LOCAL_DEF_PARAMS


def test_cv(tree=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    PARAMS['noise'] = 0.
    pprint(PARAMS)

    with stopwatch('Generating data...'):
        X, y, _ = gen_mock_data(**PARAMS)
    
    if y.var() == 0:
        raise RuntimeError(f'y is homogeneous, try another seed.')

    if tree is None:
        tree = DecisionTreeRegressor2D(
            min_samples_leaf=PARAMS['min_samples_leaf'],
        )
    
    cv_result = cross_validate_nd(
        estimator=tree,
        X=X,
        y=y,
        groups=None,
        scoring="roc_auc",
        cv=3,
        n_jobs=None,
        verbose=10,
        fit_params=None,
        pre_dispatch="2*n_jobs",
        return_train_score=False,
        return_estimator=False,
        error_score='raise',
        diagonal=False,
        train_test_combinations=None,
    )

    pprint(cv_result)

    if PARAMS['inspect']:
        breakpoint()


def test_cv_pairwise(tree=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    PARAMS['nattrs'] = PARAMS['shape']  # Make it square.

    tree = tree or DecisionTreeRegressor2D(
        min_samples_leaf=PARAMS['min_samples_leaf'],
    )
    old_tags = tree._more_tags()
    tree.__class__._more_tags = lambda self: old_tags | dict(pairwise=True)

    try:
        test_cv(tree=tree, **PARAMS)
    finally:
        tree.__class__._more_tags = lambda self: old_tags


def test_cv_ensemble(tree=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    if tree is None:
        tree = ExtraTreesRegressor2D(
            min_samples_leaf=PARAMS['min_samples_leaf']
        )
    test_cv(tree=tree, **PARAMS)


def test_cv_pairwise_ensemble(tree=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    PARAMS['nattrs'] = PARAMS['shape']  # Make it square.

    if tree is None:
        tree = ExtraTreesRegressor2D(
            min_samples_leaf=PARAMS['min_samples_leaf']
        )
    old_tags = tree._more_tags()
    tree.__class__._more_tags = lambda self: old_tags | dict(pairwise=True)

    try:
        test_cv(tree=tree, **PARAMS)
    finally:
        tree.__class__._more_tags = lambda self: old_tags


def main(**PARAMS):
    test_cv(**PARAMS)
    test_cv_pairwise(**PARAMS)
    test_cv_ensemble(**PARAMS)
    test_cv_pairwise_ensemble(**PARAMS)


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))