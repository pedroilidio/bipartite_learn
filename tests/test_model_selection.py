from pprint import pprint
import numpy as np

from make_examples import make_interaction_data
from test_nd_classes import parse_args
from hypertreestree import DecisionTreeRegressor2D
from hypertreesmodel_selection._split import check_cv_nd, CrossValidatorNDWrapper
from hypertreesmodel_selection._search import GridSearchCVND, RandomizedSearchCVND
from test_ensembles import print_duration, test_compare_estimators


DEF_PARAMS = dict(
    seed=7,
    shape=(50, 60),
    nattrs=(10, 9),
    nrules=10,
    min_samples_leaf=100,
    transpose_test=False,
    noise=0.1,
    inspect=False,
    plot=False,
    save_trees=False,
)


def test_cv_wrapper(cv=3, **kwargs):
    kwargs = DEF_PARAMS | kwargs
    print('Starting with parameters:')
    pprint(kwargs)

    rng = np.random.default_rng(kwargs['seed'])

    print('Making mock interaction data...')
    with print_duration():
        X, y, _ = make_interaction_data(
            shape=kwargs['shape'],
            nattrs=kwargs['nattrs'],
            nrules=kwargs['nrules'],
            noise=kwargs['noise'],
            random_state=rng,
        )

    cv = check_cv_nd(cv, y, classifier=False, diagonal=False)
    splits = list(cv.split(X, y))
    print(len(splits))
    print(len(splits[0]))
    print(len(splits[0][0]))
    for s in cv.split(X, y):
        # print(len(s[0][0]), len(s[0][1]))
        print(f'{s[0][0][0]}-{s[0][0][-1]}, {s[0][1][0]}-{s[0][1][-1]}' )
        print(f'{s[1][0][0]}-{s[1][0][-1]}, {s[1][1][0]}-{s[1][1][-1]}' )
        print()


def test_search_cv(cv=3, **kwargs):
    est = GridSearchCVND(
        estimator=DecisionTreeRegressor2D(),
        param_grid=dict(
            min_samples_leaf=[10, 50, 100],
        ),
        scoring=["roc_auc", "average_precision"],
        verbose=10,
        refit="roc_auc",
    )
    test_compare_estimators(
        estimators2d={'GridSearchCVND': est},
        estimators1d={},
        **kwargs
    )

    if kwargs['inspect']:
        breakpoint()


if __name__ == '__main__':
    args = parse_args(**DEF_PARAMS)
    test_search_cv(
        **vars(args),
    )
