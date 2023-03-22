from pprint import pprint
import numpy as np
import pytest

from hypertrees.tree import BipartiteDecisionTreeRegressor, BipartiteExtraTreeRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from hypertrees.ensemble import (
    BipartiteExtraTreesRegressor,
    BipartiteRandomForestRegressor,
)
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from hypertrees.tree._semisupervised_classes import (
    DecisionTreeRegressorSS,
    # BipartiteExtraTreeRegressorSS,  # TODO
)

from hypertrees.ensemble._semisupervised_forest import (
    RandomForestRegressorSS,
    ExtraTreesRegressorSS,
    BipartiteRandomForestRegressorSS,
    BipartiteExtraTreesRegressorSS,
)

from test_utils import (
    gen_mock_data, stopwatch, DEF_PARAMS, parse_args,
)

DEF_PARAMS = DEF_PARAMS | dict(n_estimators=10)


def eval_model(model, X, y):
    pred = model.predict(X)
    mse = np.mean((pred-y)**2)
    print('* MSE:', mse)
    return mse


def compare_estimators(estimators1d, estimators2d, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    X, y, X1d, y1d = gen_mock_data(melt=True, **PARAMS)
    y1d = y1d.reshape(-1)

    for name in estimators1d.keys():
        with stopwatch(f'Fitting {name}...'):
            estimators1d[name].fit(X1d, y1d)
    for name in estimators2d.keys():
        with stopwatch(f'Fitting {name}...'):
            estimators2d[name].fit(X, y)

    yvar = y.var()
    print('\nEval baseline, var(y): ', yvar)

    for name, est in (estimators1d|estimators2d).items():
        with stopwatch(f'Evaluating {name}...'):
            assert eval_model(est, X1d, y1d) <= yvar

    return estimators1d, estimators2d


def test_trees(tree_params=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    tree_params = tree_params or {}
    tree_params = dict(
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    ) | tree_params

    estimators1d = {
        'DT': DecisionTreeRegressor(**tree_params),
        'sXT': ExtraTreeRegressor(**tree_params),
    }
    estimators2d = {
        'DT2D': BipartiteDecisionTreeRegressor(**tree_params),
        'sXT2D': BipartiteExtraTreeRegressor(**tree_params),
    }
    compare_estimators(estimators1d, estimators2d, **PARAMS)


def test_forests(forest_params=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    forest_params = forest_params or {}
    forest_params = dict(
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
        n_estimators=PARAMS['n_estimators'],
    ) | forest_params

    estimators1d = {
        'RF': RandomForestRegressor(**forest_params),
        'XT': ExtraTreesRegressor(**forest_params),
    }
    estimators2d = {
        'BRF': BipartiteRandomForestRegressor(**forest_params),
        'BXT': BipartiteExtraTreesRegressor(**forest_params),
    }
    compare_estimators(estimators1d, estimators2d, **PARAMS)


def test_semisupervised_forests(forest_params=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    forest_params = forest_params or {}
    forest_params = dict(
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
        n_estimators=PARAMS['n_estimators'],
        bootstrap=False,
        # max_samples=None,
    ) | forest_params
    pprint(forest_params)

    estimators1d = {
        'SSRF': RandomForestRegressorSS(**forest_params),
        'SSXT': ExtraTreesRegressorSS(**forest_params),
    }
    estimators2d = {
        'SSBRF': BipartiteRandomForestRegressorSS(**forest_params),
        'SSBXT': BipartiteExtraTreesRegressorSS(**forest_params),
    }
    compare_estimators(estimators1d, estimators2d, **PARAMS)
