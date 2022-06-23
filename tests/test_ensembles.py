from pprint import pprint
import numpy as np

from hypertrees.tree import DecisionTreeRegressor2D, ExtraTreeRegressor2D
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from hypertrees.ensemble import ExtraTreesRegressor2D, RandomForestRegressor2D
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from hypertrees.tree._semisupervised_classes import (
    DecisionTreeRegressorSS,
    DecisionTreeRegressorDS,
    ExtraTreeRegressor2DSS,
    ExtraTreeRegressor2DDS,
)

from hypertrees.ensemble._semisupervised_forest import (
    RandomForestRegressorSS,
    ExtraTreesRegressorSS,
    RandomForestRegressor2DSS,
    ExtraTreesRegressor2DSS,
)

from hypertrees.melter import row_cartesian_product, MelterND
from make_examples import make_interaction_data
from test_utils import (
    gen_mock_data, melt_2d_data, stopwatch, DEF_PARAMS, parse_args,
)


def eval_model(model, X, y):
    pred = model.predict(X)
    mse = np.mean((pred-y)**2)
    print('MSE:', mse)
    return mse


def compare_estimators(estimators1d, estimators2d, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    X, y, X1d, y1d, _ = gen_mock_data(melt=True, **PARAMS)
    y1d = y1d.reshape(-1)

    for name in estimators1d.keys():
        if isinstance(estimators1d[name], type):
            estimators1d[name] = estimators1d[name]()
        with stopwatch(f'Fitting {name}...'):
            estimators1d[name].fit(X1d, y1d)
    for name in estimators2d.keys():
        if isinstance(estimators2d[name], type):
            estimators2d[name] = estimators2d[name]()
        with stopwatch(f'Fitting {name}...'):
            estimators2d[name].fit(X, y)

    yvar = y.var()
    print('\nEval baseline, var(y): ', yvar)

    for name, est in (estimators1d|estimators2d).items():
        with stopwatch(f'Evaluating {name}...'):
            assert eval_model(est, X1d, y1d) < yvar

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
        'DT2D': DecisionTreeRegressor2D(**tree_params),
        'sXT2D': ExtraTreeRegressor2D(**tree_params),
    }
    return compare_estimators(estimators1d, estimators2d, **PARAMS)


def test_forests(forest_params=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    forest_params = forest_params or {}
    forest_params = dict(
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    ) | forest_params

    estimators1d = {
        'RF': RandomForestRegressor(**forest_params),
        'XT': ExtraTreesRegressor(**forest_params),
    }
    estimators2d = {
        'RF2D': RandomForestRegressor2D(**forest_params),
        'XT2D': ExtraTreesRegressor2D(**forest_params),
    }
    return compare_estimators(estimators1d, estimators2d, **PARAMS)


def test_semisupervised_forests(forest_params=None, **PARAMS):
    PARAMS = DEF_PARAMS | PARAMS

    forest_params = forest_params or {}
    forest_params = dict(
        min_samples_leaf=PARAMS['min_samples_leaf'],
        random_state=PARAMS['seed'],
    ) | forest_params

    estimators1d = {
        'SSRF': RandomForestRegressorSS(**forest_params),
        'SSXT': ExtraTreesRegressorSS(**forest_params),
        'DSXT': ExtraTreesRegressorSS(
            criterion='dynamic_ssmse', **forest_params),
    }
    estimators2d = {
        'SSRF2D': RandomForestRegressor2DSS(**forest_params),
        'SSXT2D': ExtraTreesRegressor2DSS(**forest_params),
        'DSXT2D': ExtraTreesRegressor2DSS(
            ss_criterion='dynamic_ssmse', **forest_params),
    }
    return compare_estimators(estimators1d, estimators2d, **PARAMS)


def main(**PARAMS):
    test_trees(**PARAMS)
    test_forests(**PARAMS)
    test_semisupervised_forests(**PARAMS)


if __name__ == '__main__':
    args = parse_args(**DEF_PARAMS)
    main(**vars(args))
