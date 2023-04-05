import warnings
import logging
import numpy as np
import pytest

from pprint import pprint
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_random_state
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.utils._testing import assert_allclose

from hypertrees.tree import BipartiteDecisionTreeRegressor, BipartiteExtraTreeRegressor

from hypertrees.ensemble import (
    BipartiteExtraTreesRegressor,
    BipartiteRandomForestRegressor,
)
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from hypertrees.tree._semisupervised_classes import (
    DecisionTreeRegressorSS,
    # BipartiteExtraTreeRegressorSS,  # TODO
)

from hypertrees.melter import (
    row_cartesian_product,
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
from make_examples import (
    make_interaction_blobs,
    make_interaction_regression,
)


DEF_PARAMS = DEF_PARAMS | dict(n_estimators=10)


def eval_model(model, X, y):
    pred = model.predict(X)
    mse = np.mean((pred-y)**2)
    print('* MSE:', mse)
    return mse


def apply_rbf_kernel(regression_data):
    X, Y, x, y = regression_data
    X = [rbf_kernel(X[0]), rbf_kernel(X[1])]
    x = row_cartesian_product(X)
    return X, Y, x, y


@pytest.fixture(params=range(3))
def random_state(request):
    return check_random_state(request.param)


@pytest.fixture
def n_samples(request):
    return (50, 30)


@pytest.fixture#(params=[(50, 30)])
def n_features(request):
    return (10, 10)


@pytest.fixture
def classification_data(n_samples, n_features, random_state):
    X, Y, x, y = make_interaction_blobs(
        return_molten=True,
        n_features=n_features,
        n_samples=n_samples,
        random_state=random_state,
        noise=0.0,
        centers=10,
        row_kwargs={'center_box': [.3, .7], 'cluster_std': .1},
        col_kwargs={'center_box': [.3, .7], 'cluster_std': .1},
    )
    return X, Y, x, y


@pytest.fixture
def regression_data(n_samples, n_features, random_state):
    X, Y, x, y = make_interaction_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=None,
        min_target=0.0,
        max_target=1.0,
        noise=0.0,
        return_molten=True,
        return_tree=False,
        random_state=random_state,
        max_depth=None,
    )
    return X, Y, x, y


@pytest.fixture
def pairwise_regression_data(regression_data):
    return apply_rbf_kernel(regression_data)


@pytest.fixture
def pairwise_classification_data(classification_data):
    return apply_rbf_kernel(classification_data)


@pytest.mark.parametrize(
    'estimator_name, estimator', [
        (
            'BipartiteRandomForestRegressorSS',
            BipartiteRandomForestRegressorSS(
                unsupervised_criterion_rows='mean_distance',
                unsupervised_criterion_cols='mean_distance',
            ),
        ),
    ],
)
def test_all_regressors(
    regression_data,
    pairwise_regression_data,
    estimator_name,
    estimator,
):
    if estimator._get_tags()['pairwise']:
        X, Y, x, y = pairwise_regression_data
    else:
        X, Y, x, y = regression_data

    estimator.fit(X, Y)

    # The two formats must be supported:
    pred1 = estimator.predict(X)
    try:
        pred2 = estimator.predict(x)
        assert_allclose(pred1, pred2)
    except ValueError:
        warnings.warn(f"{estimator_name} does not accept melted input.")

    score = estimator.score(X, y)

    logging.info(f"{estimator_name}'s score: {score}")
    assert score > 0.1


@pytest.mark.parametrize(
    'estimator_name, estimator', [],
)
def test_all_classifiers(
    classification_data,
    pairwise_classification_data,
    estimator_name,
    estimator,
):
    estimator = estimator()
    dummy = DummyClassifier()

    if estimator._get_tags()['pairwise']:
        X, Y, x, y = pairwise_classification_data
    else:
        X, Y, x, y = classification_data

    estimator.fit(X, Y)
    dummy.fit(x, y)

    # The two formats must be supported:
    pred1 = estimator.predict(X)
    try:
        pred2 = estimator.predict(x)
        assert_allclose(pred1, pred2)
    except ValueError:
        warnings.warn(f"{estimator_name} does not accept melted input.")

    score = estimator.score(X, y)

    logging.info(f"{estimator_name}'s score: {score}")
    assert score > dummy.score(x, y)


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
        bootstrap=True,
        max_samples=0.7,
    ) | forest_params
    pprint(forest_params)

    estimators1d = {
        'SSRF': RandomForestRegressorSS(**forest_params),
        'SSXT': ExtraTreesRegressorSS(**forest_params),
        'SSXT2': ExtraTreesRegressorSS(**forest_params),
        'SSXT3': ExtraTreesRegressorSS(
            unsupervised_criterion='mean_distance',
            **forest_params,
        ),
    }
    estimators2d = {
        'SSBRF': BipartiteRandomForestRegressorSS(**forest_params),
        'SSBXT': BipartiteExtraTreesRegressorSS(**forest_params),
        'SSBXTado': BipartiteExtraTreesRegressorSS(
            axis_decision_only=True,
            **forest_params,
        ),
        'SSBXT_mean_dist': BipartiteExtraTreesRegressorSS(
            unsupervised_criterion_rows='mean_distance',
            unsupervised_criterion_cols='mean_distance',
            **forest_params,
        ),
    }
    compare_estimators(estimators1d, estimators2d, **PARAMS)
