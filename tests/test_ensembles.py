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
from sklearn.base import is_classifier

from bipartite_learn.tree import (
    BipartiteDecisionTreeRegressor,
    BipartiteExtraTreeRegressor,
)

from bipartite_learn.ensemble import (
    BipartiteExtraTreesRegressor,
    BipartiteRandomForestRegressor,
)
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from bipartite_learn.tree._semisupervised_classes import (
    DecisionTreeRegressorSS,
    BipartiteExtraTreeRegressorSS,
)

from bipartite_learn.melter import (
    row_cartesian_product,
)
from bipartite_learn.ensemble._semisupervised_forest import (
    RandomForestRegressorSS,
    ExtraTreesRegressorSS,
    BipartiteRandomForestRegressorSS,
    BipartiteExtraTreesRegressorSS,
    # BipartiteRandomForestClassifierSS,
)

from .utils.test_utils import (
    gen_mock_data, stopwatch, DEF_PARAMS, parse_args,
)
from .utils.make_examples import (
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
    return (11, 13)


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


@pytest.fixture
def get_appropriate_data(
    classification_data,
    regression_data,
    pairwise_classification_data,
    pairwise_regression_data,
):
    def _getter(estimator):
        if is_classifier(estimator):
            if estimator._get_tags()['pairwise']:
                return pairwise_classification_data
            else:
                return classification_data
        else:
            if estimator._get_tags()['pairwise']:
                return pairwise_regression_data
            else:
                return regression_data
    return _getter


@pytest.mark.parametrize(
    'estimator_name, estimator', [
        (
            'BipartiteRandomForestRegressorSS',
            BipartiteRandomForestRegressorSS(
                unsupervised_criterion_rows='squared_error',
                unsupervised_criterion_cols='squared_error',
            ),
        ),
        (
            'BipartiteRandomForestRegressorSS',
            BipartiteRandomForestRegressorSS(
                unsupervised_criterion_rows='mean_distance',
                unsupervised_criterion_cols='mean_distance',
            ),
        ),
    ],
)
def test_forests_score(
    get_appropriate_data,
    estimator_name,
    estimator,
):
    X, Y, x, y = get_appropriate_data(estimator)

    estimator.fit(X, Y)

    # The two formats must be supported:
    pred1 = estimator.predict(X)
    pred2 = estimator.predict(x)
    assert_allclose(pred1, pred2)

    score = estimator.score(X, y)

    logging.info(f"{estimator_name}'s score: {score}")
    assert score > 0.1


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
    }
    estimators2d = {
        'SSBRF': BipartiteRandomForestRegressorSS(**forest_params),
        'SSBXT': BipartiteExtraTreesRegressorSS(**forest_params),
        'SSBXTado': BipartiteExtraTreesRegressorSS(
            axis_decision_only=True,
            **forest_params,
        ),
    }
    compare_estimators(estimators1d, estimators2d, **PARAMS)


@pytest.mark.parametrize(
    'forest_class',
    [
        RandomForestRegressorSS,
        BipartiteRandomForestRegressorSS,
    ],
)
def test_ss_forest_can_set_X_targets(regression_data, forest_class):
    def preprocessor_func(x):
        return x ** 2

    X, Y, x, y = regression_data

    forest = forest_class(
        preprocess_X_targets=preprocessor_func,
        n_estimators=5,
    )

    is_bipartite = forest._get_tags().get('multipartite', False)

    if is_bipartite:
        X_processed = [preprocessor_func(Xi) for Xi in X]
    else:
        X = X[0]
        X_processed = preprocessor_func(X)

    forest.fit(X, Y)

    tree = forest.estimators_[0]
    assert tree.preprocess_X_targets is preprocessor_func
    assert forest._X_targets is not None
    assert tree._X_targets is not None
    assert tree._X_targets is forest._X_targets

    if is_bipartite:
        assert (tree._X_targets[0] == X_processed[0]).all()
        assert (tree._X_targets[1] == X_processed[1]).all()
    else:
        assert (tree._X_targets == X_processed).all()
