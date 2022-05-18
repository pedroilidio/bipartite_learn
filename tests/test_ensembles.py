from contextlib import contextmanager
from time import process_time
from pprint import pprint
import numpy as np

from hypertrees.tree import DecisionTreeRegressor2D, ExtraTreeRegressor2D
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from hypertrees.ensemble import ExtraTreesRegressor2D, RandomForestRegressor2D
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from hypertrees.melter import row_cartesian_product, MelterND
from make_examples import make_interaction_data
from test_nd_classes import parse_args


@contextmanager
def print_duration():
    t0 = process_time()
    yield
    print(f"Finished in {process_time()-t0} CPU seconds.")


def eval_model(model, X, y):
    #X, y = MelterND().fit_resample(X, y)
    pred = model.predict(X)
    print('MSE:', np.mean((pred-y)**2))


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

def test_compare_estimators(estimators1d, estimators2d, **kwargs):
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
        X1d, y1d = MelterND().fit_resample(X, y)
        X1d = X1d.astype(np.float32)

    for name in estimators1d.keys():
        if isinstance(estimators1d[name], type):
            estimators1d[name] = estimators1d[name]()
        print(f'Fitting {name}...')
        with print_duration():
            estimators1d[name].fit(X1d, y1d)
    for name in estimators2d.keys():
        if isinstance(estimators2d[name], type):
            estimators2d[name] = estimators2d[name]()
        print(f'Fitting {name}...')
        with print_duration():
            estimators2d[name].fit(X, y)

    print('\nEval baseline, var(y): ', y.var())
    for name, est in (estimators1d|estimators2d).items():
        print(f'Evaluating {name}...')
        with print_duration():
            eval_model(est, X1d, y1d)

    return estimators1d, estimators2d


def test_forests(forest_params=None, **kwargs):
    if forest_params is None:
        forest_params = {}
    estimators1d = {
        'RF': RandomForestRegressor(**forest_params),
        'XT': ExtraTreesRegressor(**forest_params),
    }
    estimators2d = {
        'RF2D': RandomForestRegressor2D(**forest_params),
        'XT2D': ExtraTreesRegressor2D(**forest_params),
    }
    return test_compare_estimators(estimators1d, estimators2d, **kwargs)


def test_trees(tree_params=None, **kwargs):
    if tree_params is None:
        tree_params = {}
    estimators1d = {
        'DT': DecisionTreeRegressor(**tree_params),
        'sXT': ExtraTreeRegressor(**tree_params),
    }
    estimators2d = {
        'DT2D': DecisionTreeRegressor2D(**tree_params),
        'sXT2D': ExtraTreeRegressor2D(**tree_params),
    }
    return test_compare_estimators(estimators1d, estimators2d, **kwargs)


if __name__ == '__main__':
    args = parse_args(**DEF_PARAMS)
    # test_trees(
    #     tree_params=dict(min_samples_leaf=100),
    #     **vars(args),
    # )
    test_forests(
        forest_params=dict(n_estimators=5, min_samples_leaf=100),
        **vars(args),
    )
