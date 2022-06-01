from test_utils import (
    parse_args, stopwatch, gen_mock_data, melt_2d_data,
)
from sklearn.tree._criterion import MSE
from sklearn.tree._splitter import BestSplitter

from splitter_test import test_splitter, test_splitter_nd
from hypertrees.tree._semisupervised_criterion import SSCompositeCriterion
from hypertrees.tree._nd_splitter import make_2d_splitter

import numpy as np
#from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

from pathlib import Path
from time import time
from pprint import pprint
import warnings

TOL = 1e-10

# Default test params
DEF_PARAMS = dict(
    seed=0,
    shape=(50, 60),
    nattrs=(10, 9),
    nrules=1,
    min_samples_leaf=100,
    transpose_test=False,
    noise=0,
    inspect=False,
    plot=False,
)


def eval_split(split, x, y, samples=None):
    samples = samples or np.arange(x.shape[0])
    sorted_indices = x[samples, split['feature']].argsort()
    y_left = y[sorted_indices][:split['pos']]
    y_right = y[sorted_indices][split['pos']:]
    manual_impurity_left = y_left.var()
    manual_impurity_right = y_right.var()


def test_ideal_split(**PARAMS):
    print('Started with params:')
    pprint(PARAMS)

    with stopwatch():
        XX, Y, _ = gen_mock_data(**PARAMS)
        x, y = melt_2d_data(XX, Y)

    if Y.var() == 0:
        raise RuntimeError(f"Bad seed ({PARAMS['seed']}), y is homogeneus."
                           " Try another one or reduce nrules.")

    if PARAMS['noise']:
        raise RuntimeError("'noise' param is not set to 0, we will not be able"
                           " to test for a perfect split.")

    assert np.unique(Y).shape == (2,)

    splitter = BestSplitter(
        criterion=MSE(n_outputs=y.shape[1], n_samples=x.shape[0]),
        max_features=x.shape[1],
        min_samples_leaf=y.shape[1],
        min_weight_leaf=0,
        random_state=np.random.RandomState(PARAMS['seed']),
    )

    splitter2d = make_2d_splitter(
        splitter_class=BestSplitter,
        criterion_class=MSE,
        max_features=[X.shape[1] for X in XX],
        n_samples=Y.shape,
        n_outputs=1,
    )

    # Run test
    with stopwatch('Testing 1D splitter...'):
        result = test_splitter(splitter, x, y)
        print('Best split found:')
        pprint(result)

    sorted_indices = x[:, result['feature']].argsort()
    manual_impurity_left = y[sorted_indices][:result['pos']].var()
    manual_impurity_right = y[sorted_indices][result['pos']:].var()
    assert manual_impurity_left == 0
    assert manual_impurity_right == 0

    assert result['improvement'] != 0
    assert result['impurity_left'] == 0
    assert result['impurity_right'] == 0

    # Run test 2d
    with stopwatch('Testing 2D splitter...'):
        result2 = test_splitter_nd(splitter2d, XX, Y)
        print('Best split found:')
        pprint(result2)

    assert result2['improvement'] != 0
    assert result2['impurity_left'] == 0
    assert result2['impurity_right'] == 0


def test_1d2d(**PARAMS):
    PARAMS['noise'] = PARAMS['noise'] or 0.1

    print('Started with params:')
    pprint(PARAMS)

    with stopwatch():
        XX, Y, _ = gen_mock_data(**PARAMS)
        x, y = melt_2d_data(XX, Y)

    if Y.var() == 0:
        raise RuntimeError(f"Bad seed ({PARAMS['seed']}), y is homogeneus."
                           " Try another one or reduce nrules.")

    splitter = BestSplitter(
        criterion=MSE(n_outputs=y.shape[1], n_samples=x.shape[0]),
        max_features=x.shape[1],
        min_samples_leaf=y.shape[1],
        min_weight_leaf=0,
        random_state=np.random.RandomState(PARAMS['seed']),
    )

    splitter2d = make_2d_splitter(
        splitter_class=BestSplitter,
        criterion_class=MSE,
        max_features=[X.shape[1] for X in XX],
        n_samples=Y.shape,
        n_outputs=1,
    )

    # Run test
    with stopwatch('Testing 1D splitter...'):
        result = test_splitter(splitter, x, y)
        print('Best split found:')
        pprint(result)

    sorted_indices = x[:, result['feature']].argsort()
    manual_impurity_left = y[sorted_indices][:result['pos']].var()
    manual_impurity_right = y[sorted_indices][result['pos']:].var()

    try:
        assert result['improvement'] >= 0
        assert abs(result['impurity_left'] - manual_impurity_left) < TOL
        assert abs(result['impurity_right'] - manual_impurity_right) < TOL

    except AssertionError as e:
        print('*'*50)
        print("result['impurity_left']:", result['impurity_left'])
        print("manual_impurity_left:   ", manual_impurity_left)
        print("result['impurity_right']:", result['impurity_right'])
        print("manual_impurity_right:   ", manual_impurity_right)
        print('*'*50)
        raise e

    # Run test 2d
    with stopwatch('Testing 2D splitter...'):
        result2 = test_splitter_nd(splitter2d, XX, Y)
        print('Best split found:')
        pprint(result2)

    assert result2['threshold'] == result['threshold']
    assert abs(result2['improvement'] - result['improvement']) < TOL
    assert abs(result2['impurity_left'] - result['impurity_left']) < TOL
    assert abs(result2['impurity_right'] - result['impurity_right']) < TOL


if __name__ == "__main__":
    args = parse_args(**DEF_PARAMS)
    test_1d2d(**vars(args))
