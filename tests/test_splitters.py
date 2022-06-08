from test_utils import (
    parse_args, stopwatch, gen_mock_data, melt_2d_data,
)
from sklearn.tree._criterion import MSE
from sklearn.tree._splitter import BestSplitter

from splitter_test import test_splitter, test_splitter_nd
from hypertrees.tree._semisupervised_criterion import SSCompositeCriterion
from hypertrees.tree._nd_splitter import make_2d_splitter
from hypertrees.tree._semisupervised_criterion import make_2dss_splitter

import numpy as np
#from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

from pathlib import Path
from time import time
from pprint import pprint
import warnings

# Default test params
DEF_PARAMS = dict(
    seed=0,
    shape=(50, 60),
    nattrs=(10, 9),
    nrules=1,
    min_samples_leaf=100,
    transpose_test=False,
    noise=.5,
    inspect=False,
    plot=False,
    start=0,
    end=0,
)


def compare_splitters_1d2d_ideal(
    splitter1d,
    splitter2d,
    tol=0,
    **PARAMS,
):
    PARAMS = DEF_PARAMS | dict(noise=0) | PARAMS

    if PARAMS['noise']:
        warnings.warn(f"noise={PARAMS['noise']}. Setting it to zero"
                      " since noise=0 is what defines an ideal split.")
        PARAMS['noise'] = 0
    
    result1d, result2d = compare_splitters_1d2d(
        splitter1d, splitter2d, tol, **PARAMS)

    assert result1d['improvement'] != 0
    assert result1d['impurity_left'] == 0
    assert result1d['impurity_right'] == 0

    assert result2d['improvement'] != 0
    assert result2d['impurity_left'] == 0
    assert result2d['impurity_right'] == 0


def compare_splitters_1d2d(
    splitter1d,
    splitter2d,
    tol=1e-10,
    **PARAMS,
):
    PARAMS = DEF_PARAMS | PARAMS

    print('Started with params:')
    pprint(PARAMS)

    with stopwatch():
        XX, Y, _ = gen_mock_data(**PARAMS)
        x, y = melt_2d_data(XX, Y)

    if Y.var() == 0:
        raise RuntimeError(f"Bad seed ({PARAMS['seed']}), y is homogeneus."
                           " Try another one or reduce nrules.")

    start = PARAMS['start'] or 0
    end = PARAMS['end'] or Y.shape[0]

    if (not isinstance(start, int)) or (not isinstance(end, int)):
        raise TypeError(f"2D start/end not possible. start ({repr(start)}) and end "
                        f"({repr(end)}) must be integers.")

    if isinstance(splitter1d, type):
        splitter1d = splitter1d(
            criterion=MSE(n_outputs=y.shape[1], n_samples=x.shape[0]),
            max_features=x.shape[1],
            min_samples_leaf=y.shape[1],
            min_weight_leaf=0,
            random_state=np.random.RandomState(PARAMS['seed']),
        )

    if isinstance(splitter2d, type):
        splitter2d = make_2d_splitter(
            splitters=splitter2d,
            criteria=MSE,
            max_features=[X.shape[1] for X in XX],
            n_samples=Y.shape,
            n_outputs=1,
        )

    # Run test
    with stopwatch(f'Testing 1D splitter ({splitter1d.__class__.__name__})...'):
        result1d = test_splitter(
            splitter1d, x, y, start=start*Y.shape[1], end=end*Y.shape[1])
        print('Best split found:')
        pprint(result1d)

    x_ = x[start*Y.shape[1] : end*Y.shape[1]]
    y_ = y[start*Y.shape[1] : end*Y.shape[1]]
    pos = result1d['pos'] - start*Y.shape[1]

    sorted_indices = x_[:, result1d['feature']].argsort()
    manual_impurity_left = y_[sorted_indices][:pos].var()
    manual_impurity_right = y_[sorted_indices][pos:].var()

    assert result1d['improvement'] >= 0, \
        'Negative reference improvement, input seems wrong.'
    assert abs(result1d['impurity_left']-manual_impurity_left) <= tol, \
        'Wrong reference impurity left.'
    assert abs(result1d['impurity_right']-manual_impurity_right) <= tol, \
        'Wrong reference impurity right.'

    # Run test 2d
    with stopwatch(f'Testing 2D splitter ({splitter2d.__class__.__name__})...'):
        result2d = test_splitter_nd(
            splitter2d, XX, Y, start=[start, 0], end=[end, Y.shape[1]])
        print('Best split found:')
        pprint(result2d)

    assert result2d['threshold'] == result1d['threshold'], \
        'threshold differs from reference.'
    assert abs(result2d['improvement']-result1d['improvement']) <= tol, \
        'improvement differs from reference.'
    assert abs(result2d['impurity_left']-result1d['impurity_left']) <= tol, \
        'impurity_left differs from reference.'
    assert abs(result2d['impurity_right']-result1d['impurity_right']) <= tol, \
        'impurity_right differs from reference.'
    
    return result1d, result2d


def splitter_test_update_pos():
    pass


def test_1d2d_ideal(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    return compare_splitters_1d2d_ideal(
        splitter1d=BestSplitter,
        splitter2d=BestSplitter,
        **PARAMS,
    )


def test_1d2d(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    return compare_splitters_1d2d(
        splitter1d=BestSplitter,
        splitter2d=BestSplitter,
        **PARAMS,
    )


def test_ss_1d2d(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    ss2d_splitter = make_2dss_splitter(
        splitters = BestSplitter,
        criteria = MSE,
        supervision=1.,
        max_features=PARAMS['nattrs'],
        n_features=PARAMS['nattrs'],
        n_samples=PARAMS['shape'],
        n_outputs=1,
    )

    return compare_splitters_1d2d(
        splitter1d=BestSplitter,
        splitter2d=ss2d_splitter,
        **PARAMS,
    )


def test_ss_1d2d_ideal_split(**PARAMS):
    PARAMS = DEF_PARAMS | PARAMS
    ss2d_splitter = make_2dss_splitter(
        splitters = BestSplitter,
        criteria = MSE,
        supervision=1.,
        max_features=PARAMS['nattrs'],
        n_features=PARAMS['nattrs'],
        n_samples=PARAMS['shape'],
        n_outputs=1,
    )

    return compare_splitters_1d2d_ideal(
        splitter1d=BestSplitter,
        splitter2d=ss2d_splitter,
        **PARAMS,
    )


if __name__ == "__main__":
    args = parse_args(**DEF_PARAMS)
    # test_1d2d_ideal(**vars(args))
    test_1d2d(**vars(args))
    test_ss_1d2d(**vars(args))
    test_ss_1d2d_ideal_split(**vars(args))
