from splitter_test import test_splitter
from hypertrees.tree._semisupervised_criterion import \
    SSCompositeCriterion, SSMSE

from time import time
from pprint import pprint
from contextlib import contextmanager

import numpy as np
from sklearn.tree._criterion import MSE
from sklearn.tree._splitter import BestSplitter
DTYPE_t, DOUBLE_t = np.float32, np.float64


SEED = 2


@contextmanager
def stopwatch():
    t0 = time()
    try:
        yield
    finally:
        print(f"Took {time()-t0} seconds.")


def split_equality(result, other):
    return \
        result['pos'] == other['pos'] and \
        result['improvement'] == other['improvement'] and \
        result['impurity_left'] == other['impurity_left'] and \
        result['impurity_right'] == other['impurity_right']


# Generate mock data ==========================================================

with stopwatch():
    shape, nattrs = (200, 100), (10, 1)
    targetcol = np.random.randint(nattrs[0])
    X = np.random.rand(shape[0], nattrs[0]).astype(DTYPE_t)
    y = np.zeros((shape[0], 1), DOUBLE_t)
    # y += np.random.rand(*y.shape)*.1  # Optionally add noise.
    y[X[:, targetcol] < .5, :] = 1.
    Xy = np.hstack([X, y])
    print("y.shape, X.shape, targetcol", y.shape, X.shape, targetcol)


# Define criterion objects  ===================================================

criteria = dict(
    mse_s=lambda: MSE(n_outputs=y.shape[1], n_samples=y.shape[0]),
    mse_u=lambda: MSE(n_outputs=X.shape[1], n_samples=X.shape[0]),
)

def ss_composite_s():
    return SSCompositeCriterion(
        supervised_criterion=MSE(n_outputs=y.shape[1], n_samples=y.shape[0]),
        unsupervised_criterion=MSE(n_outputs=X.shape[1], n_samples=X.shape[0]),
        supervision=1.,
    )

def ss_composite_u():
    return SSCompositeCriterion(
        supervised_criterion=MSE(n_outputs=y.shape[1], n_samples=y.shape[0]),
        unsupervised_criterion=MSE(n_outputs=X.shape[1], n_samples=X.shape[0]),
        supervision=0.,
    )


criteria['ss_composite_u'] = ss_composite_u
criteria['ss_composite_s'] = ss_composite_s

criteria['ssmse_s'] = lambda: SSMSE(
    n_samples=X.shape[0],
    n_features=X.shape[1],
    n_outputs=y.shape[1],
    supervision=1.,
)
criteria['ssmse_u'] = lambda: SSMSE(
    n_samples=X.shape[0],
    n_features=X.shape[1],
    n_outputs=y.shape[1],
    supervision=0.,
)


# Collect data ================================================================

splits = {}
for name, criterion in criteria.items():
    criterion = criterion()
    print(f'*** Testing {criterion.__class__.__name__} ({name})...')
    y_ = Xy if name.startswith('ss') else y
    print('Making splitter...')
    splitter = BestSplitter(
        criterion=criterion,
        max_features=X.shape[1],
        min_samples_leaf=1,
        min_weight_leaf=0,
        random_state=np.random.RandomState(0),
    )
    print('Testing...')
    with stopwatch():
        result = test_splitter(splitter, X, y_)
        print('Best split found:')
        pprint(result)

    splits[name] = result


# Test ========================================================================

assert split_equality(splits['ss_composite_s'], splits['mse_s'])
assert split_equality(splits['ssmse_s'], splits['mse_s'])
assert split_equality(splits['ss_composite_u'], splits['ssmse_u'])
# assert split_equality(splits['ss_composite_u'], splits['mse_u'])
# assert split_equality(splits['ssmse_u'], splits['mse_u'])