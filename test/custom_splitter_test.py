from pprint import pprint
from sklearn.tree._criterion import MSE
from sklearn.tree._splitter import BestSplitter

from splitter_test import test_splitter, test_splitter2d
from _nd_splitter import Splitter2D
from _nd_criterion import MSE_Wrapper2D, MSE2D, teste


import numpy as np
#from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

from pathlib import Path
import sys
from time import time

path_pbct_test = Path('../../../scripts/predictors/PBCT/tests/')
sys.path.insert(0, str(path_pbct_test))
from make_examples import gen_imatrix


##### TEST PARAMS #####
CONFIG = dict(
    seed=None,
    shape=(500, 600),
    nattrs=(10, 16),
    nrules=1,
    transpose_test=0,
    noise=0,
    inspect=0,
)
#######################


## Generate mock data
print('Starting with settings:')
pprint(CONFIG)

np.random.seed(CONFIG['seed'])
t0 = time()
XX, Y = gen_imatrix(CONFIG['shape'], CONFIG['nattrs'], nrules=CONFIG['nrules'])

if CONFIG['transpose_test']:
    print('Test transposing axis.')
    Y = np.copy(Y.T.astype(DOUBLE_t), order='C')
    XX = [np.ascontiguousarray(X, DTYPE_t) for X in XX[::-1]]
    CONFIG['nattrs'] = CONFIG['nattrs'][::-1]
    CONFIG['shape'] = CONFIG['shape'][::-1]
else:
    Y = np.ascontiguousarray(Y, DOUBLE_t)
    XX = [np.ascontiguousarray(X, DTYPE_t) for X in XX]

if CONFIG['noise']:
    Y += np.random.rand(*CONFIG['shape'])  # add some noise

print('Data generation time:', time()-t0)
print('Data density (mean):', Y.mean())
print('Data variance:', Y.var())
print('=' * 50)
## Instantiate sklearn objects

## 1D Criteria:
# criterion_rows = MSE(n_outputs=CONFIG['shape'][1], n_samples=CONFIG['shape'][0])
# criterion_cols = MSE(n_outputs=CONFIG['shape'][0], n_samples=CONFIG['shape'][1])


## 2D Criteria:

## NOTE: criterion_rows: n_outputs >= CONFIG['shape'][1]+31 gives segfault.
## That is due to memory garbage. n_output higher than 2 allocates uninitialized
## memory space.
## CONFIG['seed']=5, CONFIG['shape'] = (500, 600) CONFIG['nattrs']=(10, 16) CONFIG['nrules']=1, CONFIG['transpose_test']=0
criterion_rows = MSE2D(n_outputs=1, n_samples=CONFIG['shape'][0])
criterion_cols = MSE2D(n_outputs=1, n_samples=CONFIG['shape'][1])
print(id(criterion_rows.weighted_n_cols))
print(id(criterion_cols.weighted_n_cols))

# n_outputs=2 because ...

splitter_rows = BestSplitter(
    criterion=criterion_rows,
    max_features=CONFIG['nattrs'][0],
    min_samples_leaf=1,
    min_weight_leaf=0,
    random_state=np.random.RandomState(CONFIG['seed']),
)
splitter_cols = BestSplitter(
    criterion=criterion_cols,
    max_features=CONFIG['nattrs'][1],
    min_samples_leaf=1,
    min_weight_leaf=0,
    random_state=np.random.RandomState(CONFIG['seed']),
)

criterion_wrapper = MSE_Wrapper2D([
    splitter_rows.criterion, splitter_cols.criterion])

splitter = Splitter2D(
    splitter_rows, splitter_cols, criterion_wrapper)

# Run test
t0 = time()
result = test_splitter2d(splitter, XX, Y)
# print('Best split found:')
# pprint(result)
print('Time:', time()-t0)
pprint(result)

if CONFIG['inspect']:
    axis = result['axis']
    pos = result['pos']
    feature = result['feature']

    if axis == 1:
        feature -= CONFIG['nattrs'][0]

    mask = XX[axis][:, feature] > result['threshold']
    indices = np.arange(Y.shape[axis])

    y1 = np.take(Y, indices[mask], axis)
    y2 = np.take(Y, indices[~mask], axis)

    sorted_indices = XX[axis][:, feature].argsort()
    y12 = np.take(Y, sorted_indices[:pos], axis)
    y22 = np.take(Y, sorted_indices[pos:], axis)

    breakpoint()
