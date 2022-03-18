from time import time
from pprint import pprint
import numpy as np

from hypertree.model_selection import cross_validate_nd
from hypertree.tree import DecisionTreeRegressor2D
from make_examples import gen_imatrix

DTYPE_t, DOUBLE_t = np.float32, np.float64

##### TEST PARAMS #####
CONFIG = dict(
    seed=439,
    shape=(510, 609),
    nattrs=(10, 9),
    nrules=10,
    transpose_test=0,
    noise=0,
    inspect=1,
    plot=0,
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

########## Instantiate sklearn objects
print('Data generation time:', time()-t0)
tree = DecisionTreeRegressor2D(
    min_samples_leaf=(100 if CONFIG['noise'] else 1),
)
print(vars(tree))
cv_result = cross_validate_nd(
    estimator=tree,
    X=XX,
    y=Y,
    groups=None,
    scoring="roc_auc",
    cv=3,
    n_jobs=None,
    verbose=10,
    fit_params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    error_score='raise',
    diagonal=False,
    train_test_combinations=None,
)

print(cv_result)
if CONFIG['inspect']:
    breakpoint()
