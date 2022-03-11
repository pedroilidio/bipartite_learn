from itertools import product
from pprint import pprint
from sklearn.tree._criterion import MSE
from sklearn.tree._splitter import BestSplitter
import sklearn.tree

from splitter_test import test_splitter, test_splitter2d
from _nd_splitter import Splitter2D
from _nd_criterion import MSE_Wrapper2D
import matplotlib.pyplot as plt
from _nd_classes import DecisionTreeRegressor2D


import numpy as np
#from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

from pathlib import Path
import sys
from time import time

path_pbct_test = Path('../PBCT/tests/')
sys.path.insert(0, str(path_pbct_test))
from make_examples import gen_imatrix


##### TEST PARAMS #####
CONFIG = dict(
    seed=437819,
    shape=(510, 609),
    nattrs=(10, 9),
    nrules=10,
    transpose_test=0,
    noise=1,
    inspect=0,
    plot=1,
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
tree = DecisionTreeRegressor2D()
print(vars(tree))
print('Fitting tree...')
tree.fit(XX, Y)
x_gen = (np.hstack(x).reshape(1, -1) for x in product(*XX))
print('Done.')

# pred = np.fromiter((tree.predict(x) for x in x_gen), dtype=float, like=Y)
# print(pred)

##############################
print(sklearn.tree.export_text(tree))

if CONFIG['plot']:
    sklearn.tree.plot_tree(tree)
    plt.show()

if CONFIG['inspect']:
    breakpoint()
