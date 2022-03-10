import copy
from pprint import pprint
from sklearn.tree._criterion import MSE
from sklearn.tree._splitter import BestSplitter
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

from splitter_test import test_splitter, test_splitter2d
from _nd_splitter import Splitter2D
from _nd_criterion import MSE_Wrapper2D
from _nd_classes import DecisionTreeRegressor2D


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
SEED = 1
SHAPE = (500, 600)
NATTRS = (10, 7)
NRULES = 10
TRANSPOSE_TEST = 0
NOISE = 0
INSPECT = 0
#######################


## Generate mock data
np.random.seed(SEED)
t0 = time()
XX, Y = gen_imatrix(SHAPE, NATTRS, nrules=NRULES)

if TRANSPOSE_TEST:
    print('Test transposing axis.')
    Y = np.copy(Y.T.astype(DOUBLE_t), order='C')
    XX = [np.ascontiguousarray(X, DTYPE_t) for X in XX[::-1]]
    NATTRS = NATTRS[::-1]
    SHAPE = SHAPE[::-1]
else:
    Y = np.ascontiguousarray(Y, DOUBLE_t)
    XX = [np.ascontiguousarray(X, DTYPE_t) for X in XX]

if NOISE:
    Y += np.random.rand(*SHAPE)  # add some noise

print('Data generation time:', time()-t0)
## Instantiate sklearn objects

# ## 2D Criteria:
# criterion_rows = MSE_2D(n_outputs=2, n_samples=SHAPE[0])
# criterion_cols = MSE_2D(n_outputs=2, n_samples=SHAPE[1])
# # n_outputs=2 because ...
# 
# splitter_rows = BestSplitter(
#     criterion=criterion_rows,
#     max_features=NATTRS[0],
#     min_samples_leaf=1,
#     min_weight_leaf=0,
#     random_state=np.random.RandomState(SEED),
# )
# splitter_cols = BestSplitter(
#     criterion=criterion_cols,
#     max_features=NATTRS[1],
#     min_samples_leaf=1,
#     min_weight_leaf=0,
#     random_state=np.random.RandomState(SEED),
# )
# 
# 
# splitter = Splitter2D(splitter_rows, splitter_cols)

tree = DecisionTreeRegressor2D()
print(vars(tree))
print('Fitting tree...')
tree.fit(XX, Y)
print('Done.')
print(export_text(tree))
plot_tree(tree)
plt.show()
