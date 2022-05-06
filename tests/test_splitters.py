from pprint import pprint
from sklearn.tree._criterion import MSE
from sklearn.tree._splitter import BestSplitter

from splitter_test import test_splitter
from hypertree.tree._nd_splitter import make_2d_splitter
from hypertree.tree._nd_criterion import MSE_Wrapper2D
from hypertree.tree._semisupervised import SSBestSplitter
from hypertree.tree._semisupervised import SSCompositeCriterion


import numpy as np
#from sklearn.tree._tree import DTYPE_t, DOUBLE_t
DTYPE_t, DOUBLE_t = np.float32, np.float64

from pathlib import Path
import sys
from time import time

path_pbct_test = Path('../../scripts/predictors/PBCT/tests/')
sys.path.insert(0, str(path_pbct_test))
from make_examples import make_interaction_data

SEED = 0

## Generate mock data
np.random.seed(SEED)
t0 = time()
shape, nattrs = (100, 2000), (5, 1)
XX, Y, strfunc = make_interaction_data(shape, nattrs, nrules=1)
Y = Y.astype(DOUBLE_t)
Y += np.random.rand(*shape)  # add some noise
XX = [X.astype(DTYPE_t) for X in XX]
print('Time:', time()-t0)

## Instantiate sklearn objects
criterion = MSE(n_outputs=shape[1], n_samples=shape[0])
criterion2d = MSE(n_outputs=shape[1], n_samples=shape[0])

splitter = BestSplitter(criterion, nattrs[0], 1, 0,
                        np.random.RandomState(SEED))

splitter2d = make_2d_splitter(
            splitter_class=BestSplitter,
            criterion_class=MSE,
            n_samples=[X.shape[0] for X in XX],
            max_features=Y.shape,
        ),

## Run test
t0 = time()
result = test_splitter(splitter, XX[0], Y, Y.shape)
print('Best split found:')
pprint(result)
print('Time:', time()-t0)

exit()
t0 = time()
result = test_splitter2d(splitter2d, XX[0], Y)
print('Best split found:')
pprint(result)
print('Time:', time()-t0)