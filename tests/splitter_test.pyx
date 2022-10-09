from sklearn.tree._criterion cimport Criterion
from sklearn.tree._splitter cimport Splitter, SplitRecord
from sklearn.tree._tree cimport SIZE_t, DOUBLE_t

from hypertrees.tree._nd_splitter cimport (
    SplitRecord as SplitRecordND,
    Splitter2D
)
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np

cdef double INFINITY = np.inf

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY


cpdef test_splitter(
        Splitter splitter,
        object X, np.ndarray y,
        SIZE_t start=0,
        SIZE_t end=-1,
        verbose=False,
):
    cdef SplitRecord split
    cdef SIZE_t ncf = 0  # n_constant_features
    cdef double wnns  # weighted_n_node_samples

    if end == -1:
        end = y.shape[0]

    _init_split(&split, 0)

    if verbose:
        print('[SPLITTER_TEST] calling splitter.init(X, y, NULL)')
    splitter.init(X, y, NULL)
    if verbose:
        print('[SPLITTER_TEST] calling splitter.node_reset(start, end, &wnns)')
    splitter.node_reset(start, end, &wnns)

    impurity = splitter.node_impurity()
    # impurity = splitter.criterion.node_impurity()  # Same. Above wraps this.

    if verbose:
        print('[SPLITTER_TEST] splitter.node_impurity():', impurity)
        print('[SPLITTER_TEST] y.var():', y.var())
        print('[SPLITTER_TEST] y.var(axis=0).mean():', y.var(axis=0).mean())
        print('[SPLITTER_TEST] splitter.criterion.pos:', splitter.criterion.pos)
        print('[SPLITTER_TEST] calling splitter.node_split(impurity, &split, &ncf)')

    splitter.node_split(impurity, &split, &ncf)

    if verbose:
        print('[SPLITTER_TEST] weighted_n_samples', splitter.criterion.weighted_n_samples)
        print('[SPLITTER_TEST] weighted_n_node_samples', splitter.criterion.weighted_n_node_samples)
        print('[SPLITTER_TEST] weighted_n_right', splitter.criterion.weighted_n_right)
        print('[SPLITTER_TEST] weighted_n_left', splitter.criterion.weighted_n_left)

    return split


cpdef test_splitter_nd(
        Splitter2D splitter,
        X, y,
        start=None,
        end=None,
        ndim=None,
        verbose=False,
):
    if verbose:
        print('[SPLITTER_TEST] starting splitter_nd test')
    ndim = ndim or y.ndim
    cdef SIZE_t* end_ = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))
    cdef SIZE_t* start_ = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))
    cdef SIZE_t* ncf = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))
    cdef SplitRecordND split
    cdef double wnns  # weighted_n_node_samples
    cdef SIZE_t i

    start = start or [0] * ndim
    end = end or y.shape

    for i in range(ndim):
        end_[i] = end[i]
        start_[i] = start[i]
        ncf[i] = 0

    _init_split(&split, 0)

    if verbose:
        print('[SPLITTER_TEST] calling splitter.init(X, y, NULL)')
    splitter.init(X, y, NULL)

    if verbose:
        print('[SPLITTER_TEST] calling splitter.node_reset(start, end, &wnns)')
    splitter.node_reset(start_, end_, &wnns)

    impurity = splitter.node_impurity()
    # impurity = splitter.criterion.node_impurity()  # Same. Above wraps this.

    if verbose:
        print('[SPLITTER_TEST] splitter.node_impurity():', impurity)
        print('[SPLITTER_TEST] y.var():', y.var())
        print('[SPLITTER_TEST] y.var(axis=0).mean():', y.var(axis=0).mean())
        print('[SPLITTER_TEST] calling splitter.node_split(impurity, &split, &ncf)')

    splitter.node_split(impurity, &split, ncf)

    if verbose:
        print('[SPLITTER_TEST] (rows) weighted_n_samples',
              splitter.splitter_rows.criterion.weighted_n_samples)
        print('[SPLITTER_TEST] (rows) weighted_n_node_samples',
              splitter.splitter_rows.criterion.weighted_n_node_samples)
        print('[SPLITTER_TEST] (rows) weighted_n_right',
              splitter.splitter_rows.criterion.weighted_n_right)
        print('[SPLITTER_TEST] (rows) weighted_n_left',
              splitter.splitter_rows.criterion.weighted_n_left)

    free(start_)
    free(end_)
    free(ncf)

    return split
