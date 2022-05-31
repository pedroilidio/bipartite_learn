from sklearn.tree._criterion cimport Criterion
from sklearn.tree._splitter cimport Splitter, SplitRecord
from sklearn.tree._tree cimport SIZE_t

from hypertrees.tree._nd_splitter cimport SplitRecord as SplitRecordND, Splitter2D
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
        verbose=False,
):
    cdef SIZE_t start = 0
    cdef SIZE_t end = y.shape[0]

    cdef SplitRecord split
    _init_split(&split, 0)

    cdef SIZE_t ncf = 0  # n_constant_features
    cdef double wnns  # weighted_n_node_samples

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
        print('[SPLITTER_TEST] splitter.criterion.pos:', splitter.criterion.pos)
        print('[SPLITTER_TEST] calling splitter.node_split(impurity, &split, &ncf)')

    splitter.node_split(impurity, &split, &ncf)

    return split


cpdef test_splitter_nd(
        Splitter2D splitter,
        object X, np.ndarray y,
        SIZE_t ndim=2,
        verbose=False,
):
    cdef SIZE_t* end = y.shape
    cdef SIZE_t* start = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))
    cdef SIZE_t* ncf = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))

    cdef SIZE_t i
    for i in range(ndim):
        start[i] = 0
        ncf[i] = 0

    cdef SplitRecordND split
    _init_split(&split, 0)
    cdef double wnns  # weighted_n_node_samples

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
        print('[SPLITTER_TEST] calling splitter.node_split(impurity, &split, &ncf)')

    splitter.node_split(impurity, &split, ncf)

    free(start)
    return split
