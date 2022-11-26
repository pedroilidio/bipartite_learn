# cython: profile=True
import cython
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._splitter cimport Splitter, SplitRecord
from sklearn.tree._tree cimport SIZE_t, DOUBLE_t

from hypertrees.tree._nd_splitter cimport (
    SplitRecord as SplitRecordND,
    Splitter2D
)
from hypertrees.tree._nd_criterion cimport AxisRegressionCriterion
from libc.stdlib cimport malloc, free


import numpy as np
cimport numpy as np

cdef double INFINITY = np.inf


cdef object split_to_dict(SplitRecord split):
    return dict(
        impurity_left=split.impurity_left,
        impurity_right=split.impurity_right,
        pos=split.pos,
        feature=split.feature,
        threshold=split.threshold,
        improvement=split.improvement,
    )


cdef object split_nd_to_dict(SplitRecordND split):
    return dict(
        axis=split.axis,
        impurity_left=split.impurity_left,
        impurity_right=split.impurity_right,
        pos=split.pos,
        feature=split.feature,
        threshold=split.threshold,
        improvement=split.improvement,
    )


cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY


cdef inline void _init_split_nd(SplitRecordND* self, SIZE_t start_pos) nogil:
    self.axis = -1
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY


# @cython.profile(False)  # Profiling yields segfault
cpdef object test_splitter(
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

    return split_to_dict(split)


# @cython.profile(False)  # Profiling yields segfault
cpdef SplitRecordND test_splitter_nd(
        Splitter2D splitter,
        X, y,
        start=None,
        end=None,
        verbose=False,
):
    if verbose:
        print('[SPLITTER_TEST] starting splitter_nd test')
    cdef SIZE_t[2] end_
    cdef SIZE_t[2] start_
    cdef SIZE_t[2] ncf
    cdef SplitRecordND split
    cdef double wnns  # weighted_n_node_samples
    cdef SIZE_t i

    start = start or (0, 0)
    end = end or y.shape

    for i in range(2):
        end_[i] = end[i]
        start_[i] = start[i]
        ncf[i] = 0

    _init_split_nd(&split, 0)

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

    return <SplitRecordND>split


cpdef object test_gmo_splitter_symmetry(
    Splitter2D splitter_nd,
    SIZE_t axis,
    SIZE_t pos,
    X, y,
    start=None,
    end=None,
):
    """Node impurity by both (rows and columns) splitters must be equal."""
    cdef:
        double impurity_left
        double impurity_right
        double other_imp_left
        double other_imp_right
        double manual_impurity_left
        double manual_impurity_right
        double wnns  # Discarded
        AxisRegressionCriterion criterion
        AxisRegressionCriterion other_criterion
        Splitter splitter
        Splitter other_splitter
        SIZE_t i
        SIZE_t start_[2]
        SIZE_t end_[2]
    
    start = start or (0, 0)
    end = end or y.shape
    for i in range(2):
        start_[i] = start[i]
        end_[i] = end[i]

    splitter_nd.init(X, y, NULL)
    splitter_nd.node_reset(start_, end_, &wnns)

    if axis == 0:
        splitter = splitter_nd.splitter_rows
        other_splitter = splitter_nd.splitter_cols
    elif axis == 1:
        splitter = splitter_nd.splitter_cols
        other_splitter = splitter_nd.splitter_rows
    else:
        raise ValueError(f"axis must be 1 or 0 ({axis} received)")
    
    criterion = splitter.criterion
    other_criterion = other_splitter.criterion
    criterion.update(pos)

    print('*** splitter_nd.start[axis], pos, splitter_nd.end[axis]',
           splitter_nd.start[axis], pos, splitter_nd.end[axis])

    other_criterion.set_columns(
        start_[axis],
        pos,
        criterion.samples,
        criterion.sample_weight,
    )
    other_splitter.node_reset(start_[1-axis], end_[1-axis], &wnns)
    other_imp_left = other_splitter.node_impurity()

    print('* LEFT')
    print('*** othercrit. wnns, wnncols',
          other_criterion.weighted_n_node_samples,
          other_criterion.weighted_n_node_cols)
    print('*** othercrit. sq_sum_total, sum_total',
          other_criterion.sq_sum_total, other_criterion.sum_total[0])
    print('*** other axis y sum left', np.sum(other_criterion.y))

    manual_imp_left = 0.5 * (
        np.asarray(other_criterion.y).var(1).mean()
        + np.asarray(other_criterion.y).var(0).mean()
    )

    other_criterion.set_columns(
        pos,
        end_[axis],
        criterion.samples,
        criterion.sample_weight,
    )
    other_splitter.node_reset(start_[1-axis], end_[1-axis], &wnns)
    other_imp_right = other_splitter.node_impurity()

    print('* RIGHT')
    print('*** othercrit. wnns, wnncols',
          other_criterion.weighted_n_node_samples,
          other_criterion.weighted_n_node_cols)
    print('*** othercrit. sq_sum_total, sum_total',
          other_criterion.sq_sum_total,
          other_criterion.sum_total[0])
    print('*** other axis y sum right', np.sum(other_criterion.y))
    manual_imp_right = 0.5 * (
        np.asarray(other_criterion.y).var(1).mean()
        + np.asarray(other_criterion.y).var(0).mean()
    )

    print(f'*** other_imp_left /right {other_imp_left:<20} {other_imp_right:<20}')
    print('*** crit. wnns, wnncols',
          criterion.weighted_n_node_samples,
          criterion.weighted_n_node_cols)

    criterion.children_impurity(&impurity_left, &impurity_right)

    manual_imp_total = 0.5 * (
        np.asarray(criterion.y).var(1).mean()
        + np.asarray(criterion.y).var(0).mean()
    )

    return (
        (impurity_left, other_imp_left, "impurity_left, other_imp_left"),
        (impurity_right, other_imp_right, "impurity_right, other_imp_right"),
        # FIXME
        # (manual_impurity_right, other_imp_right, "manual_impurity_right, other_imp_right"),
        # (manual_impurity_left, other_imp_left, "manual_impurity_left, other_imp_left"),
        # (manual_imp_total, criterion.node_impurity(), "manual_imp_total, criterion.node_impurity()"),
    )
