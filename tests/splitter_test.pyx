from collections import defaultdict
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._splitter cimport Splitter, SplitRecord
from sklearn.tree._tree cimport SIZE_t, DOUBLE_t, DTYPE_t

from hypertrees.tree._nd_criterion cimport BipartiteCriterion
from hypertrees.tree._semisupervised_criterion cimport (
    SSCompositeCriterion, BipartiteSemisupervisedCriterion,
)
from hypertrees.tree._nd_splitter cimport (
    MultipartiteSplitRecord,
    BipartiteSplitter
)
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as cnp

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
    object X, cnp.ndarray y,
    SIZE_t start=0,
    SIZE_t end=0,
    verbose=False,
):
    cdef SplitRecord split
    cdef SIZE_t ncf = 0  # n_constant_features
    cdef double wnns  # weighted_n_node_samples

    end = end if end > 0 else y.shape[0] + end
    start = start if start >= 0 else y.shape[0] + start

    _init_split(<SplitRecord*>&split, 0)

    if verbose:
        print('[SPLITTER_TEST] calling splitter.init(X, y, None)')
    splitter.init(X, y, None)
    if verbose:
        print(f'[SPLITTER_TEST] calling splitter.node_reset(start={start}, end={end}, &wnns)')
    splitter.node_reset(start, end, &wnns)

    impurity = splitter.node_impurity()
    # impurity = splitter.criterion.node_impurity()  # Same. Above wraps this.

    if verbose:
        print('[SPLITTER_TEST] splitter.node_impurity():', impurity)
        print('[SPLITTER_TEST] y.var():', y.var())
        print('[SPLITTER_TEST] y.var(axis=0).mean():', y.var(axis=0).mean())
        print('[SPLITTER_TEST] splitter.criterion.pos:', splitter.criterion.pos)
        print('[SPLITTER_TEST] calling splitter.node_split(impurity, &split, &ncf)')

    splitter.node_split(impurity,<SplitRecord*>&split, &ncf)

    if verbose:
        print('[SPLITTER_TEST] splitter.criterion.pos:', splitter.criterion.pos)
        print('[SPLITTER_TEST] weighted_n_samples', splitter.criterion.weighted_n_samples)
        print('[SPLITTER_TEST] weighted_n_node_samples', splitter.criterion.weighted_n_node_samples)
        print('[SPLITTER_TEST] weighted_n_right', splitter.criterion.weighted_n_right)
        print('[SPLITTER_TEST] weighted_n_left', splitter.criterion.weighted_n_left)

    return dict(
        impurity_parent=impurity,
        impurity_left=split.impurity_left,
        impurity_right=split.impurity_right,
        pos=split.pos,
        feature=split.feature,
        threshold=split.threshold,
        improvement=split.improvement,
    )


cpdef test_splitter_nd(
    BipartiteSplitter splitter,
    X, y,
    start=[0, 0],
    end=[0, 0],
    ndim=None,
    verbose=False,
):
    if verbose:
        print('[SPLITTER_TEST] starting splitter_nd test')
    ndim = ndim or y.ndim
    cdef SIZE_t* end_ = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))
    cdef SIZE_t* start_ = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))
    cdef SIZE_t* ncf = <SIZE_t*> malloc(ndim * sizeof(SIZE_t))
    cdef MultipartiteSplitRecord split
    cdef double wnns  # weighted_n_node_samples
    cdef SIZE_t i

    for i in range(ndim):
        start_[i] = start[i] if start[i] >= 0 else y.shape[i] + start[i]
        end_[i] = end[i] if end[i] > 0 else y.shape[i] + end[i]
        ncf[i] = 0

    _init_split(<SplitRecord*>&split, 0)

    if verbose:
        print('[SPLITTER_TEST] calling splitter.init(X, y, None)')
    splitter.init(X, y, None)

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

    return dict(
        impurity_parent=impurity,
        impurity_left=split.split_record.impurity_left,
        impurity_right=split.split_record.impurity_right,
        pos=split.split_record.pos,
        feature=split.split_record.feature,
        axis=split.axis,
        threshold=split.split_record.threshold,
        improvement=split.split_record.improvement,
    )


cpdef update_criterion(Criterion criterion, SIZE_t pos):
    criterion.reset()
    criterion.update(pos)


cpdef get_criterion_status(
    Criterion criterion,
):
    cdef double imp_left, imp_right, node_imp
    node_imp = criterion.node_impurity()
    criterion.children_impurity(&imp_left, &imp_right)

    return {
        # 'y': np.asarray(criterion.y),
        'start': criterion.start,
        'end': criterion.end,
        'pos': criterion.pos,
        'n_outputs': criterion.n_outputs,
        'n_samples': criterion.n_samples,
        'n_node_samples': criterion.n_node_samples,
        'weighted_n_samples': criterion.weighted_n_samples,
        'weighted_n_node_samples': criterion.weighted_n_node_samples,
        'weighted_n_left': criterion.weighted_n_left,
        'weighted_n_right': criterion.weighted_n_right,
        'impurity_parent': node_imp,
        'impurity_left': imp_left,
        'impurity_right': imp_right,
        'proxy_improvement': criterion.proxy_impurity_improvement(),
        'improvement': criterion.impurity_improvement(
            node_imp, imp_left, imp_right,
        ),
    }


cpdef get_bipartite_criterion_status(
    BipartiteCriterion bipartite_criterion,
    SIZE_t axis,
    Criterion axis_criterion=None,
):
    cdef:
        double imp_left, imp_right, node_imp

    if axis_criterion is None:
        if axis == 0:
            axis_criterion = bipartite_criterion.criterion_rows
        elif axis == 1:
            axis_criterion = bipartite_criterion.criterion_cols
        elif axis == -1:
            raise ValueError("One must provide either axis or axis_criterion")
        else:
            raise ValueError("axis must be 0 or 1")

    node_imp = bipartite_criterion.node_impurity()
    bipartite_criterion.children_impurity(&imp_left, &imp_right, axis)

    return {
        # 'y': np.asarray(criterion.y),
        'start': axis_criterion.start,
        'end': axis_criterion.end,
        'pos': axis_criterion.pos,
        'n_outputs': axis_criterion.n_outputs,

        'weighted_n_samples': bipartite_criterion.weighted_n_samples,
        'weighted_n_node_samples': bipartite_criterion.weighted_n_node_samples,

        'axis_n_samples': axis_criterion.n_samples,
        'axis_n_node_samples': axis_criterion.n_node_samples,
        'axis_weighted_n_samples': axis_criterion.weighted_n_samples,
        'axis_weighted_n_node_samples': axis_criterion.weighted_n_node_samples,
        'weighted_n_left': axis_criterion.weighted_n_left,
        'weighted_n_right': axis_criterion.weighted_n_right,

        'impurity_parent': node_imp,
        'impurity_left': imp_left,
        'impurity_right': imp_right,
        'proxy_improvement': axis_criterion.proxy_impurity_improvement(),
        'improvement': bipartite_criterion.impurity_improvement(
            node_imp, imp_left, imp_right, axis,
        ),
    }


cpdef apply_criterion(
    Criterion criterion,
    DOUBLE_t[:, ::1] y,
    DOUBLE_t[:] sample_weight=None,
    SIZE_t start=0,
    SIZE_t end=0,
):
    cdef double weighted_n_samples
    cdef double impurity_left, impurity_right
    cdef SIZE_t pos, i
    cdef SIZE_t[::1] sample_indices = np.arange(y.shape[0])

    # Note, however, that the criterion is able to get to end=y.shape[0], such
    # that n_left=n_samples and n_right=0.
    end = end if end > 0 else y.shape[0] + end
    start = start if start >= 0 else y.shape[0] + start

    if sample_weight is None:
        weighted_n_samples = y.shape[0]
    else:
        weighted_n_samples = sample_weight.sum()
    
    criterion.init(
        y=y,
        sample_weight=sample_weight,
        weighted_n_samples=weighted_n_samples,
        sample_indices=sample_indices,
        start=start,
        end=end,
    )

    result = defaultdict(list)

    # Consider min_samples_leaf=1, to avoid ZeroDivision errors.
    for pos in range(start + 1, end):
        criterion.update(pos)
        for k, v in get_criterion_status(criterion).items():
            result[k].append(v)
    
    for k, v in result.items():
        if isinstance(v, list):
            result[k] = np.array(v)
    
    return dict(result)


cpdef apply_bipartite_criterion(
    BipartiteCriterion bipartite_criterion,
    cnp.ndarray X_rows,
    cnp.ndarray X_cols,
    cnp.ndarray y,
    cnp.ndarray sample_weight=None,
    start=[0, 0],
    end=[0, 0],
):
    cdef:
        SIZE_t ax, pos
        double impurity_left, impurity_right
        DOUBLE_t[:] row_weights = None
        DOUBLE_t[:] col_weights = None
        DTYPE_t[:, ::1] X_rows_
        DTYPE_t[:, ::1] X_cols_
        DOUBLE_t[:, ::1] y_
        DOUBLE_t[:, ::1] y_transposed
        Criterion criterion
        Criterion criterion_rows
        Criterion criterion_cols

        double weighted_n_rows
        double weighted_n_cols
        SIZE_t[::1] row_indices = np.arange(y.shape[0])
        SIZE_t[::1] col_indices = np.arange(y.shape[1])
        SIZE_t n_rows = y.shape[0]
        SIZE_t[2] start_
        SIZE_t[2] end_

    for ax in range(2):
        start_[ax] = start[ax] if start[ax] >= 0 else y.shape[ax] + start[ax]
        end_[ax] = end[ax] if end[ax] > 0 else y.shape[ax] + end[ax]

    X_rows_ = X_rows.astype('float32', order='C')
    X_cols_ = X_cols.astype('float32', order='C')
    y_ = y.astype('float64')
    
    n_splits_rows = end_[0] - start_[0]
    n_splits_cols = end_[1] - start_[1]

    if sample_weight is None:
        weighted_n_rows = y.shape[0]
        weighted_n_cols = y.shape[1]
    else:
        row_weights = sample_weight[:n_rows].astype('float64', order='C')
        col_weights = sample_weight[n_rows:].astype('float64', order='C')
        weighted_n_rows = row_weights.sum()
        weighted_n_cols = col_weights.sum()
    
    bipartite_criterion.init(
        X_rows=X_rows_,
        X_cols=X_cols_,
        y=y_,
        row_weights=row_weights,
        col_weights=col_weights,
        weighted_n_rows=weighted_n_rows,
        weighted_n_cols=weighted_n_cols,
        row_indices=row_indices,
        col_indices=col_indices,
        start=start_,
        end=end_,
    )

    criterion_rows = bipartite_criterion.criterion_rows
    criterion_cols = bipartite_criterion.criterion_cols

    result = []

    for ax, criterion in ((0, criterion_rows), (1, criterion_cols)):
        axis_result = defaultdict(list)
        axis_result['axis'] = ax

        # Consider min_samples_leaf=1, to avoid ZeroDivision errors.
        for pos in range(start_[ax] + 1, end_[ax]):
            criterion.update(pos)
            for k, v in get_bipartite_criterion_status(bipartite_criterion, ax).items():
                axis_result[k].append(v)
        
        for k, v in axis_result.items():
            if isinstance(v, list):
                axis_result[k] = np.array(v)
        
        result.append(dict(axis_result))

    return result


cpdef apply_bipartite_ss_criterion(
    BipartiteSemisupervisedCriterion bipartite_ss_criterion,
    cnp.ndarray X_rows,
    cnp.ndarray X_cols,
    cnp.ndarray y,
    cnp.ndarray sample_weight=None,
    start=[0, 0],
    end=[0, 0],
):
    cdef:
        SIZE_t ax, pos
        double impurity_left, impurity_right
        double sup_impurity_left, sup_impurity_right
        double unsup_impurity_left, unsup_impurity_right
        DOUBLE_t[:] row_weights = None
        DOUBLE_t[:] col_weights = None
        DTYPE_t[:, ::1] X_rows_
        DTYPE_t[:, ::1] X_cols_
        DOUBLE_t[:, ::1] y_
        DOUBLE_t[:, ::1] y_transposed
        Criterion sup_criterion, unsup_criterion, ss_criterion
        Criterion sup_criterion_rows, unsup_criterion_rows
        Criterion sup_criterion_cols, unsup_criterion_cols
        SSCompositeCriterion ss_criterion_rows, ss_criterion_cols

        double weighted_n_rows
        double weighted_n_cols
        SIZE_t[::1] row_indices = np.arange(y.shape[0])
        SIZE_t[::1] col_indices = np.arange(y.shape[1])
        SIZE_t n_rows = y.shape[0]
        SIZE_t[2] start_
        SIZE_t[2] end_

    for ax in range(2):
        start_[ax] = start[ax] if start[ax] >= 0 else y.shape[ax] + start[ax]
        end_[ax] = end[ax] if end[ax] > 0 else y.shape[ax] + end[ax]

    X_rows_ = X_rows.astype('float32', order='C')
    X_cols_ = X_cols.astype('float32', order='C')
    y_ = y.astype('float64')
    
    if sample_weight is None:
        weighted_n_rows = y.shape[0]
        weighted_n_cols = y.shape[1]
    else:
        row_weights = sample_weight[:n_rows].astype('float64', order='C')
        col_weights = sample_weight[n_rows:].astype('float64', order='C')
        weighted_n_rows = row_weights.sum()
        weighted_n_cols = col_weights.sum()
    
    bipartite_ss_criterion.set_X(
        X_rows.astype('float64'),
        X_cols.astype('float64'),
    )
    bipartite_ss_criterion.init(
        X_rows=X_rows_,
        X_cols=X_cols_,
        y=y_,
        row_weights=row_weights,
        col_weights=col_weights,
        weighted_n_rows=weighted_n_rows,
        weighted_n_cols=weighted_n_cols,
        row_indices=row_indices,
        col_indices=col_indices,
        start=start_,
        end=end_,
    )

    unsup_criterion_rows = bipartite_ss_criterion.unsupervised_criterion_rows
    unsup_criterion_cols = bipartite_ss_criterion.unsupervised_criterion_cols
    sup_criterion_rows = bipartite_ss_criterion.supervised_criterion_rows
    sup_criterion_cols = bipartite_ss_criterion.supervised_criterion_cols
    ss_criterion_rows = bipartite_ss_criterion.ss_criterion_rows
    ss_criterion_cols = bipartite_ss_criterion.ss_criterion_cols

    result = []

    for ax, sup_criterion, unsup_criterion, ss_criterion in (
        (0, sup_criterion_rows, unsup_criterion_rows, ss_criterion_rows),
        (1, sup_criterion_cols, unsup_criterion_cols, ss_criterion_cols),
    ):
        axis_result = defaultdict(list)
        axis_result['axis'] = ax

        # Consider min_samples_leaf=1, to avoid ZeroDivision errors.
        for pos in range(start_[ax] + 1, end_[ax]):
            ss_criterion.update(pos)

            for k, v in get_criterion_status(unsup_criterion).items():
                axis_result['unsupervised_' + k].append(v)

            for k, v in get_bipartite_criterion_status(
                bipartite_ss_criterion.supervised_bipartite_criterion,
                axis=ax,
            ).items():
                axis_result['supervised_' + k].append(v)
        
            for k, v in get_bipartite_criterion_status(
                bipartite_ss_criterion,
                axis=ax,
                axis_criterion=ss_criterion,
            ).items():
                axis_result[k].append(v)

        for k, v in axis_result.items():
            if isinstance(v, list):
                axis_result[k] = np.array(v)
        
        result.append(dict(axis_result))

    return result