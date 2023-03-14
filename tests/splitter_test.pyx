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


cpdef apply_criterion(
    Criterion criterion,
    cnp.ndarray y,
    DOUBLE_t[:] sample_weight=None,
    SIZE_t start=0,
    SIZE_t end=0,
):
    cdef double weighted_n_samples
    cdef double impurity_left, impurity_right
    cdef SIZE_t pos, i, n_splits
    cdef SIZE_t[::1] sample_indices = np.arange(y.shape[0])
    cdef DOUBLE_t[::1] sample_weight_

    # Note, however, that the criterion is able to get to end=y.shape[0], such
    # that n_left=n_samples and n_right=0.
    end = end if end > 0 else y.shape[0] + end
    start = start if start >= 0 else y.shape[0] + start

    if sample_weight is None:
        weighted_n_samples = y.shape[0]
    else:
        sample_weight_ = sample_weight.astype('float64', order='C')
        weighted_n_samples = sample_weight.sum()
    
    criterion.init(
        y,
        sample_weight,
        weighted_n_samples,
        sample_indices,
        start,
        end,
    )
    criterion.reset()
    n_splits = end - start - 1
    result = {
        'impurity_parent': criterion.node_impurity(),
        'weighted_n_samples': criterion.weighted_n_samples,
        'pos': np.zeros(n_splits, dtype='uint32'),
        'impurity_right': np.zeros(n_splits, dtype='double'),
        'impurity_left': np.zeros(n_splits, dtype='double'),
        'weighted_n_left': np.zeros(n_splits, dtype='double'),
        'weighted_n_right': np.zeros(n_splits, dtype='double'),
        'proxy_improvement': np.zeros(n_splits, dtype='double'),
        'improvement': np.zeros(n_splits, dtype='double'),
        # sum_left and sum_right second dimension differs between classification
        # or regression criteria, that is why dtype=object.
        'sum_left': np.empty(n_splits, dtype=object),
        'sum_right': np.empty(n_splits, dtype=object),
    }

    for pos in range(start+1, end):  # Start from 1 element on left
        i = pos - start - 1
        criterion.update(pos)
        result['pos'][i] = criterion.pos
        criterion.children_impurity(&impurity_left, &impurity_right)
        result['impurity_right'][i] = impurity_right
        result['impurity_left'][i] = impurity_left
        result['weighted_n_left'][i] = criterion.weighted_n_left
        result['weighted_n_right'][i] = criterion.weighted_n_right
        result['proxy_improvement'][i] = criterion.proxy_impurity_improvement()
        result['improvement'][i] = criterion.impurity_improvement(
            result['impurity_parent'],
            result['impurity_left'][i],
            result['impurity_right'][i],
        )
        if hasattr(criterion, 'sum_left') and hasattr(criterion, 'sum_right'):
            result['sum_left'][i] = criterion.sum_left
            result['sum_right'][i] = criterion.sum_right
    
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
        SIZE_t ax, pos, i, n_splits
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
        n_splits = end_[ax] - start_[ax] - 1
        sup_criterion.reset()
        unsup_criterion.reset()
        axis_result = {
            'axis': ax,
            'pos': np.zeros(n_splits, dtype='uint32'),
            'weighted_n_samples': bipartite_ss_criterion.weighted_n_samples,
            'weighted_n_left': np.zeros(n_splits, dtype='double'),
            'weighted_n_right': np.zeros(n_splits, dtype='double'),
            'impurity_parent': bipartite_ss_criterion.node_impurity(),
            'supervised_impurity_parent': sup_criterion.node_impurity(),
            'unsupervised_impurity_parent': unsup_criterion.node_impurity(),
            'impurity_left': np.zeros(n_splits, dtype='double'),
            'impurity_right': np.zeros(n_splits, dtype='double'),
            'supervised_impurity_left': np.zeros(n_splits, dtype='double'),
            'supervised_impurity_right': np.zeros(n_splits, dtype='double'),
            'unsupervised_impurity_left': np.zeros(n_splits, dtype='double'),
            'unsupervised_impurity_right': np.zeros(n_splits, dtype='double'),
            'improvement': np.zeros(n_splits, dtype='double'),
            'supervised_improvement': np.zeros(n_splits, dtype='double'),
            'unsupervised_improvement': np.zeros(n_splits, dtype='double'),
            'proxy_improvement': np.zeros(n_splits, dtype='double'),
            'supervised_proxy_improvement': np.zeros(n_splits, dtype='double'),
            'unsupervised_proxy_improvement': np.zeros(n_splits, dtype='double'),
            # sum_left and sum_right second dimension differs between classification
            # or regression criteria, that is why dtype=object.
            # 'sum_left': np.empty(n_splits, dtype=object),
            # 'sum_right': np.empty(n_splits, dtype=object),
        }

        for pos in range(start_[ax]+1, end_[ax]):  # Start from 1 element on left
            i = pos - start_[ax] - 1
            sup_criterion.reset()
            unsup_criterion.reset()
            sup_criterion.update(pos)
            unsup_criterion.update(pos)
            axis_result['pos'][i] = sup_criterion.pos

            bipartite_ss_criterion.children_impurity(&impurity_left, &impurity_right, axis=ax)
            axis_result['impurity_left'][i] = impurity_left
            axis_result['impurity_right'][i] = impurity_right

            sup_criterion.children_impurity(&sup_impurity_left, &sup_impurity_right)
            unsup_criterion.children_impurity(&unsup_impurity_left, &unsup_impurity_right)
            axis_result['supervised_impurity_left'][i] = sup_impurity_left
            axis_result['supervised_impurity_right'][i] = sup_impurity_right
            axis_result['unsupervised_impurity_left'][i] = unsup_impurity_left
            axis_result['unsupervised_impurity_right'][i] = unsup_impurity_right

            axis_result['weighted_n_left'][i] = sup_criterion.weighted_n_left
            axis_result['weighted_n_right'][i] = sup_criterion.weighted_n_right

            if ss_criterion is None:
                axis_result['proxy_improvement'][i] = sup_criterion.proxy_impurity_improvement()
            else:
                axis_result['proxy_improvement'][i] = ss_criterion.proxy_impurity_improvement()

            axis_result['supervised_proxy_improvement'][i] = sup_criterion.proxy_impurity_improvement()
            axis_result['unsupervised_proxy_improvement'][i] = unsup_criterion.proxy_impurity_improvement()

            axis_result['improvement'][i] = bipartite_ss_criterion.impurity_improvement(
                axis_result['impurity_parent'],
                axis_result['impurity_left'][i],
                axis_result['impurity_right'][i],
                axis=ax,
            )
            axis_result['supervised_improvement'][i] = sup_criterion.impurity_improvement(
                axis_result['supervised_impurity_parent'],
                axis_result['supervised_impurity_left'][i],
                axis_result['supervised_impurity_right'][i],
            )
            axis_result['unsupervised_improvement'][i] = unsup_criterion.impurity_improvement(
                axis_result['unsupervised_impurity_parent'],
                axis_result['unsupervised_impurity_left'][i],
                axis_result['unsupervised_impurity_right'][i],
            )
            # if hasattr(criterion, 'sum_left') and hasattr(criterion, 'sum_right'):
            #     axis_result['sum_left'][i] = criterion.sum_left
            #     axis_result['sum_right'][i] = criterion.sum_right
        
        result.append(axis_result)

    return result


cpdef get_criterion_status(
    Criterion criterion,
):
    cdef double imp_left, imp_right, node_imp
    result = {
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
    }
    node_imp = criterion.node_impurity()
    criterion.children_impurity(&imp_left, &imp_right)
    result['node_impurity'] = node_imp
    result['impurity_left'] = imp_left
    result['impurity_right'] = imp_right
    result['proxy_impurity_improvement'] = criterion.proxy_impurity_improvement()
    result['impurity_improvement'] = criterion.impurity_improvement(
        node_imp, imp_left, imp_right,
    )
    return result


cpdef update_criterion(Criterion criterion, SIZE_t pos):
    criterion.reset()
    criterion.update(pos)
