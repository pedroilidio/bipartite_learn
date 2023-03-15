import numpy as np
cimport numpy as np

from sklearn.tree._tree cimport DTYPE_t         # Type of X
from sklearn.tree._tree cimport DOUBLE_t        # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t          # Type for indices and counters

from sklearn.tree._criterion cimport RegressionCriterion
from ._axis_criterion cimport AxisCriterion


cdef class BipartiteCriterion:
    """Abstract base class."""
    # TODO: X dtype must be DOUBLE_t, not DTYPE_t (float32) to use
    # semisupervision.
    cdef const DTYPE_t[:, ::1] X_rows
    cdef const DTYPE_t[:, ::1] X_cols
    cdef const DOUBLE_t[:, :] y

    cdef const DOUBLE_t[:] row_weights
    cdef const DOUBLE_t[:] col_weights
    cdef const SIZE_t[:] row_indices
    cdef const SIZE_t[:] col_indices
    cdef SIZE_t[2] start
    cdef SIZE_t[2] end

    # NOTE: A source of confusion is that sometimes n_outputs is actually
    #       treated as the number of outputs, but sometimes it is just an
    #       alias for y.shape[1]. In monopartite data, they have the same
    #       value, but for bipartite interaction data one should have this
    #       distinction in mind.
    cdef SIZE_t n_outputs
    cdef SIZE_t n_outputs_rows
    cdef SIZE_t n_outputs_cols

    cdef SIZE_t n_rows
    cdef SIZE_t n_cols
    cdef SIZE_t n_node_rows
    cdef SIZE_t n_node_cols
    cdef double sq_sum_total
    cdef double[::1] sum_total
    cdef double weighted_n_rows
    cdef double weighted_n_cols
    cdef double weighted_n_samples

    cdef double weighted_n_node_samples
    cdef double weighted_n_node_rows
    cdef double weighted_n_node_cols

    cdef int init(
        self,
        const DTYPE_t[:, ::1] X_rows,
        const DTYPE_t[:, ::1] X_cols,
        const DOUBLE_t[:, :] y,
        const DOUBLE_t[:] row_weights,
        const DOUBLE_t[:] col_weights,
        double weighted_n_rows,
        double weighted_n_cols,
        const SIZE_t[:] row_indices,
        const SIZE_t[:] col_indices,
        SIZE_t[2] start,
        SIZE_t[2] end,
    ) nogil except -1

    cdef void node_value(self, double* dest) nogil

    cdef double node_impurity(self) nogil

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ) nogil

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
        SIZE_t axis,
    ) nogil


cdef class RegressionCriterionGSO(BipartiteCriterion):
    cdef public RegressionCriterion criterion_rows
    cdef public RegressionCriterion criterion_cols

    cdef DOUBLE_t[:, ::1] y_row_sums
    cdef DOUBLE_t[:, ::1] y_col_sums

    cdef void* _get_criterion(self, SIZE_t axis) nogil except NULL

    cdef inline int _init_child_criterion(
            self,
            RegressionCriterion criterion,
            const DOUBLE_t[:, ::1] y,
            const DOUBLE_t[:] sample_weight,
            const SIZE_t[:] sample_indices,
            SIZE_t start,
            SIZE_t end,
            SIZE_t n_node_samples,
            double weighted_n_samples,
            double weighted_n_node_samples,
    ) nogil except -1


cdef class GMO(BipartiteCriterion):
    """Applies Predictive Bi-Clustering Trees method.

    See [Pliakos _et al._, 2018](https://doi.org/10.1007/s10994-018-5700-x).
    """
    cdef public AxisCriterion criterion_rows
    cdef public AxisCriterion criterion_cols
    cdef SIZE_t max_n_classes

    cdef void* _get_criterion(self, SIZE_t axis) nogil except NULL


cdef class SquaredErrorGSO(RegressionCriterionGSO):
    pass


cdef class FriedmanGSO(SquaredErrorGSO):
    pass