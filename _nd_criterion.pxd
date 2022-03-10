import numpy as np
cimport numpy as np

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters

from sklearn.tree._criterion cimport RegressionCriterion


cdef class RegressionCriterionWrapper2D:
    cdef public list child_criteria  # TODO: const?
    cdef const DOUBLE_t[:, ::1] y_2D
    cdef DOUBLE_t[:, ::1] y_row_sums
    cdef DOUBLE_t[:, ::1] y_col_sums

    cdef DOUBLE_t* row_sample_weight
    cdef DOUBLE_t* col_sample_weight
    cdef SIZE_t* row_samples
    cdef SIZE_t* col_samples
    cdef SIZE_t[2] start
    cdef SIZE_t[2] end

    cdef SIZE_t n_outputs
    cdef DOUBLE_t sq_sum_total
    cdef DOUBLE_t* sum_total
    cdef DOUBLE_t weighted_n_node_samples
    cdef DOUBLE_t weighted_n_samples
    cdef DOUBLE_t* total_row_sample_weight
    cdef DOUBLE_t* total_col_sample_weight

    cdef DOUBLE_t weighted_n_row_samples
    cdef DOUBLE_t weighted_n_col_samples

    cdef int init(
            self, const DOUBLE_t[:, ::1] y_2D,
            DOUBLE_t* row_sample_weight,
            DOUBLE_t* col_sample_weight,
            double weighted_n_samples,
            SIZE_t* row_samples, SIZE_t* col_samples,
            SIZE_t[2] start, SIZE_t[2] end,
            SIZE_t[2] y_shape,
        ) except -1 # nogil

    cdef int _init_child_criterion(
            self,
            MSE2D child_criterion,
            const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            SIZE_t* samples, SIZE_t start,
            SIZE_t end,
            DOUBLE_t weighted_n_cols,
    ) except -1  # nogil

    cdef void node_value(self, double* dest)
    cdef double node_impurity(self)
    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    )# nogil
    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right,
                                     SIZE_t axis)

cdef class MSE_Wrapper2D(RegressionCriterionWrapper2D):
    pass

# cdef class RegressionCriterion2D(RegressionCriterion):
# 
cdef class MSE2D(RegressionCriterion):
    cdef readonly DOUBLE_t weighted_n_cols
