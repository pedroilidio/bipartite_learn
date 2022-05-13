import numpy as np
cimport numpy as np

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters

from sklearn.tree._splitter cimport Splitter
from sklearn.tree._criterion cimport RegressionCriterion


cdef class RegressionCriterionWrapper2D:
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

    cdef Splitter splitter_rows
    cdef Splitter splitter_cols

    cdef int init(
            self, const DOUBLE_t[:, ::1] y_2D,
            DOUBLE_t* row_sample_weight,
            DOUBLE_t* col_sample_weight,
            double weighted_n_samples,
            SIZE_t* row_samples, SIZE_t* col_samples,
            SIZE_t[2] start, SIZE_t[2] end,
            SIZE_t[2] y_shape,
        ) nogil except -1

    cdef int _node_reset_child_splitter(
            self,
            Splitter child_splitter,
            const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            SIZE_t start,
            SIZE_t end,
            DOUBLE_t* weighted_n_node_samples,
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
            self, double impurity_parent, double
            impurity_left, double impurity_right,
            SIZE_t axis,
    ) nogil


cdef class MSE_Wrapper2D(RegressionCriterionWrapper2D):
    pass
