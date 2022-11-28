# Author: Pedro Ilídio <pedrilidio@gmail.com>
# Modified from scikit-learn.
#
# License: BSD 3 clause

# See _axis_criterion.pyx for implementation details.

from sklearn.tree._tree cimport DTYPE_t         # Type of X
from sklearn.tree._tree cimport DOUBLE_t        # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t          # Type for indices and counters
from sklearn.tree._tree cimport INT32_t         # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t        # Unsigned 32 bit integer
from sklearn.tree._criterion cimport Criterion        # Unsigned 32 bit integer

cdef class AxisCriterion(Criterion):
    cdef const DOUBLE_t* col_sample_weight

    cdef SIZE_t* col_samples
    # Since indices are reordered by splitters in node_split, indices of
    # selected columns must be saved during AxisCriterion.init_columns().
    # Otherwise, Criterion.sum_total will be built in an order that may not
    # correspond to col_samples anymore when the time of calling
    # Criterion.init() comes.
    cdef SIZE_t[::1] _col_indices
    cdef SIZE_t col_start
    cdef SIZE_t col_end

    # cdef SIZE_t n_cols  # n_outputs represents the number of columns
    # cdef double weighted_n_cols  # TODO: not needed for now

    cdef SIZE_t n_node_cols
    cdef double weighted_n_node_cols

    cdef bint _columns_are_set

    cdef void init_columns(
        self,
        SIZE_t* col_samples,
        const DOUBLE_t* col_sample_weight,
        SIZE_t col_start,
        SIZE_t col_end,
    ) nogil

# (TODO)
# cdef class ClassificationCriterion(AxisCriterion):
#     """Abstract criterion for classification."""
# 
#     cdef SIZE_t[::1] n_classes
#     cdef SIZE_t max_n_classes
# 
#     cdef double[:, ::1] sum_total   # The sum of the weighted count of each label.
#     cdef double[:, ::1] sum_left    # Same as above, but for the left side of the split
#     cdef double[:, ::1] sum_right   # Same as above, but for the right side of the split

cdef class AxisRegressionCriterion(AxisCriterion):
    """Abstract regression criterion."""

    cdef double sq_sum_total
    # TODO: sq_row_sums is naturally calculated by the criterion in the other
    #       axis. We could set it as a pointer to the other axis criterion's
    #       sum_total.
    cdef double sq_row_sums  # np.sum(sample_weights * np.sum(y, axis=1) ** 2)

    cdef double[::1] sum_total  # The sum of w*y.
    cdef double[::1] sum_left   # Same as above, but for the left side of the split
    cdef double[::1] sum_right  # Same as above, but for the right side of the split
