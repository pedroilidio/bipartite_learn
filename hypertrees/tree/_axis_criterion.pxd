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
    """Criterion that is able to select a subset of columns to consider.

    It can also calculate the impurity on the other axis, for the current node
    and children.
    """
    cdef const DOUBLE_t* col_sample_weight

    cdef SIZE_t* col_samples
    cdef SIZE_t col_start
    cdef SIZE_t col_end

    # Since samples are reordered by splitters in node_split, indices of
    # selected columns must be saved during AxisCriterion.init_columns().
    # Otherwise, Criterion.sum_total will be built in an order that may not
    # correspond to col_samples anymore when the time of calling
    # Criterion.init() comes.

    # In principle, only row_samples needs to be copied, since
    # Splitter2D cals splitter_rows.node_reset() before
    # splitter_cols.node_reset(). However, we choose to copy both
    # to be independent of this specific order.

    # FIXME: If splitters become able of finding the best split without
    # reordering, this strategy can be dropped and we avoid the memory
    # burden.
    cdef SIZE_t[::1] _col_indices

    # cdef SIZE_t n_cols  # n_outputs represents the number of columns
    # cdef double weighted_n_cols  # TODO: not needed for now

    cdef SIZE_t n_node_cols
    cdef double weighted_n_cols
    cdef double weighted_n_node_cols

    cdef bint _columns_are_set

    # FIXME: Since we cannot access rows_impurity and cols_impurity separately,
    # we have to discard self.impurity_improvement()'s input and calculate
    # them again. To mitigate this problem, self.axes_children_impurities()
    # caches the previous values it calculated and reuses them if
    # self.pos == self._cached_pos.
    # NOTE: self._cached_pos must be reset at AxisCriterion.init() to
    # ensure it always corresponds to the current tree node.
    cdef SIZE_t _cached_pos
    cdef double _cached_rows_node_impurity
    cdef double _cached_cols_node_impurity
    cdef double _cached_rows_impurity_left
    cdef double _cached_rows_impurity_right
    cdef double _cached_cols_impurity_left
    cdef double _cached_cols_impurity_right

    cdef void init_columns(
        self,
        const DOUBLE_t* col_sample_weight,
        double weighted_n_cols,
        SIZE_t* col_samples,
        SIZE_t col_start,
        SIZE_t col_end,
    ) nogil

    cdef double node_axes_impurities(
        self,
        double* rows_impurity,
        double* cols_impurity,
    ) nogil

    cdef void children_axes_impurities(
        self,
        double* rows_impurity_left,
        double* rows_impurity_right,
        double* cols_impurity_left,
        double* cols_impurity_right,
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
    cdef double sum_sq_row_sums  # np.sum(sample_weights * np.sum(y, axis=1) ** 2)

    cdef double[::1] sum_total  # The sum of w*y.
    cdef double[::1] sum_left   # Same as above, but for the left side of the split
    cdef double[::1] sum_right  # Same as above, but for the right side of the split
