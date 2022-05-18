from sklearn.tree._splitter cimport Splitter

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters

from ._nd_criterion cimport RegressionCriterionWrapper2D


cdef struct SplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    double threshold       # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split.
    double impurity_right  # Impurity of the right split.

    # ND new:
    SIZE_t axis             # Axis in which the split occurred.


cdef class Splitter2D:
    # Wrapper class to coordinate 2 Splitters at the same time.

    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.
    cdef Splitter splitter_rows
    cdef Splitter splitter_cols
    cdef RegressionCriterionWrapper2D criterion_wrapper
    cdef SIZE_t n_row_features      # Number of row features (X[0].shape[0]).
    cdef const DOUBLE_t[:, ::1] y
    cdef SIZE_t[2] shape

    cdef SIZE_t n_samples
    cdef double weighted_n_samples

    cdef SIZE_t min_samples_leaf
    cdef SIZE_t min_rows_leaf
    cdef SIZE_t min_cols_leaf
    cdef double min_weight_leaf

    cdef DOUBLE_t* row_sample_weight
    cdef DOUBLE_t* col_sample_weight
    cdef SIZE_t[2] start
    cdef SIZE_t[2] end

    # Methods
    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
    ) except -1

    cdef int node_reset(self, SIZE_t[2] start, SIZE_t[2] end,
                        double* weighted_n_node_samples) nogil except -1

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t[2] n_constant_features) nogil except -1

    cdef void node_value(self, double* dest) nogil

    cdef double node_impurity(self) nogil


# cpdef Splitter2D make_2d_splitter(
#        splitter_class,
#        criterion_class,
#        shape,
#        n_attrs,
#        SIZE_t n_outputs=*,
#        min_samples_leaf=*,
#        min_weight_leaf=*,
#        random_state=*,
#        criteria_wrapper_class=*,
#     )
