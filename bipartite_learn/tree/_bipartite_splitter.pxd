from sklearn.tree._splitter cimport SplitRecord, Splitter

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters

from ._bipartite_criterion cimport BipartiteCriterion


# Data to track sample split
cdef struct MultipartiteSplitRecord:
    SplitRecord split_record  # Monopartite object to use struct composition.
    SIZE_t axis               # Axis in which the split occurred.


cdef class BipartiteSplitter:
    # Wrapper class to coordinate 2 Splitters at the same time.

    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end].
    #
    # The impurity computations are delegated to a criterion object.
    cdef public Splitter splitter_rows
    cdef public Splitter splitter_cols
    cdef public BipartiteCriterion bipartite_criterion
    cdef public SIZE_t n_row_features  # Number of row features (X[0].shape[0]).
    # TODO: X dtype must be DOUBLE_t, not DTYPE_t (float32) to use
    # semisupervision.
    cdef const DTYPE_t[:, ::1] X_rows
    cdef const DTYPE_t[:, ::1] X_cols
    cdef const DOUBLE_t[:, :] y

    cdef SIZE_t n_rows
    cdef SIZE_t n_cols
    cdef SIZE_t n_samples
    cdef double weighted_n_rows
    cdef double weighted_n_cols
    cdef double weighted_n_samples

    cdef public SIZE_t min_samples_leaf
    cdef public SIZE_t min_rows_leaf
    cdef public SIZE_t min_cols_leaf
    cdef public double min_weight_leaf
    # TODO
    # cdef public double min_row_weight_leaf
    # cdef public double min_col_weight_leaf

    cdef const DOUBLE_t[:] row_weights
    cdef const DOUBLE_t[:] col_weights
    cdef SIZE_t[::1] row_indices  # TODO: Drop contiguity?
    cdef SIZE_t[::1] col_indices
    cdef SIZE_t[2] start
    cdef SIZE_t[2] end

    # Methods
    cdef int init(
        self,
        object X,
        const DOUBLE_t[:, :] y,
        const DOUBLE_t[:] sample_weight,
    ) except -1

    cdef int node_reset(
        self,
        SIZE_t[2] start,
        SIZE_t[2] end,
        double* weighted_n_node_samples
    ) nogil except -1

    cdef int node_split(
        self,
        double impurity,
        MultipartiteSplitRecord* split,
        SIZE_t[2] n_constant_features,
    ) nogil except -1

    cdef void node_value(self, double* dest) nogil

    cdef double node_impurity(self) nogil