# cython: boundscheck=True
import copy
import warnings
from sklearn.tree._splitter cimport Splitter
from sklearn.tree._criterion cimport RegressionCriterion, Criterion
from ._nd_criterion cimport RegressionCriterionWrapper2D, MSE_Wrapper2D

import numpy as np
cimport numpy as np

np.import_array()
cdef double INFINITY = np.inf


cdef inline void _init_split(
        SplitRecord* self, SIZE_t start_pos,
        SIZE_t axis
) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY
    self.axis = axis


cdef class Splitter2D:
    """PBCT splitter implementation.

    Wrapper class to coordinate one Splitter for each axis in a two-dimensional
    problem as described by Pliakos _et al._ (2018).
    """
    def __cinit__(
            self, Splitter splitter_rows, Splitter splitter_cols,
            CriterionWrapper2D criterion_wrapper,
            min_samples_leaf, min_weight_leaf,
    ):
        """Store each axis' splitter."""
        self.n_samples = 0
        self.weighted_n_samples = 0.0
        self.splitter_rows = splitter_rows
        self.splitter_cols = splitter_cols
        self.criterion_wrapper = criterion_wrapper

        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.min_rows_leaf = self.splitter_rows.min_samples_leaf
        self.min_cols_leaf = self.splitter_cols.min_samples_leaf

    def __reduce__(self):
        return (
            type(self),
            (
                self.splitter_rows,
                self.splitter_cols,
                self.criterion_wrapper,
                self.min_samples_leaf,
                self.min_weight_leaf,
            ),
            self.__getstate__(),
        )

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
    ) except -1:
        """Initialize the axes' splitters.
        Take in the input data X, the target Y, and optional sample weights.
        Use them to initialize each axis' splitter.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. It is a list-like of 2d numpy arrays, one
            array for each axis.
        y : ndarray, dtype=DOUBLE_t
            This is the matrix of targets, or true labels, for the samples
        sample_weight : (DOUBLE_t*)[2]
            The weights of the row and column samples, where higher weighted
            samples are fit closer than lower weight samples. If not provided,
            all samples are assumed to have uniform weight.
        """
        # TODO: use memoryview's .transpose
        # TODO: test sample_weight
        cdef const DOUBLE_t[:, ::1] yT = np.ascontiguousarray(y.T)
        # FIXME: only need to set criterion_wrapper.X* because 
        # BaseDenseSplitter.X is not accessibe (sklearn problem).
        # TODO: receive in criterion.init
        self.criterion_wrapper.X_rows = np.ascontiguousarray(X[0], dtype=np.float64)
        self.criterion_wrapper.X_cols = np.ascontiguousarray(X[1], dtype=np.float64)
        self.n_row_features = X[0].shape[1]
        self.shape[0] = y.shape[0]
        self.shape[1] = y.shape[1]

        self.y = y

        if sample_weight == NULL:
            self.row_sample_weight = NULL
            self.col_sample_weight = NULL
        else:
            # First self.shape[0] sample weights are rows' the others
            # are columns'.
            self.row_sample_weight = sample_weight
            self.col_sample_weight = sample_weight + self.shape[0]

        self.splitter_rows.init(X[0], y, self.row_sample_weight)
        self.splitter_cols.init(X[1], yT, self.col_sample_weight)

        self.n_rows = self.splitter_rows.n_samples
        self.n_cols = self.splitter_cols.n_samples
        self.n_samples = self.n_rows * self.n_cols
        self.weighted_n_rows = self.splitter_rows.weighted_n_samples
        self.weighted_n_cols = self.splitter_cols.weighted_n_samples
        self.weighted_n_samples = self.weighted_n_rows * self.weighted_n_cols

        return 0

    cdef int node_reset(self, SIZE_t[2] start, SIZE_t[2] end,
                        double* weighted_n_node_samples) nogil except -1:
        """Reset splitters on node samples[start[0]:end[0], start[1]:end[1]].
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : SIZE_t[2]
            The (i, j) indices of the first sample to consider
        end : SIZE_t[2]
            The (i, j) indices of the last sample to consider
        weighted_n_node_samples[2] : ndarray, dtype=double pointer
            The total weight of those samples
        """
        cdef SIZE_t n_node_rows, n_node_cols
        cdef SIZE_t eff_min_rows_leaf
        cdef SIZE_t eff_min_cols_leaf

        n_node_rows = (end[0]-start[0])
        n_node_cols = (end[1]-start[1])
        # Ceil division.
        eff_min_rows_leaf = 1 + (self.min_samples_leaf-1) // n_node_cols
        eff_min_cols_leaf = 1 + (self.min_samples_leaf-1) // n_node_rows

        self.splitter_rows.min_samples_leaf = max(
            self.min_rows_leaf, eff_min_rows_leaf)
        self.splitter_cols.min_samples_leaf = max(
            self.min_cols_leaf, eff_min_cols_leaf)

        self.splitter_rows.start = start[0]
        self.splitter_rows.end = end[0]
        self.splitter_cols.start = start[1]
        self.splitter_cols.end = end[1]

        self.criterion_wrapper.init(
            self.y,
            self.row_sample_weight,
            self.col_sample_weight,
            self.weighted_n_rows,
            self.weighted_n_cols,
            start, end,
        )

        weighted_n_node_samples[0] = (
            self.criterion_wrapper.weighted_n_node_samples
        )
        # TODO: consider implementing
        # weighted_n_node_samples[1] = \
        #     self.criterion_wrapper.weighted_n_node_rows
        # weighted_n_node_samples[2] = \
        #     self.criterion_wrapper.weighted_n_node_cols

        return 0

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t[2] n_constant_features) nogil except -1:
        """Find the best split on node samples.
        It should return -1 upon errors.
        """
        cdef SplitRecord current_split, best_split
        cdef DOUBLE_t imp_left, imp_right

        _init_split(&best_split, self.splitter_rows.end, 0)

        # TODO: No need to "reorganize into samples" in each axis.
        # (sklearn.tree._splitter.pyx, line 417)

        # TODO: DRY. Cumbersome to have an array of Splitters in Cython.
        if (
            self.splitter_rows.end - self.splitter_rows.start
            >= 2 * self.splitter_rows.min_samples_leaf
        ):
            self.splitter_rows.node_split(impurity, &current_split,
                                          &n_constant_features[0])
            # NOTE: When no nice split have been  found, the child splitter sets
            # the split position at the end.
            if current_split.pos < self.splitter_rows.end:
                # Correct impurities.
                with gil:  # TODO: nogil
                    self.criterion_wrapper.children_impurity(
                        &imp_left, &imp_right, 0)
                imp_improve = self.criterion_wrapper.impurity_improvement(
                    impurity, imp_left, imp_right, 0)

                if imp_improve > best_split.improvement:  # Always?
                    best_split = current_split
                    best_split.improvement = imp_improve
                    best_split.impurity_left = imp_left
                    best_split.impurity_right = imp_right
                    best_split.axis = 0

        if (
            self.splitter_cols.end - self.splitter_cols.start
            >= 2 * self.splitter_cols.min_samples_leaf
        ):
            self.splitter_cols.node_split(impurity, &current_split,
                                          &n_constant_features[1])

            # NOTE: When no nice split have been  found, the child splitter sets
            # the split position at the end.
            if current_split.pos < self.splitter_cols.end:
                # Correct impurities.
                with gil:  # TODO: nogil
                    self.criterion_wrapper.children_impurity(
                        &imp_left, &imp_right, 1)
                imp_improve = self.criterion_wrapper.impurity_improvement(
                    impurity, imp_left, imp_right, 1)

                if imp_improve > best_split.improvement:
                    best_split = current_split
                    best_split.improvement = imp_improve
                    best_split.impurity_left = imp_left
                    best_split.impurity_right = imp_right
                    best_split.feature += self.n_row_features  # axis 1-exclusive.
                    best_split.axis = 1

        split[0] = best_split

    cdef void node_value(self, double* dest) nogil:
        """Copy the value (prototype) of node samples into dest."""
        self.criterion_wrapper.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""
        return self.criterion_wrapper.node_impurity()
