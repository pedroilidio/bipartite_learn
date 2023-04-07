# cython: boundscheck=False
import copy
import warnings
from sklearn.tree._splitter cimport Splitter
from sklearn.tree._criterion cimport RegressionCriterion, Criterion

import numpy as np
cimport numpy as np

np.import_array()
cdef double INFINITY = np.inf


cdef inline void _init_split(
    MultipartiteSplitRecord* self,
    SIZE_t start_pos,
    SIZE_t axis
) nogil:
    self.axis = axis
    self.split_record.impurity_left = INFINITY
    self.split_record.impurity_right = INFINITY
    self.split_record.pos = start_pos
    self.split_record.feature = 0
    self.split_record.threshold = 0.
    self.split_record.improvement = -INFINITY


cdef class BipartiteSplitter:
    """PBCT splitter implementation.

    Wrapper class to coordinate one Splitter for each axis in a two-dimensional
    problem as described by Pliakos _et al._ (2018).
    """
    def __cinit__(
        self,
        Splitter splitter_rows,
        Splitter splitter_cols,
        BipartiteCriterion bipartite_criterion,
        min_samples_leaf,
        min_weight_leaf,
    ):
        """Store each axis' splitter."""
        self.n_samples = 0
        self.weighted_n_samples = 0.0
        self.splitter_rows = splitter_rows
        self.splitter_cols = splitter_cols
        self.bipartite_criterion = bipartite_criterion

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
                self.bipartite_criterion,
                self.min_samples_leaf,
                self.min_weight_leaf,
            ),
            self.__getstate__(),
        )

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(
        self,
        object X,
        const DOUBLE_t[:, :] y,
        const DOUBLE_t[:] sample_weight,  # TODO: test sample_weight
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
        self.y = y
        self.n_rows = y.shape[0]
        self.n_cols = y.shape[1]

        self.X_rows = X[0]
        self.X_cols = X[1]

        self.n_row_features = self.X_rows.shape[1]

        if sample_weight is None:
            self.row_weights = None
            self.col_weights = None
        else:
            # First self.n_rows sample weights refer to rows, the others
            # refer to columns.
            self.row_weights = sample_weight
            self.col_weights = sample_weight[self.n_rows:]

        # splitters do not receive y, the Bipartite criterion is who takes
        # charge of calling Criterion.init() passing y to the criterion.
        self.splitter_rows.init(X[0], None, self.row_weights)
        self.splitter_cols.init(X[1], None, self.col_weights)
        self.row_indices = self.splitter_rows.samples
        self.col_indices = self.splitter_cols.samples

        self.n_samples = self.n_rows * self.n_cols
        self.weighted_n_rows = self.splitter_rows.weighted_n_samples
        self.weighted_n_cols = self.splitter_cols.weighted_n_samples
        self.weighted_n_samples = self.weighted_n_rows * self.weighted_n_cols

        return 0

    cdef int node_reset(
        self,
        SIZE_t[2] start,
        SIZE_t[2] end,
        double* weighted_n_node_samples
    ) nogil except -1:
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

        n_node_rows = end[0] - start[0]
        n_node_cols = end[1] - start[1]

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

        self.start[0], self.start[1] = start[0], start[1]
        self.end[0], self.end[1] = end[0], end[1]

        self.bipartite_criterion.init(
            self.X_rows,
            self.X_cols,
            self.y,
            self.row_weights,
            self.col_weights,
            self.weighted_n_rows,
            self.weighted_n_cols,
            self.row_indices,
            self.col_indices,
            start,
            end,
        )

        weighted_n_node_samples[0] = (
            self.bipartite_criterion.weighted_n_node_samples
        )
        weighted_n_node_samples[1] = (
            self.bipartite_criterion.weighted_n_node_rows
        )
        weighted_n_node_samples[2] = (
            self.bipartite_criterion.weighted_n_node_cols
        )

        return 0

    cdef int node_split(
        self,
        double impurity,
        MultipartiteSplitRecord* split,
        SIZE_t[2] n_constant_features
    ) nogil except -1:
        """Find the best split on node samples.
        It should return -1 upon errors.
        """
        cdef SIZE_t ax  # axis, 0 == rows, 1 == columns
        cdef MultipartiteSplitRecord current_split, best_split
        cdef DOUBLE_t imp_left, imp_right

        # Using Cython casting trick to iterate over splitters without the GIL
        cdef (void*)[2] splitters
        splitters[0] = <void*> self.splitter_rows
        splitters[1] = <void*> self.splitter_cols

        _init_split(&best_split, self.splitter_rows.end, 0)

        # TODO: No need to "reorganize into samples" in each axis.
        # (sklearn.tree._splitter.pyx, line 417)
        for ax in range(2):
            if (
                (self.end[ax] - self.start[ax])
                < 2 * (<Splitter>splitters[ax]).min_samples_leaf
            ):
                continue  # min_samples_leaf not satisfied.

            (<Splitter>splitters[ax]).node_split(
                impurity,
                <SplitRecord*> &current_split,
                &n_constant_features[ax],
            )

            # When no valid split has been found, the child splitter sets
            # the split position at the end of the current node.
            if current_split.split_record.pos == self.end[ax]:
                continue

            self.bipartite_criterion.children_impurity(
                &imp_left, &imp_right, axis=ax,
            )
            imp_improve = self.bipartite_criterion.impurity_improvement(
                impurity, imp_left, imp_right, axis=ax,
            )

            if imp_improve > best_split.split_record.improvement:
                best_split = current_split
                best_split.split_record.improvement = imp_improve
                best_split.split_record.impurity_left = imp_left
                best_split.split_record.impurity_right = imp_right

                best_split.axis = ax
                if ax == 1:
                    best_split.split_record.feature += self.n_row_features

        split[0] = best_split

    cdef void node_value(self, double* dest) nogil:
        """Copy the value (prototype) of node samples into dest."""
        self.bipartite_criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""
        return self.bipartite_criterion.node_impurity()
