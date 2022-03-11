from pprint import pprint  # TODO
from sklearn.tree._splitter cimport Splitter
from sklearn.tree._criterion cimport RegressionCriterion
from _nd_criterion cimport RegressionCriterionWrapper2D, MSE_Wrapper2D

import numpy as np
cimport numpy as np

np.import_array()
cdef double INFINITY = np.inf


cdef class Splitter2D:
    """PBCT splitter implementation.

    Wrapper class to coordinate one Splitter for each axis in a two-dimensional
    problem as described by Pliakos _et al._ (2018).
    """
    def __cinit__(self, Splitter splitter_rows, Splitter splitter_cols,
                  RegressionCriterionWrapper2D criterion_wrapper):
        """Store each axis' splitter."""
        self.n_samples = 0
        self.weighted_n_samples = 0.0
        self.splitter_rows = splitter_rows
        self.splitter_cols = splitter_cols
        self.criterion_wrapper = criterion_wrapper

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self,
                  object X,
                  const DOUBLE_t[:, ::1] y,
                  (DOUBLE_t*)[2] sample_weight,
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
        self.n_row_features = X[0].shape[1]
        self.shape[0] = y.shape[0]
        self.shape[1] = y.shape[1]

        self.y = y

        if sample_weight == NULL:
            self.row_sample_weight = NULL
            self.col_sample_weight = NULL
        else:
            self.row_sample_weight = sample_weight[0]
            self.col_sample_weight = sample_weight[1]

        self.splitter_rows.init(X[0], y, self.row_sample_weight)
        self.splitter_cols.init(X[1], yT, self.col_sample_weight)

        self.n_samples = self.splitter_rows.n_samples * \
                         self.splitter_cols.n_samples
        self.weighted_n_samples = self.splitter_rows.weighted_n_samples * \
                                  self.splitter_cols.weighted_n_samples

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
        # self.start = start
        # self.end = end
        self.splitter_rows.start = start[0]
        self.splitter_rows.end = end[0]

        self.splitter_cols.start = start[1]
        self.splitter_cols.end = end[1]

        self.criterion_wrapper.init(
            self.y,
            self.row_sample_weight,
            self.col_sample_weight,
            self.weighted_n_samples,
            self.splitter_rows.samples,
            self.splitter_cols.samples,
            start, end,
            self.shape
        )

        self.splitter_rows.criterion.reset()
        self.splitter_cols.criterion.reset()

        weighted_n_node_samples[0] = \
            self.criterion_wrapper.weighted_n_node_samples

        return 0

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t[2] n_constant_features) nogil except -1:
        """Find the best split on node samples.
        It should return -1 upon errors.
        """
        cdef SplitRecord current_split, best_split
        cdef DOUBLE_t imp_left, imp_right

        best_split.improvement = -INFINITY

        # TODO: No need to "reorganize into samples" in each axis.
        # (sklearn.tree._splitter.pyx, line 417)

        # TODO: DRY. Cumbersome to have an array of Splitters in Cython.

        # If only one sample in axis, do not split that axis.
        # Necessary since min_samples_leaf only sees both axes at the same time,
        # so eventually, for instance, it would try to split a 1x23 matrix in
        # the rows direction.
        # TODO: axis specific stopping criteria (min_samples_leaf for example).
        if (self.splitter_rows.end - self.splitter_rows.start) >= 2:
            self.splitter_rows.node_split(impurity, &current_split,
                                          n_constant_features)
            # Correct impurities.
            self.criterion_wrapper.children_impurity(
                &imp_left, &imp_right, 0)
            self.criterion_wrapper.children_impurity(
                &imp_left, &imp_right, 0)
            imp_improve = self.criterion_wrapper.impurity_improvement(
                impurity, imp_left, imp_right, 0)

            if imp_improve > best_split.improvement:
                best_split = current_split
                best_split.improvement = imp_improve
                best_split.impurity_left = imp_left
                best_split.impurity_right = imp_right
                best_split.axis = 0

        if (self.splitter_cols.end - self.splitter_cols.start) >= 2:
            self.splitter_cols.node_split(impurity, &current_split,
                                          n_constant_features+1)
            # Correct impurities.
            self.criterion_wrapper.children_impurity(
                &imp_left, &imp_right, 1)
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


cpdef Splitter2D gen_splitter_2d(
       type splitter_class,
       type criterion_class,
       object shape,
       object n_attrs,
       SIZE_t n_outputs=1,
       object min_samples_leaf=None,
       object min_weight_leaf=None,
       object random_state=None,
       type criteria_wrapper_class=MSE_Wrapper2D,
    ):
    if type(min_samples_leaf) not in {list, tuple}:
        min_samples_leaf = [min_samples_leaf, min_samples_leaf]
    if type(min_weight_leaf) not in {list, tuple}:
        min_weight_leaf = [min_weight_leaf, min_weight_leaf]

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    # Criteria.
    criterion_rows = criterion_class(
        n_outputs=n_outputs,
        n_samples=shape[0])
    criterion_cols = criterion_class(
        n_outputs=n_outputs,
        n_samples=shape[1])

    # Splitters.
    cdef Splitter splitter_rows = splitter_class(
        criterion=criterion_rows,
        max_features=n_attrs[0],
        min_samples_leaf=min_samples_leaf[0],
        min_weight_leaf=min_weight_leaf[0],
        random_state=random_state,
    )
    cdef Splitter splitter_cols = splitter_class(
        criterion=criterion_cols,
        max_features=n_attrs[1],
        min_samples_leaf=min_samples_leaf[1],
        min_weight_leaf=min_weight_leaf[1],
        random_state=random_state,
    )

    # Wrap criteria.
    cdef RegressionCriterionWrapper2D criteria_wrapper = criteria_wrapper_class(
        splitter_rows.criterion,
        splitter_cols.criterion,
    )

    # Wrap splitters.
    return Splitter2D(
        splitter_rows, splitter_cols, criteria_wrapper)
