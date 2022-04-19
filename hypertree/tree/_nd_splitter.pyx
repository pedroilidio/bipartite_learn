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
            RegressionCriterionWrapper2D criterion_wrapper,
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

        self.n_samples = self.splitter_rows.n_samples * \
                         self.splitter_cols.n_samples
        self.weighted_n_samples = self.splitter_rows.weighted_n_samples * \
                                  self.splitter_cols.weighted_n_samples

        ## Done in criterion_wrapper.init() # FIXME: it shouldn't be!
        #self.splitter_rows.weighted_n_samples = self.weighted_n_samples
        #self.splitter_cols.weighted_n_samples = self.weighted_n_samples

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

        # max.
        if self.min_rows_leaf > eff_min_rows_leaf:
            self.splitter_rows.min_samples_leaf = self.min_rows_leaf
        else:
            self.splitter_rows.min_samples_leaf = eff_min_rows_leaf
        if self.min_cols_leaf > eff_min_cols_leaf:
            self.splitter_cols.min_samples_leaf = self.min_cols_leaf
        else:
            self.splitter_cols.min_samples_leaf = eff_min_cols_leaf

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

        ## Done in criterion_wrapper.init()
        # self.splitter_rows.criterion.reset()
        # self.splitter_cols.criterion.reset()

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

        _init_split(&best_split, self.splitter_rows.end, 0)

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
            # NOTE: When no nice split have been  found, the child splitter sets
            # the split position at the end.
            if current_split.pos < self.splitter_rows.end:
                # Correct impurities.
                self.criterion_wrapper.children_impurity(
                    &imp_left, &imp_right, 0)
                imp_improve = self.splitter_rows.criterion.impurity_improvement(
                    impurity, imp_left, imp_right)


                # if imp_improve > best_split.improvement:  # Always.
                best_split = current_split
                best_split.improvement = imp_improve
                best_split.impurity_left = imp_left
                best_split.impurity_right = imp_right
                best_split.axis = 0

        if (self.splitter_cols.end - self.splitter_cols.start) >= 2:
            self.splitter_cols.node_split(impurity, &current_split,
                                          n_constant_features+1)
            # # FIXME: shouldn't need this if.

            # NOTE: When no nice split have been  found, the child splitter sets
            # the split position at the end.
            if current_split.pos < self.splitter_cols.end:
                # Correct impurities.
                self.criterion_wrapper.children_impurity(
                    &imp_left, &imp_right, 1)
                imp_improve = self.splitter_cols.criterion.impurity_improvement(
                    impurity, imp_left, imp_right)

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


def make_2d_splitter(
       splitter_class,
       criterion_class,
       n_samples,
       max_features,
       n_outputs=1,
       min_samples_leaf=1,
       min_weight_leaf=0.0,
       ax_min_samples_leaf=1,
       ax_min_weight_leaf=0.0,
       random_state=None,
       criterion_wrapper_class=MSE_Wrapper2D,
    ):
    """Factory function of Splitter2D instances.

    Since the building of a Splitter2D is somewhat counterintuitive, this func-
    tion is provided to simplificate the process. With exception of n_samples,
    the remaining parameters may be set to a single value or a 2-valued
    tuple or list, to specify them for each axis.
    """
    if type(max_features) not in {list, tuple}:
        max_features = [max_features, max_features]
    if type(ax_min_samples_leaf) not in {list, tuple}:
        ax_min_samples_leaf = [ax_min_samples_leaf, ax_min_samples_leaf]
    if type(ax_min_weight_leaf) not in {list, tuple}:
        ax_min_weight_leaf = [ax_min_weight_leaf, ax_min_weight_leaf]
    if type(splitter_class) not in {list, tuple}:
        splitter_class = [splitter_class, splitter_class]
    if type(criterion_class) not in {list, tuple}:
        criterion_class = [criterion_class, criterion_class]

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    # Criteria.
    if isinstance(criterion_class[0], Criterion):
        criterion_rows = copy.deepcopy(criterion_class[0])
    else:
        criterion_rows = criterion_class[0](
            n_outputs=n_outputs,
            n_samples=n_samples[0])

    if isinstance(criterion_class[1], Criterion):
        criterion_cols = copy.deepcopy(criterion_class[1])
    else:
        criterion_cols = criterion_class[1](
            n_outputs=n_outputs,
            n_samples=n_samples[1])

    if isinstance(splitter_class[0], Splitter):
        if criterion_class[0] is not None:
            warnings.warn("Since splitter_class[0] is not a class, the provided"
                          " criterion_class[0] is being ignored.")
        splitter_rows = copy.deepcopy(splitter_class[0])
    else:
        splitter_rows = splitter_class[0](
            criterion=criterion_rows,
            max_features=max_features[0],
            min_samples_leaf=ax_min_samples_leaf[0],
            min_weight_leaf=ax_min_weight_leaf[0],
            random_state=random_state,
        )
    if isinstance(splitter_class[1], Splitter):
        if criterion_class[1] is not None:
            warnings.warn("Since splitter_class[1] is not a class, the provided"
                          " criterion_class[1] is being ignored.")
        splitter_cols = copy.deepcopy(splitter_class[1])
    else:
        splitter_cols = splitter_class[1](
            criterion=criterion_cols,
            max_features=max_features[1],
            min_samples_leaf=ax_min_samples_leaf[1],
            min_weight_leaf=ax_min_weight_leaf[1],
            random_state=random_state,
        )

    # Wrap criteria.
    criterion_wrapper = \
        criterion_wrapper_class(
            splitter_rows,
            splitter_cols,
    )

    # Wrap splitters.
    return Splitter2D(
        splitter_rows=splitter_rows,
        splitter_cols=splitter_cols,
        criterion_wrapper=criterion_wrapper,
        min_samples_leaf=min_samples_leaf,
        min_weight_leaf=min_weight_leaf,
    )
