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

        self.n_samples = self.splitter_rows.n_samples * \
                         self.splitter_cols.n_samples
        self.weighted_n_samples = self.splitter_rows.weighted_n_samples * \
                                  self.splitter_cols.weighted_n_samples

        # TODO: Do it in criterion_wrapper.init()
        self.splitter_rows.weighted_n_samples = self.weighted_n_samples
        self.splitter_cols.weighted_n_samples = self.weighted_n_samples

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
            start, end,
        )

        # # Done in criterion_wrapper.init()
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
                with gil:
                    self.criterion_wrapper.children_impurity(
                        &imp_left, &imp_right, 0)
                imp_improve = self.criterion_wrapper.impurity_improvement(
                    impurity, imp_left, imp_right, 0)

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
                with gil:
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


def make_2d_splitter(
       splitters,
       criteria,
       n_samples=None,
       max_features=None,
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

    ax_min_samples_leaf represents [min_rows_leaf, min_cols_leaf]
    """
    if not isinstance(n_samples, (list, tuple)):
        n_samples = [n_samples, n_samples]
    if not isinstance(max_features, (list, tuple)):
        max_features = [max_features, max_features]
    if not isinstance(n_outputs, (list, tuple)):
        n_outputs = [n_outputs, n_outputs]
    if not isinstance(ax_min_samples_leaf, (list, tuple)):
        ax_min_samples_leaf = [ax_min_samples_leaf, ax_min_samples_leaf]
    if not isinstance(ax_min_weight_leaf, (list, tuple)):
        ax_min_weight_leaf = [ax_min_weight_leaf, ax_min_weight_leaf]
    if not isinstance(splitters, (list, tuple)):
        splitters = [copy.deepcopy(splitters), copy.deepcopy(splitters)]
    if not isinstance(criteria, (list, tuple)):
        criteria = [copy.deepcopy(criteria), copy.deepcopy(criteria)]
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    for ax in range(2):
        # Make criterion
        if isinstance(criteria[ax], type):
            if n_samples[ax] is None:
                raise ValueError(
                    f"n_samples[{ax}] must be provided if criteria"
                    f"[{ax}]={criteria[ax]} is a Criterion type.")
            criteria[ax] = criteria[ax](
                n_outputs=n_outputs[ax],
                n_samples=n_samples[ax])
        else:
            criteria[ax] = copy.deepcopy(criteria[ax])

        # Make splitter
        if isinstance(splitters[ax], type):
            if max_features[ax] is None:
                raise ValueError(
                    f"max_features[{ax}] must be provided if splitters"
                    f"[{ax}]={splitters[ax]} is a Splitter type.")
            splitters[ax] = splitters[ax](
                criterion=criteria[ax],
                max_features=max_features[ax],
                min_samples_leaf=ax_min_samples_leaf[ax],
                min_weight_leaf=ax_min_weight_leaf[ax],
                random_state=random_state,
            )
        else:
            if criteria[ax] is not None:
                warnings.warn("Since splitters[ax] is not a class, the provided"
                            " criteria[ax] is being ignored.")
            splitters[ax] = copy.deepcopy(splitters[ax])

    # Wrap criteria.
    criterion_wrapper = \
        criterion_wrapper_class(splitters[0], splitters[1])

    # Wrap splitters.
    return Splitter2D(
        splitter_rows=splitters[0],
        splitter_cols=splitters[1],
        criterion_wrapper=criterion_wrapper,
        min_samples_leaf=min_samples_leaf,
        min_weight_leaf=min_weight_leaf,
    )
