# cython: boundscheck=False
import numpy as np
from libc.stdlib cimport malloc, calloc, free, realloc
from libc.string cimport memset
from sklearn.tree._criterion cimport RegressionCriterion, Criterion
from ._axis_criterion cimport AxisCriterion, AxisClassificationCriterion


class InvalidAxisError(ValueError):
    def __init__(self, axis: int | None = None):
        self.axis = axis
    def __str__(self):
        return (
            "'axis' parameter can only be 0 or 1"
            f", not {self.axis}." if self.axis is not None else "."
        )


cdef class BipartiteCriterion:
    """Abstract base class."""

    cdef int init(
        self,
        const DTYPE_t[:, ::1] X_rows,
        const DTYPE_t[:, ::1] X_cols,
        const DOUBLE_t[:, :] y,
        DOUBLE_t[:] row_weights,
        DOUBLE_t[:] col_weights,
        double weighted_n_rows,
        double weighted_n_cols,
        const SIZE_t[:] row_indices,
        const SIZE_t[:] col_indices,
        SIZE_t[2] start,
        SIZE_t[2] end,
    ) nogil except -1:
        return -1

    cdef void node_value(self, double* dest) nogil:
        pass

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ) nogil:
        pass

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
        SIZE_t axis,
    ) nogil:
        """The final value to express the split quality. 
        """
    

cdef class GMO(BipartiteCriterion):
    """Applies Predictive Bi-Clustering Trees method.

    See [Pliakos _et al._, 2018](https://doi.org/10.1007/s10994-018-5700-x).
    """

    def __reduce__(self):
        return (
            type(self),
            (self.criterion_rows, self.criterion_cols),
            self.__getstate__(),
        )

    def __getstate__(self):
        return {}

    def __cinit__(self):
        # Default values
        self.sq_sum_total = 0.0

        self.start[0] = 0
        self.start[1] = 0
        self.end[0] = 0
        self.end[1] = 0

        self.weighted_n_node_samples = 0.0
        self.weighted_n_node_rows = 0.0
        self.weighted_n_node_cols = 0.0

        self.sum_total = np.zeros(self.n_outputs, dtype=np.float64)

    def __init__(
        self,
        AxisCriterion criterion_rows,
        AxisCriterion criterion_cols,
    ):
        # Objects must be set here, to ensure they are fully initialised
        self.criterion_rows = criterion_rows
        self.criterion_cols = criterion_cols

        self.n_rows = self.criterion_rows.n_samples
        self.n_cols = self.criterion_cols.n_samples
        self.n_outputs_rows = self.criterion_rows.n_outputs
        self.n_outputs_cols = self.criterion_cols.n_outputs

        # FIXME: should not validate arguments here.
        if isinstance(criterion_rows, AxisClassificationCriterion):
            if not isinstance(criterion_cols, AxisClassificationCriterion):
                raise TypeError(
                    "None or both axes criteria must be "
                    "AxisClassificationCriterion."
                )
            self.max_n_classes = (
                (<AxisClassificationCriterion>criterion_cols).max_n_classes
            )
        else:
            self.max_n_classes = 1

        self.n_outputs = self.n_outputs_rows + self.n_outputs_cols

    cdef int init(
        self,
        const DTYPE_t[:, ::1] X_rows,
        const DTYPE_t[:, ::1] X_cols,
        const DOUBLE_t[:, :] y,
        DOUBLE_t[:] row_weights,
        DOUBLE_t[:] col_weights,
        double weighted_n_rows,
        double weighted_n_cols,
        const SIZE_t[:] row_indices,
        const SIZE_t[:] col_indices,
        SIZE_t[2] start,
        SIZE_t[2] end,
    ) nogil except -1:
        """This function adapts RegressionCriterion.init to 2D data."""

        # Initialize fields
        self.X_rows = X_rows
        self.X_cols = X_cols
        self.y = y
        self.row_weights = row_weights
        self.col_weights = col_weights
        self.weighted_n_rows = weighted_n_rows
        self.weighted_n_cols = weighted_n_cols
        self.weighted_n_samples = weighted_n_rows * weighted_n_cols
        self.row_indices = row_indices
        self.col_indices = col_indices

        self.start[0], self.start[1] = start[0], start[1]
        self.end[0], self.end[1] = end[0], end[1]

        cdef SIZE_t n_node_rows = end[0] - start[0]
        cdef SIZE_t n_node_cols = end[1] - start[1]

        self.criterion_rows.axis_init(
            y=self.y,
            sample_weight=self.row_weights,
            col_weights=self.col_weights,
            weighted_n_samples=self.weighted_n_rows,
            weighted_n_cols=self.weighted_n_cols,
            sample_indices=self.row_indices,
            col_indices=self.col_indices,
            start=self.start[0],
            end=self.end[0],
            start_col=self.start[1],
            end_col=self.end[1],
        )
        self.criterion_cols.axis_init(
            y=self.y.T,
            sample_weight=self.col_weights,
            col_weights=self.row_weights,
            weighted_n_samples=self.weighted_n_cols,
            weighted_n_cols=self.weighted_n_rows,
            sample_indices=self.col_indices,
            col_indices=self.row_indices,
            start=self.start[1],
            end=self.end[1],
            start_col=self.start[0],
            end_col=self.end[0],
        )

        # Will be used by TreeBuilder as stopping criteria.
        self.weighted_n_node_rows = self.criterion_rows.weighted_n_node_samples
        self.weighted_n_node_cols = self.criterion_cols.weighted_n_node_samples

        # Will further be used by the BipartiteSplitter to set the Tree object
        self.weighted_n_node_samples = (
            self.weighted_n_node_rows * self.weighted_n_node_cols
        )

        return 0

    cdef void node_value(self, double* dest) nogil:
        """Copy the value (prototype) of node sample_indices into dest.
        """
        self.criterion_cols.node_value(dest)
        self.criterion_rows.node_value(
            dest + self.n_outputs_cols * self.max_n_classes
        )

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node.

        In scikit-learn trees it is only used at the root node.
        """
        # Should be equal among axes.
        return self.criterion_rows.node_impurity()

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ) nogil:
        (<AxisCriterion> self._get_criterion(axis)).children_impurity(
            impurity_left,
            impurity_right,
        )

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
        SIZE_t axis,
    ) nogil:
        """The final value to express the split quality. 
        """
        return (<AxisCriterion>self._get_criterion(axis)).impurity_improvement(
            impurity_parent,
            impurity_left,
            impurity_right,
        )
 
    cdef void* _get_criterion(self, SIZE_t axis) nogil:
        if axis == 0:
            return <void*> self.criterion_rows
        elif axis == 1:
            return <void*> self.criterion_cols
        else:
            with gil:
                raise InvalidAxisError(axis)


cdef class GMOSA(GMO):
    def __init__(
        self,
        AxisCriterion criterion_rows,
        AxisCriterion criterion_cols,
    ):
        super().__init__(criterion_rows, criterion_cols)
        self.n_outputs = 1

    cdef void node_value(self, double* dest) nogil:
        """Copy the value (prototype) of node sample_indices into dest.
        """
        self.criterion_rows.total_node_value(dest)