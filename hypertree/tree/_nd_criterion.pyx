# cython: boundscheck=False
from sklearn.tree._criterion cimport RegressionCriterion, Criterion
from sklearn.tree._criterion import MSE
from time import time
from libc.stdlib cimport malloc, calloc, free, realloc
from libc.string cimport memset

import numpy as np
cimport numpy as np

np.import_array()
from copy import deepcopy


cdef class RegressionCriterionWrapper2D:
    def __cinit__(self, list children_criteria):
        # cdef RegressionCriterion2D criterion
        self.children_criteria = children_criteria
        # Should be same as children_criteria[1].n_outputs
        self.criterion_rows = self.children_criteria[0]
        self.criterion_cols = self.children_criteria[1]
        self.n_outputs = self.criterion_rows.n_outputs

        # Default values
        self.row_sample_weight = NULL
        self.col_sample_weight = NULL
        self.total_row_sample_weight = NULL
        self.total_col_sample_weight = NULL
        # self.y_row_sums = NULL
        # self.y_col_sums = NULL

        self.row_samples = NULL
        self.col_samples = NULL

        self.sq_sum_total = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL

        # Allocate memory for the accumulators
        self.sum_total = <double*> calloc(self.n_outputs, sizeof(double))

        if self.sum_total == NULL:
            raise MemoryError()

    def __init__(self, list children_criteria):
        cdef RegressionCriterion criterion
        for criterion in children_criteria:
            if criterion.n_outputs > 1:
                raise NotImplementedError(
                    "Multi-output not implemented. Set n_outputs=1 for all"
                    "Criterion objects.")

    def __dealloc__(self):
        free(self.sum_total)
        free(self.total_row_sample_weight)
        free(self.total_col_sample_weight)

    cdef int init(
            self, const DOUBLE_t[:, ::1] y_2D,
            DOUBLE_t* row_sample_weight,
            DOUBLE_t* col_sample_weight,
            double weighted_n_samples,
            SIZE_t* row_samples, SIZE_t* col_samples,
            SIZE_t[2] start, SIZE_t[2] end,
            SIZE_t[2] y_shape,
        ) nogil except -1:
        """This function adapts RegressionCriterion.init to 2D data."""
        # NOTE: A problem is sometimes n_outputs is actually treated the
        # number of outputs, but sometimes it is just an alias for y.shape[1].
        # In 1D, they have the same value, but now we have to discern them.

        cdef int rc  # Return code
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t p
        cdef SIZE_t q
        cdef DOUBLE_t y_ij
        cdef DOUBLE_t w_y_ij
        cdef DOUBLE_t w=1.0, wi=1.0, wj=1.0

        cdef SIZE_t n_rows = y_shape[0]
        cdef SIZE_t n_cols = y_shape[1]

        # total_row_sample_weight will correspond, for each row, to the weight
        # of the row times the total weight of all columns (i.e. the sum of all 
        # col_sample_weight's elements). If they were numpy arrays, it would be:
        #
        #       sample_weights * col_sample_weight.sum()
        #
        # NOTE: maybe we should use a [:, ::1] sample_weight matrix instead.
        # TODO: weight sum is stored in Splitter.weighted_n_sample
        if self.total_row_sample_weight == NULL:
        ##if self.total_col_samples_weight == NULL:  # same.
            self.total_row_sample_weight = \
                <DOUBLE_t*> malloc(n_rows * sizeof(DOUBLE_t))
            self.total_col_sample_weight = \
                <DOUBLE_t*> malloc(n_cols * sizeof(DOUBLE_t))

            if (self.total_row_sample_weight == NULL or
                self.total_col_sample_weight == NULL):
                raise MemoryError()

        # Initialize fields
        self.y_2D = y_2D
        self.row_sample_weight = row_sample_weight
        self.col_sample_weight = col_sample_weight
        self.weighted_n_samples = weighted_n_samples
        self.row_samples = row_samples
        self.col_samples = col_samples
        self.start[0], self.start[1] = start[0], start[1]
        self.end[0], self.end[1] = end[0], end[1]

        self.weighted_n_node_samples = 0.0
        self.sq_sum_total = 0.0

        self.weighted_n_row_samples = 0.0
        self.weighted_n_col_samples = 0.0

        # TODO: malloc/memset instead of np.
        # TODO: Since single output, use fortran contiguous? Does it make any
        # difference?
        # TODO: only zero where you need.
        with gil:
            self.y_row_sums = np.zeros((n_rows, self.n_outputs), order='C')
            self.y_col_sums = np.zeros((n_cols, self.n_outputs), order='C')


        # TODO: implement multi-output.
        memset(self.sum_total, 0, self.n_outputs * sizeof(double))

        for p in range(start[0], end[0]):
            i = row_samples[p]
            # self.y_row_sums[p] = 0  # TODO: only zero where you need.

            if row_sample_weight != NULL:
                wi = row_sample_weight[i]

            for q in range(start[1], end[1]):
                j = col_samples[q]

                if col_sample_weight != NULL:
                    wj = col_sample_weight[j]

                # TODO: multi-output
                y_ij = y_2D[i, j]
                w = wi * wj
                w_y_ij = w * y_ij

                # self.y_row_sums[i, 0] = self.y_row_sums[i, 0] + y_ij
                # self.y_col_sums[j, 0] = self.y_col_sums[j, 0] + y_ij
                self.y_row_sums[i, 0] += wj * y_ij
                self.y_col_sums[j, 0] += wi * y_ij

                # NOTE: we apparently save operations looping after instead.
                # self.total_row_sample_weight[i] += w
                # self.total_col_sample_weight[j] += w

                self.sum_total[0] += w_y_ij
                self.sq_sum_total += w_y_ij * y_ij

                self.weighted_n_node_samples += w

        # Set weighted axis n_samples.
        if self.row_sample_weight == NULL:
            self.weighted_n_row_samples = end[0] - start[0]
        else:
            for p in range(start[0], end[0]):
                i = row_samples[p]
                self.weighted_n_row_samples += row_sample_weight[i]

        if self.col_sample_weight == NULL:
            self.weighted_n_col_samples = end[1] - start[1]
        else:
            for q in range(start[1], end[1]):
                j = col_samples[q]
                self.weighted_n_col_samples += col_sample_weight[j]

        # Build total_row[col]_sample_weight
        for p in range(start[0], end[0]):
            i = row_samples[p]

            if self.row_sample_weight != NULL:
                self.total_row_sample_weight[i] = \
                    self.row_sample_weight[i] * self.weighted_n_col_samples
            else:
                self.total_row_sample_weight[i] = self.weighted_n_col_samples

            # TODO: Multioutput
            self.y_row_sums[i, 0] = \
                self.y_row_sums[i, 0]/self.weighted_n_col_samples

        for q in range(start[1], end[1]):
            j = col_samples[q]

            if self.col_sample_weight != NULL:
                self.total_col_sample_weight[j] = \
                    self.col_sample_weight[j] * self.weighted_n_row_samples
            else:
                self.total_col_sample_weight[j] = self.weighted_n_row_samples

            # NOTE: divide to multiply after in Criterion.update().
            # Not the most efficient, but it was the only way I saw to keep
            # the sklearn class unmodified.
            # FIXME: the name should be y_col_means.
            # TODO: Multioutput
            self.y_col_sums[j, 0] = \
                self.y_col_sums[j, 0] / self.weighted_n_row_samples

        rc = self._init_child_criterion(
            self.criterion_rows,
            self.y_row_sums,
            self.total_row_sample_weight,
            self.row_samples,
            self.start[0], self.end[0],
        )
        rc += self._init_child_criterion(
            self.criterion_cols,
            self.y_col_sums,
            self.total_col_sample_weight,
            self.col_samples,
            self.start[1], self.end[1],
        )

        if rc:
            rc = -1
        return rc

    cdef int _init_child_criterion(
            self,
            RegressionCriterion child_criterion,
            const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            SIZE_t* samples, SIZE_t start,
            SIZE_t end,
    ) nogil except -1:
        """Substitutes criterion.init() initializing child criterion on 2D data.

        This initializes the children criteria at node samples[start:end] and children
        samples[start:start] and samples[start:end].
        """
        # Initialize fields
        child_criterion.y = y
        child_criterion.sample_weight = sample_weight
        child_criterion.samples = samples
        child_criterion.start = start
        child_criterion.end = end
        child_criterion.n_node_samples = end - start

        # Copy some from self
        child_criterion.sum_total[0] = self.sum_total[0]
        child_criterion.weighted_n_samples = self.weighted_n_samples
        child_criterion.weighted_n_node_samples = self.weighted_n_node_samples
        child_criterion.sq_sum_total = self.sq_sum_total

        # Reset to pos=start
        child_criterion.reset()

        return 0

    cdef void node_value(self, double* dest) nogil:
        """Copy the value (prototype) of node samples into dest."""
        # It should be the same as criterion_cols.node_values().
        self.criterion_rows.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""
        # It should be the same as criterion_cols.node_impurity().
        return self.criterion_rows.node_impurity()

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ) nogil:
        if axis:
            self.criterion_cols.children_impurity(impurity_left, impurity_right)
        else:
            self.criterion_rows.children_impurity(impurity_left, impurity_right)

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right,
                                     SIZE_t axis) nogil:
        if axis:
            return self.criterion_cols.impurity_improvement(
                impurity_parent,
                impurity_left,
                impurity_right,
            )
        else:
            return self.criterion_rows.impurity_improvement(
                impurity_parent,
                impurity_left,
                impurity_right,
            )


cdef class MSE_Wrapper2D(RegressionCriterionWrapper2D):
    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ) nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """
        cdef SIZE_t[2] end
        cdef DOUBLE_t y_ij

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i, j, q, p, k
        cdef DOUBLE_t weighted_n_left, weighted_n_right
        cdef DOUBLE_t w = 1.0

        cdef SIZE_t pos
        cdef double* sum_left
        cdef double* sum_right

        with gil:
            if axis:
                criterion = self.criterion_cols
            else:
                criterion = self.criterion_rows

        pos = criterion.pos

        sum_left = criterion.sum_left
        sum_right = criterion.sum_right
        weighted_n_left = criterion.weighted_n_left
        weighted_n_right = criterion.weighted_n_right

        end[0], end[1] = self.end[0], self.end[1]
        end[axis] = pos

        for p in range(self.start[0], end[0]):
            i = self.row_samples[p]
            for q in range(self.start[1], end[1]):
                j = self.col_samples[q]

                w = 1.0
                if self.row_sample_weight != NULL:
                    w = self.row_sample_weight[i]
                if self.col_sample_weight != NULL:
                    w *= self.col_sample_weight[j]

                # TODO: multi-output
                y_ij = self.y_2D[i, j]
                sq_sum_left += w * y_ij * y_ij

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / weighted_n_left
        impurity_right[0] = sq_sum_right / weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (sum_left[k] / weighted_n_left) ** 2.0
            impurity_right[0] -= (sum_right[k] / weighted_n_right) ** 2.0
        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs
