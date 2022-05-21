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
    def __cinit__(self, Splitter splitter_rows, Splitter splitter_cols):
        self.splitter_rows = splitter_rows
        self.splitter_cols = splitter_cols
        self.n_outputs = self.splitter_rows.criterion.n_outputs

        # Default values
        self.row_sample_weight = NULL
        self.col_sample_weight = NULL
        self.total_row_sample_weight = NULL
        self.total_col_sample_weight = NULL
        self.sq_sum_total = 0.0

    def __dealloc__(self):
        free(self.row_sample_weight)
        free(self.col_sample_weight)
        free(self.total_row_sample_weight)
        free(self.total_col_sample_weight)

    cdef int init(
            self, const DOUBLE_t[:, ::1] y_2D,
            DOUBLE_t* row_sample_weight,
            DOUBLE_t* col_sample_weight,
            double weighted_n_samples,
            SIZE_t[::1] row_samples, SIZE_t[::1] col_samples,
            SIZE_t[2] start, SIZE_t[2] end,
            SIZE_t[2] y_shape,
        ) nogil except -1:
        """This function adapts RegressionCriterion.init to 2D data."""
        # NOTE: A problem is sometimes n_outputs is actually treated the
        # number of outputs, but sometimes it is just an alias for y.shape[1].
        # In 1D, they have the same value, but now we have to discern them.

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
        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))

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

                self.y_row_sums[i, 0] += wj * y_ij
                self.y_col_sums[j, 0] += wi * y_ij

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

        # FIXME what to do with this? Get self.weighted_n_samples from it?
        cdef double[2] wnns  # will be discarded

        if -1 == self._node_reset_child_splitter(
            child_splitter=self.splitter_rows,
            y=self.y_row_sums,
            sample_weight=self.total_row_sample_weight,
            start=start[0],
            end=end[0],
            weighted_n_node_samples=wnns,
        ):
            return -1

        if -1 == self._node_reset_child_splitter(
            child_splitter=self.splitter_cols,
            y=self.y_col_sums,
            sample_weight=self.total_col_sample_weight,
            start=start[1],
            end=end[1],
            weighted_n_node_samples=wnns+1,
        ):
            return -1

        return 0

    cdef int _node_reset_child_splitter(
            self,
            Splitter child_splitter,
            const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            SIZE_t start,
            SIZE_t end,
            DOUBLE_t* weighted_n_node_samples,
    ) nogil except -1:
        """Substitutes splitter.node_reset() setting child splitter on 2D data.
        """
        # TODO: It is done in Splitter2D.init(). Should we do it here?
        # child_splitter.weighted_n_samples = self.weighted_n_samples
        child_splitter.y = y
        child_splitter.sample_weight = sample_weight
        return child_splitter.node_reset(start, end, weighted_n_node_samples)

    cdef void node_value(self, double* dest) nogil:
        """Copy the value (prototype) of node samples into dest."""
        # It should be the same as splitter_cols.node_values().
        self.splitter_rows.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""
        # It should be the same as splitter_cols.node_impurity().
        return self.splitter_rows.node_impurity()

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ) nogil:
        if axis == 1:
            self.splitter_cols.criterion.children_impurity(
                impurity_left, impurity_right)
        else:  # axis == 0
            self.splitter_rows.criterion.children_impurity(
                impurity_left, impurity_right)

    cdef double impurity_improvement(
            self, double impurity_parent, double
            impurity_left, double impurity_right,
            SIZE_t axis,
    ) nogil:
        if axis == 1:
            return self.splitter_cols.criterion.impurity_improvement(
                impurity_parent, impurity_left, impurity_right)
        else:  # axis == 0
            return self.splitter_rows.criterion.impurity_improvement(
                impurity_parent, impurity_left, impurity_right)


cdef class MSE_Wrapper2D(RegressionCriterionWrapper2D):
    cdef double node_impurity(self) nogil:
        # Copied from sklearn.tree._criterion.MSE.node_impurity()
        """Evaluate the impurity of the current node.

        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef double[::1] sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ) nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Is done here because sq_sum_* of children criterion is messed up, as
        they receive axis means as y.
        """
        cdef SIZE_t[2] end
        cdef DOUBLE_t y_ij

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i, j, q, p, k
        cdef DOUBLE_t weighted_n_left, weighted_n_right
        cdef DOUBLE_t w = 1.0

        cdef SIZE_t pos
        cdef double[::1] sum_left
        cdef double[::1] sum_right

        with gil:
            # FIXME: lots of repetition to avoid Python local
            if axis:
                pos = self.splitter_cols.criterion.pos
                sum_left = self.splitter_cols.criterion.sum_left
                sum_right = self.splitter_cols.criterion.sum_right
                weighted_n_left = self.splitter_cols.criterion.weighted_n_left
                weighted_n_right = self.splitter_cols.criterion.weighted_n_right

            else:
                pos = self.splitter_rows.criterion.pos
                sum_left = self.splitter_rows.criterion.sum_left
                sum_right = self.splitter_rows.criterion.sum_right
                weighted_n_left = self.splitter_rows.criterion.weighted_n_left
                weighted_n_right = self.splitter_rows.criterion.weighted_n_right

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
