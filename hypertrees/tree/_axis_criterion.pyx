# Author: Pedro Ilídio <pedrilidio@gmail.com>
# Modified from scikit-learn.
#
# License: BSD 3 clause

import numpy as np
from sklearn.tree._criterion import Criterion
from libc.string cimport memset, memcpy

cdef double NAN = np.nan


cdef class AxisCriterion(Criterion):
    """Enables selecting subset of columns to calculate impurities on."""

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        self._columns_are_set = False
        # TODO: avoid using memory
        # TODO: remove if splitter can find split without reordering samples
        self._col_indices = np.empty(n_outputs, dtype=np.intp, order='C')
    
    cdef void init_columns(
        self,
        SIZE_t* col_samples,
        const DOUBLE_t* col_sample_weight,
        SIZE_t col_start,
        SIZE_t col_end,
    ) nogil:
        self.col_samples = col_samples
        self.col_sample_weight = col_sample_weight
        self.col_start = col_start
        self.col_end = col_end

        self.n_node_cols = col_end - col_start
        self.weighted_n_node_cols = 0.0

        cdef SIZE_t j, q
        cdef DOUBLE_t w = 1.0

        for q in range(col_start, col_end):
            j = col_samples[q]
            self._col_indices[q - col_start] = j

            if col_sample_weight != NULL:
                w = col_sample_weight[j]
            self.weighted_n_node_cols += w

        self._columns_are_set = True


cdef class AxisRegressionCriterion(AxisCriterion):
    r"""Abstract regression criterion.
    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::
        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        """Initialize parameters for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted
        n_samples : SIZE_t
            The total number of samples to fit on
        """
        # Default values
        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        self.sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.sum_right = np.zeros(n_outputs, dtype=np.float64)

        self._columns_are_set = False

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    cdef int init(
        self,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t* sample_weight,
        double weighted_n_samples,
        SIZE_t* samples,
        SIZE_t start,
        SIZE_t end,
    ) nogil except -1:
        """Initialize the criterion.
        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].
        """

        if not self._columns_are_set:
            return -1

        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples

        cdef const DOUBLE_t* col_sample_weight = self.col_sample_weight
        cdef SIZE_t[::1] _col_indices = self._col_indices
        cdef SIZE_t col_start = self.col_start
        cdef SIZE_t col_end = self.col_end
        cdef SIZE_t n_node_cols = self.n_node_cols

        cdef SIZE_t i, j, p, k
        cdef DOUBLE_t y_ij
        cdef DOUBLE_t wi = 1.0, wj = 1.0
        cdef DOUBLE_t row_sum

        self.sq_sum_total = 0.0
        self.sq_row_sums = 0.0
        self.weighted_n_node_samples = 0.0
        memset(&self.sum_total[0], 0, self.n_node_cols * sizeof(double))

        for p in range(start, end):
            i = samples[p]
            if sample_weight != NULL:
                wi = sample_weight[i]

            self.weighted_n_node_samples += wi
            row_sum = 0.0

            for k in range(n_node_cols):
                j = _col_indices[k]
                if col_sample_weight != NULL:
                    wj = col_sample_weight[j]

                y_ij = y[i, j]

                row_sum += wj * y_ij
                self.sum_total[k] += wi * y_ij
                self.sq_sum_total += wi * wj * y_ij * y_ij

            self.sq_row_sums += wi * row_sum * row_sum

        # Reset to pos=start
        self.reset()
        self._columns_are_set = False
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_node_cols * sizeof(double)
        memset(&self.sum_left[0], 0, n_bytes)
        memcpy(&self.sum_right[0], &self.sum_total[0], n_bytes)

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_node_cols * sizeof(double)
        memset(&self.sum_right[0], 0, n_bytes)
        memcpy(&self.sum_left[0], &self.sum_total[0], n_bytes)

        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""
        cdef const DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t n_node_cols = self.n_node_cols

        cdef const DOUBLE_t* col_sample_weight = self.col_sample_weight
        cdef SIZE_t[::1] _col_indices = self._col_indices
        cdef SIZE_t col_start = self.col_start
        cdef SIZE_t col_end = self.col_end

        cdef SIZE_t i, j, p, k
        cdef DOUBLE_t y_ij
        cdef DOUBLE_t w_y_ij
        cdef DOUBLE_t wi = 1.0, wj = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]
                if sample_weight != NULL:
                    wi = sample_weight[i]

                for k in range(n_node_cols):
                    j = _col_indices[k]
                    self.sum_left[k] += wi * self.y[i, j]

                self.weighted_n_left += wi
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]
                if sample_weight != NULL:
                    wi = sample_weight[i]

                for k in range(n_node_cols):
                    j = _col_indices[k]
                    self.sum_left[k] -= wi * self.y[i, j]

                self.weighted_n_left -= wi

        self.weighted_n_right = (
            self.weighted_n_node_samples - self.weighted_n_left
        )

        for k in range(n_node_cols):
            self.sum_right[k] = self.sum_total[k] - self.sum_left[k]

        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        cdef SIZE_t j, k

        for j in range(self.n_outputs):
            dest[j] = NAN

        for k in range(self.n_node_cols):
            j = self._col_indices[k]
            dest[j] = self.sum_total[k] / self.weighted_n_node_samples


cdef class AxisMSE(AxisRegressionCriterion):
    """Mean squared error impurity criterion.
        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef SIZE_t k
        cdef double impurity = self.sq_sum_total
        impurity -= 0.5 * self.sq_row_sums / self.weighted_n_node_cols

        for k in range(self.n_node_cols):
            impurity -= 0.5 * self.sum_total[k] ** 2 / self.weighted_n_node_samples

        impurity /= self.weighted_n_node_samples * self.weighted_n_node_cols
        return impurity

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        The MSE proxy is derived from
            sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
            = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2
        Neglecting constant terms, this gives:
            - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
        """
        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        # TODO: Use sq_row_sums in proxy as well?
        for k in range(self.n_node_cols):
            proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
            proxy_impurity_right += self.sum_right[k] * self.sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """
        cdef const DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef const DOUBLE_t* col_sample_weight = self.col_sample_weight
        cdef SIZE_t[::1] _col_indices = self._col_indices
        cdef SIZE_t col_start = self.col_start
        cdef SIZE_t col_end = self.col_end
        cdef SIZE_t n_node_cols = self.n_node_cols

        cdef DOUBLE_t y_ij

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right
        cdef double sq_row_sums_left = 0.0
        cdef double row_sum
        cdef double col_sums_sq_left = 0.0
        cdef double col_sums_sq_right = 0.0

        cdef SIZE_t i, j, p, k
        cdef DOUBLE_t wi = 1.0, wj = 1.0

        for p in range(start, pos):
            i = samples[p]
            if sample_weight != NULL:
                wi = sample_weight[i]

            row_sum = 0.0

            for k in range(n_node_cols):
                j = _col_indices[k]
                if col_sample_weight != NULL:
                    wj = col_sample_weight[j]

                y_ij = self.y[i, j]
                sq_sum_left += wi * wj * y_ij * y_ij
                row_sum += wj * y_ij
            
            sq_row_sums_left += wi * row_sum * row_sum

        sq_sum_right = self.sq_sum_total - sq_sum_left
        sq_row_sums_right = self.sq_row_sums - sq_row_sums_left

        impurity_left[0] = sq_sum_left
        impurity_right[0] = sq_sum_right

        for k in range(self.n_node_cols):
            col_sums_sq_left += self.sum_left[k] * self.sum_left[k]
            col_sums_sq_right += self.sum_right[k] * self.sum_right[k]
        
        impurity_left[0] -= col_sums_sq_left / (2.0 * self.weighted_n_left)
        impurity_right[0] -= col_sums_sq_right / (2.0 * self.weighted_n_right)
        
        impurity_left[0] -= 0.5 * sq_row_sums_left / self.weighted_n_node_cols
        impurity_right[0] -= 0.5 * sq_row_sums_right / self.weighted_n_node_cols

        impurity_left[0] /= self.weighted_n_node_cols * self.weighted_n_left
        impurity_right[0] /= self.weighted_n_node_cols * self.weighted_n_right
