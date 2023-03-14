# cython: boundscheck=False

# Author: Pedro Ilídio <pedrilidio@gmail.com>
# Modified from scikit-learn.
#
# License: BSD 3 clause

from libc.string cimport memset, memcpy
from libc.stdint cimport SIZE_MAX
cimport numpy as cnp
import numpy as np
from sklearn.tree._utils cimport log

cdef double NAN = np.nan


cdef class AxisCriterion(Criterion):
    """Criterion that is able to select a subset of columns to consider.

    Criterion objects gather methods to evaluate the impurity of a partition
    and the impurity improvement of a split, each subclass implementing a
    different impurity metric.

    They are held by a Splitter object that provides the y target matrix using
    Criterion.init() (called in Splitter.node_reset()) and directs the
    Criterion using Criterion.update() to specific split positions where the
    children impurities and impurity improvement can be calculated.

    AxisCriterion is analogous to sklearn.tree._criterion.Criterion, but can
    also calculate the impurity on the other axis for the current node and
    children (as if each child partition were transposed), and has an
    additional init_columns() method to receive column indices and column
    weights.

    y's contiguity is also dismissed, since y columns will now be randomly
    acessed.
    """
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    def __cinit__(self, SIZE_t n_outputs, *args, **kwargs):
        self.y = None
        self._columns_are_set = False
        self._cached_pos = SIZE_MAX
        self._cached_rows_impurity_left = 0.0
        self._cached_rows_impurity_right = 0.0
        self._cached_cols_impurity_left = 0.0
        self._cached_cols_impurity_right = 0.0
        # TODO: use cached node impurity
        self._cached_rows_node_impurity = 0.0
        self._cached_cols_node_impurity = 0.0
        # TODO: Possibly remove if Splitter can find split without reordering
        # sample_indices. The upside is that we reduce memory usage, the downside is
        # that we lose C contiguity, which seems important here.
        self._node_col_indices = np.empty(n_outputs, dtype=np.intp, order='C')

    cdef int init(
        self,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] sample_indices,
        SIZE_t start,
        SIZE_t end,
    ) except -1 nogil:
        with gil:
            raise RuntimeError("Please use axis_init() instead of init()")
    
    cdef void init_columns(
        self,
        const DOUBLE_t[:] col_weights,
        double weighted_n_cols,
        SIZE_t[:] col_indices,
        SIZE_t start_col,
        SIZE_t end_col,
    ) nogil:
        """Initialize the column sample indices and column weights.
        Parameters
        ----------
        col_weights : ndarray, dtype=DOUBLE_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_cols : double
            The total column weight of the y partition being considered
        col_indices : ndarray, dtype=SIZE_t
            A mask on the sample_indices. Indices of the sample_indices in X and y we want to use,
            where sample_indices[start:end] correspond to the sample_indices in this node.
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node
        """
        self.col_indices = col_indices
        self.col_weights = col_weights
        self.start_col = start_col
        self.end_col = end_col

        self.n_node_cols = end_col - start_col
        self.weighted_n_cols = weighted_n_cols
        self.weighted_n_node_cols = 0.0

        cdef SIZE_t j, q
        cdef DOUBLE_t w = 1.0

        for q in range(start_col, end_col):
            j = col_indices[q]
            self._node_col_indices[q - start_col] = j

            if col_weights is not None:
                w = col_weights[j]
            self.weighted_n_node_cols += w

        self._columns_are_set = True
 
    cdef int axis_init(
        self,
        const DOUBLE_t[:, :] y,
        const DOUBLE_t[:] sample_weight,
        const DOUBLE_t[:] col_weights,
        const SIZE_t[:] sample_indices,
        const SIZE_t[:] col_indices,
        double weighted_n_samples,
        double weighted_n_cols,
        SIZE_t start,
        SIZE_t end,
        SIZE_t start_col,
        SIZE_t end_col,
    ) except -1 nogil:
        """Placeholder for a method which will initialize the criterion.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        y : ndarray, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
            stored as a Cython memoryview.
        sample_weight : ndarray, dtype=DOUBLE_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : double
            The total weight of the sample_indices being considered
        sample_indices : ndarray, dtype=SIZE_t
            A mask on the sample_indices. Indices of the sample_indices in X and y we want to use,
            where sample_indices[start:end] correspond to the sample_indices in this node.
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node
        """
        pass
   
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        cdef:
            SIZE_t k
            double rows_impurity
            double cols_impurity

        self.node_axes_impurities(&rows_impurity, &cols_impurity)

        return 0.5 * (rows_impurity + cols_impurity)

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right
    ) nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        """
        cdef:
            double rows_impurity_left
            double rows_impurity_right
            double cols_impurity_left
            double cols_impurity_right

        self.children_axes_impurities(
            &rows_impurity_left,
            &rows_impurity_right,
            &cols_impurity_left,
            &cols_impurity_right,
        )

        impurity_left[0] = 0.5 * (rows_impurity_left + cols_impurity_left)
        impurity_right[0] = 0.5 * (rows_impurity_right + cols_impurity_right)
 
    cdef void node_axes_impurities(
        self,
        double* rows_impurity,
        double* cols_impurity,
    ) nogil:
        """Evaluate the impurity of the current node in both directions.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """

    cdef void children_axes_impurities(
        self,
        double* rows_impurity_left,
        double* rows_impurity_right,
        double* cols_impurity_left,
        double* cols_impurity_right,
    ) nogil:
        """Evaluate the impurity in children nodes, in both directions.
        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        Should employ self._cached_* attributes to minimize computation.
        """

    cdef double proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef:
            double rows_impurity_left
            double rows_impurity_right
            double cols_impurity_left   # Discarded
            double cols_impurity_right  # Discarded

        self.children_axes_impurities(
            &rows_impurity_left,
            &rows_impurity_right,
            &cols_impurity_left,
            &cols_impurity_right,
        )

        return (- self.weighted_n_right * rows_impurity_right
                - self.weighted_n_left * rows_impurity_left)

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right,
    ) nogil:
        """Compute the improvement in impurity.

        Differently from the usual sklearn objects, the impurity improvement
        is NOT:
            (imp_parent - imp_children) / imp_parent
        but instead:
            0.5 * (
                (rows_imp_parent - rows_imp_children) / rows_imp_parent
                +
                (cols_imp_parent - cols_imp_children) / cols_imp_parent
            )
        In many cases, however, columns impurity improvement is zero, and we
        return only the impurity over rows.
        """
        # FIXME: Since we cannot access rows_impurity and cols_impurity
        # separately, we have to discard this method's input and calculate
        # them again. To mitigate this problem, self.children_impurity()
        # caches the previous values it calculated and reuses them if
        # self.pos == self._cached_pos.
        # NOTE: self._cached_pos must be reset at AxisCriterion.init() to
        # ensure it always corresponds to the current tree node.
        cdef:
            double rows_impurity_parent
            double rows_impurity_left
            double rows_impurity_right

            # To be discarded:
            double cols_impurity_parent
            double cols_impurity_left
            double cols_impurity_right

        self.node_axes_impurities(
            &rows_impurity_parent,
            &cols_impurity_parent,
        )
        self.children_axes_impurities(
            &rows_impurity_left,
            &rows_impurity_right,
            &cols_impurity_left,
            &cols_impurity_right,
        )

        # NOTE: Columns impurity improvement should be zero.
        return (
            (self.weighted_n_node_samples / self.weighted_n_samples)
            * (self.weighted_n_node_cols / self.weighted_n_cols)
            * (
                rows_impurity_parent
                - rows_impurity_right
                    * (self.weighted_n_right / self.weighted_n_node_samples)
                - rows_impurity_left
                    * (self.weighted_n_left / self.weighted_n_node_samples)
            )
        )

    cdef void node_value(self, double* dest) nogil:
        pass

    cdef void total_node_value(self, double* dest) nogil:
        pass


# FIXME: do not multiply by column wights on update, only in impurities
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
            The total number of rows to fit on
        """
        # Default values
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

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    cdef int axis_init(
        self,
        const DOUBLE_t[:, :] y,
        const DOUBLE_t[:] sample_weight,
        const DOUBLE_t[:] col_weights,
        const SIZE_t[:] sample_indices,
        const SIZE_t[:] col_indices,
        double weighted_n_samples,
        double weighted_n_cols,
        SIZE_t start,
        SIZE_t end,
        SIZE_t start_col,
        SIZE_t end_col,
    ) except -1 nogil:
        """Initialize the criterion.
        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        """
        self._cached_pos = SIZE_MAX  # Important to reset cached values.
        self.init_columns(
            col_weights=col_weights,
            weighted_n_cols=weighted_n_cols,
            col_indices=col_indices,
            start_col=start_col,
            end_col=end_col,
        )
        # Initialize fields
        self.y_ = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples

        cdef:
            SIZE_t[::1] _node_col_indices = self._node_col_indices
            SIZE_t n_node_cols = self.n_node_cols
            SIZE_t i, j, p, k
            DOUBLE_t y_ij
            DOUBLE_t wi = 1.0, wj = 1.0
            DOUBLE_t row_sum

        self.sq_sum_total = 0.0
        self.sum_sq_row_sums = 0.0
        self.weighted_n_node_samples = 0.0
        memset(&self.sum_total[0], 0, self.n_node_cols * sizeof(double))

        for p in range(start, end):
            i = sample_indices[p]
            if sample_weight is not None:
                wi = sample_weight[i]

            self.weighted_n_node_samples += wi
            row_sum = 0.0

            for k in range(n_node_cols):
                j = _node_col_indices[k]
                if col_weights is not None:
                    wj = col_weights[j]

                y_ij = y[i, j]

                row_sum += wj * y_ij
                self.sum_total[k] += wi * y_ij
                self.sq_sum_total += wi * wj * y_ij * y_ij

            self.sum_sq_row_sums += wi * row_sum * row_sum

        # Reset to pos=start
        self.reset()
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
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""
        cdef:
            const DOUBLE_t[:] sample_weight = self.sample_weight
            const SIZE_t[:] sample_indices = self.sample_indices
            SIZE_t pos = self.pos
            SIZE_t end = self.end
            SIZE_t n_node_cols = self.n_node_cols

            SIZE_t[::1] _node_col_indices = self._node_col_indices

            SIZE_t i, p, k
            DOUBLE_t wi = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]
                if sample_weight is not None:
                    wi = sample_weight[i]

                for k in range(n_node_cols):
                    self.sum_left[k] += wi * self.y_[i, _node_col_indices[k]]

                self.weighted_n_left += wi
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]
                if sample_weight is not None:
                    wi = sample_weight[i]

                for k in range(n_node_cols):
                    self.sum_left[k] -= wi * self.y_[i, _node_col_indices[k]]

                self.weighted_n_left -= wi

        self.weighted_n_right = (
            self.weighted_n_node_samples - self.weighted_n_left
        )

        for k in range(n_node_cols):
            self.sum_right[k] = self.sum_total[k] - self.sum_left[k]

        self.pos = new_pos
        return 0

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of sample_indices[start:end] into dest."""
        cdef SIZE_t j, k

        for j in range(self.n_outputs):
            dest[j] = NAN

        for k in range(self.n_node_cols):
            j = self._node_col_indices[k]
            dest[j] = self.sum_total[k] / self.weighted_n_node_samples

    cdef void total_node_value(self, double* dest) nogil:
        """Compute the node value of sample_indices[start:end] into dest."""
        cdef SIZE_t k
        dest[0] = 0.0
        for k in range(self.n_node_cols):
            dest[0] += self.sum_total[k]
        dest[0] /= (self.weighted_n_node_samples * self.weighted_n_node_cols)


cdef class AxisMSE(AxisRegressionCriterion):
    """Mean squared error impurity criterion.
        MSE = var_left + var_right
    """

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

        The improvement on the columns axis is always 0, so we disregard it.
        """
        cdef:
            SIZE_t k
            double proxy_impurity_left = 0.0
            double proxy_impurity_right = 0.0
            double wj = 1.0

        for k in range(self.n_node_cols):
            if self.col_weights is not None:
                wj = self.col_weights[self._node_col_indices[k]]

            proxy_impurity_left += wj * self.sum_left[k] * self.sum_left[k]
            proxy_impurity_right += wj * self.sum_right[k] * self.sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void node_axes_impurities(
        self,
        double* rows_impurity,
        double* cols_impurity,
    ) nogil:
        """Evaluate the impurity of the current node in both directions.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        # TODO: cache output as we do with self.axes_children_impurities()?
        cdef:
            SIZE_t k
            double wj = 1.0

        rows_impurity[0] = self.sq_sum_total / self.weighted_n_node_samples
        cols_impurity[0] = self.sq_sum_total / self.weighted_n_node_cols

        for k in range(self.n_node_cols):
            if self.col_weights is not None:
                wj = self.col_weights[self._node_col_indices[k]]

            rows_impurity[0] -= (
                wj * (self.sum_total[k] / self.weighted_n_node_samples) ** 2.0
            )

        cols_impurity[0] -= (
            self.sum_sq_row_sums / self.weighted_n_node_cols ** 2.0
        )

        rows_impurity[0] /= self.weighted_n_node_cols
        cols_impurity[0] /= self.weighted_n_node_samples

    cdef void children_axes_impurities(
        self,
        double* rows_impurity_left,
        double* rows_impurity_right,
        double* cols_impurity_left,
        double* cols_impurity_right,
    ) nogil:
        """Evaluate the impurity in children nodes, in both directions.
        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        """
        if self.pos == self._cached_pos:
            rows_impurity_left[0] = self._cached_rows_impurity_left
            rows_impurity_right[0] = self._cached_rows_impurity_right
            cols_impurity_left[0] = self._cached_cols_impurity_left
            cols_impurity_right[0] = self._cached_cols_impurity_right
            return

        cdef:
            const DOUBLE_t[:] sample_weight = self.sample_weight
            const SIZE_t[:] sample_indices = self.sample_indices
            SIZE_t pos = self.pos
            SIZE_t start = self.start

            const DOUBLE_t[:] col_weights = self.col_weights
            SIZE_t[::1] _node_col_indices = self._node_col_indices
            SIZE_t start_col = self.start_col
            SIZE_t end_col = self.end_col
            SIZE_t n_node_cols = self.n_node_cols

            DOUBLE_t y_ij

            double sq_sum_left = 0.0
            double sq_sum_right
            double sum_sq_row_sums_left = 0.0
            double row_sum
            double sum_sq_col_sums_left = 0.0
            double sum_sq_col_sums_right = 0.0

            SIZE_t i, j, p, k
            DOUBLE_t wi = 1.0, wj = 1.0

        for p in range(start, pos):
            i = sample_indices[p]
            if sample_weight is not None:
                wi = sample_weight[i]

            row_sum = 0.0

            for k in range(n_node_cols):
                j = _node_col_indices[k]
                if col_weights is not None:
                    wj = col_weights[j]

                y_ij = self.y_[i, j]
                sq_sum_left += wi * wj * y_ij * y_ij
                row_sum += wj * y_ij
            
            sum_sq_row_sums_left += wi * row_sum * row_sum

        sq_sum_right = self.sq_sum_total - sq_sum_left
        sum_sq_row_sums_right = self.sum_sq_row_sums - sum_sq_row_sums_left

        rows_impurity_left[0] = sq_sum_left / self.weighted_n_left
        rows_impurity_right[0] = sq_sum_right / self.weighted_n_right
        cols_impurity_left[0] = sq_sum_left / self.weighted_n_node_cols
        cols_impurity_right[0] = sq_sum_right / self.weighted_n_node_cols

        for k in range(self.n_node_cols):
            if col_weights is not None:
                wj = col_weights[_node_col_indices[k]]
            sum_sq_col_sums_left += wj * self.sum_left[k] * self.sum_left[k]
            sum_sq_col_sums_right += wj * self.sum_right[k] * self.sum_right[k]
        
        rows_impurity_left[0] -= (
            sum_sq_col_sums_left / self.weighted_n_left ** 2.0
        )
        rows_impurity_right[0] -= (
            sum_sq_col_sums_right / self.weighted_n_right ** 2.0
        )
        cols_impurity_left[0] -= (
            sum_sq_row_sums_left / self.weighted_n_node_cols ** 2.0
        )
        cols_impurity_right[0] -= (
            sum_sq_row_sums_right / self.weighted_n_node_cols ** 2.0
        )

        rows_impurity_left[0] /= self.weighted_n_node_cols
        rows_impurity_right[0] /= self.weighted_n_node_cols
        cols_impurity_left[0] /= self.weighted_n_left
        cols_impurity_right[0] /= self.weighted_n_right

        self._cached_pos = self.pos
        self._cached_rows_impurity_left = rows_impurity_left[0]
        self._cached_rows_impurity_right = rows_impurity_right[0]
        self._cached_cols_impurity_left = cols_impurity_left[0]
        self._cached_cols_impurity_right = cols_impurity_right[0]


cdef class AxisFriedmanMSE(AxisMSE):
    """Mean squared error impurity criterion with improvement score by Friedman.
    Uses the formula (35) in Friedman's original Gradient Boosting paper:
        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """

    cdef double proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef:
            double total_sum_left = 0.0
            double total_sum_right = 0.0

            SIZE_t k
            double diff = 0.0
            double wj = 1.0

        for k in range(self.n_node_cols):
            if self.col_weights is not None:
                wj = self.col_weights[self._node_col_indices[k]]

            total_sum_left += wj * self.sum_left[k]
            total_sum_right += wj * self.sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right)

        return diff * diff / (self.weighted_n_left * self.weighted_n_right)

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right
    ) noexcept nogil:
        # Note: none of the arguments are used here
        cdef:
            double total_sum_left = 0.0
            double total_sum_right = 0.0
            double diff = 0.0
            double wj = 1.0
            SIZE_t k

        for k in range(self.n_node_cols):
            if self.col_weights is not None:
                wj = self.col_weights[self._node_col_indices[k]]

            total_sum_left += wj * self.sum_left[k]
            total_sum_right += wj * self.sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right) / self.weighted_n_node_cols

        return (diff * diff / (self.weighted_n_left * self.weighted_n_right *
                               self.weighted_n_node_samples))


cdef class AxisClassificationCriterion(AxisCriterion):
    """Abstract criterion for classification."""

    def __cinit__(
        self,
        SIZE_t n_outputs,
        cnp.ndarray[SIZE_t, ndim=1] n_classes,
    ):
        """Initialize attributes for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=SIZE_t
            The number of unique classes in each target
        """
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.n_classes = np.empty(n_outputs, dtype=np.intp)

        cdef SIZE_t k = 0
        cdef SIZE_t max_n_classes = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > max_n_classes:
                max_n_classes = n_classes[k]

        self.max_n_classes = max_n_classes

        # Count labels for each output
        self.sum_total = np.zeros((n_outputs, max_n_classes), dtype=np.float64)
        self.sum_left = np.zeros((n_outputs, max_n_classes), dtype=np.float64)
        self.sum_right = np.zeros((n_outputs, max_n_classes), dtype=np.float64)

    def __reduce__(self):
        return (type(self),
                (self.n_outputs, np.asarray(self.n_classes)), self.__getstate__())

    cdef int axis_init(
        self,
        const DOUBLE_t[:, :] y,
        const DOUBLE_t[:] sample_weight,
        const DOUBLE_t[:] col_weights,
        const SIZE_t[:] sample_indices,
        const SIZE_t[:] col_indices,
        double weighted_n_samples,
        double weighted_n_cols,
        SIZE_t start,
        SIZE_t end,
        SIZE_t start_col,
        SIZE_t end_col,
    ) except -1 nogil:
        """Initialize the criterion.
        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        y : ndarray, dtype=DOUBLE_t
            The target stored as a buffer for memory efficiency.
        sample_weight : ndarray, dtype=DOUBLE_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : double
            The total weight of all sample_indices
        sample_indices : ndarray, dtype=SIZE_t
            A mask on the sample_indices. Indices of the sample_indices in X and y we want to use,
            where sample_indices[start:end] correspond to the sample_indices in this node.
        start : SIZE_t
            The first sample to use in the mask
        end : SIZE_t
            The last sample to use in the mask
        """
        self._cached_pos = SIZE_MAX  # Important to reset cached values.
        self.init_columns(
            col_weights=col_weights,
            weighted_n_cols=weighted_n_cols,
            col_indices=col_indices,
            start_col=start_col,
            end_col=end_col,
        )
        self.y_ = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef:
            SIZE_t[::1] _node_col_indices = self._node_col_indices
            SIZE_t n_node_cols = self.n_node_cols
            SIZE_t i, p, k, c, n_classes_k
            DOUBLE_t wi = 1.0

        for k in range(n_node_cols):
            # TODO: in init_columns, store n_classes contiguously like we do
            # with _node_col_indices.
            n_classes_k = self.n_classes[_node_col_indices[k]]
            memset(&self.sum_total[k, 0], 0, n_classes_k * sizeof(double))

        for p in range(start, end):
            i = sample_indices[p]
            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0.
            if sample_weight is not None:
                wi = sample_weight[i]

            self.weighted_n_node_samples += wi

            for k in range(n_node_cols):
                # Count weighted class frequency for each target
                c = <SIZE_t> self.y_[i, _node_col_indices[k]]
                self.sum_total[k, c] += wi

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef SIZE_t k, n_classes_k
        self.pos = self.start
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        for k in range(self.n_node_cols):
            n_classes_k = self.n_classes[self._node_col_indices[k]]
            memset(&self.sum_left[k, 0], 0, n_classes_k * sizeof(double))
            memcpy(&self.sum_right[k, 0], &self.sum_total[k, 0], n_classes_k * sizeof(double))
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef SIZE_t k, n_classes_k
        self.pos = self.end
        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0

        for k in range(self.n_node_cols):
            n_classes_k = self.n_classes[self._node_col_indices[k]]
            memset(&self.sum_right[k, 0], 0, n_classes_k * sizeof(double))
            memcpy(&self.sum_left[k, 0],  &self.sum_total[k, 0], n_classes_k * sizeof(double))
        return 0

    cdef int update(self, SIZE_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        new_pos : SIZE_t
            The new ending position for which to move sample_indices from the right
            child to the left child.
        """
        cdef:
            SIZE_t pos = self.pos
            SIZE_t end = self.end
            const SIZE_t[:] sample_indices = self.sample_indices
            const DOUBLE_t[:] sample_weight = self.sample_weight
            const SIZE_t[:] _node_col_indices = self._node_col_indices
            SIZE_t i, p, k, c
            DOUBLE_t wi = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]
                if sample_weight is not None:
                    wi = sample_weight[i]

                self.weighted_n_left += wi

                for k in range(self.n_node_cols):
                    self.sum_left[k, <SIZE_t> self.y_[i, _node_col_indices[k]]] += wi

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]
                if sample_weight is not None:
                    wi = sample_weight[i]

                self.weighted_n_left -= wi

                for k in range(self.n_node_cols):
                    self.sum_left[k, <SIZE_t> self.y_[i, _node_col_indices[k]]] -= wi

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_node_cols):
            for c in range(self.n_classes[_node_col_indices[k]]):
                self.sum_right[k, c] = self.sum_total[k, c] - self.sum_left[k, c]

        self.pos = new_pos
        return 0

    cdef void node_value(self, double* dest) noexcept nogil:
        """Compute the node value of sample_indices[start:end] and save it into dest.
        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        # FIXME: use missing classes?
        cdef SIZE_t j, k, n_classes_k#, n_missing_classes
        # cdef double[:, :self.max_n_classes:1] dest_  # TRY

        for j in range(self.n_outputs * self.max_n_classes):
            dest[j] = NAN

        for k in range(self.n_node_cols):
            j = self._node_col_indices[k]
            n_classes_k = self.n_classes[j]
            memcpy(
                &dest[j * self.max_n_classes],
                &self.sum_total[k, 0],
                n_classes_k * sizeof(double),
            )
            # n_missing_classes = self.max_n_classes - n_classes
            # memset(&dest[j, n_classes], 0, n_missing_classes * sizeof(double))

    cdef void total_node_value(self, double* dest) noexcept nogil:
        """Compute the node value of sample_indices[start:end] and save it into dest.
        Parameters
        ----------
        dest : double pointer
            The memory address which we will save the node value into.
        """
        cdef SIZE_t j, k, n_classes_k, class_index
        memset(dest, 0, self.max_n_classes * sizeof(double))

        for k in range(self.n_node_cols):
            j = self._node_col_indices[k]
            n_classes_k = self.n_classes[j]

            for class_index in range(n_classes_k):
                dest[class_index] += self.sum_total[k, class_index]

cdef class AxisEntropy(AxisClassificationCriterion):
    r"""Cross Entropy impurity criterion.
    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let
        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)
    be the proportion of class k observations in node m.
    The cross-entropy is then defined as
        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """

    cdef void node_axes_impurities(
        self,
        double* rows_impurity,
        double* cols_impurity,
    ) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the cross-entropy criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        cdef:
            double entropy = 0.0
            double count_k
            double wj = 1.0
            SIZE_t k, c, j

        for k in range(self.n_node_cols):
            j = self._node_col_indices[k]

            if self.col_weights is not None:
                wj = self.col_weights[j]

            for c in range(self.n_classes[j]):
                count_k = self.sum_total[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k) * wj

        rows_impurity[0] = entropy / self.weighted_n_node_cols
        cols_impurity[0] = self.weighted_n_node_cols / self.weighted_n_cols  # FIXME

    cdef void children_axes_impurities(
        self,
        double* rows_impurity_left,
        double* rows_impurity_right,
        double* cols_impurity_left,
        double* cols_impurity_right,
    ) noexcept nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node
        impurity_right : double pointer
            The memory address to save the impurity of the right node
        """
        cdef:
            double entropy_left = 0.0
            double entropy_right = 0.0
            double wj = 1.0
            double count_k
            SIZE_t k, c, j

        for k in range(self.n_node_cols):
            j = self._node_col_indices[k]

            if self.col_weights is not None:
                wj = self.col_weights[j]

            for c in range(self.n_classes[j]):
                count_k = self.sum_left[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= wj * count_k * log(count_k)

                count_k = self.sum_right[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= wj * count_k * log(count_k)

        rows_impurity_left[0] = entropy_left / self.weighted_n_node_cols
        rows_impurity_right[0] = entropy_right / self.weighted_n_node_cols
        cols_impurity_left[0] = self.weighted_n_node_cols / self.weighted_n_cols  # FIXME
        cols_impurity_right[0] = self.weighted_n_node_cols / self.weighted_n_cols  # FIXME


cdef class AxisGini(AxisClassificationCriterion):
    r"""Gini Index impurity criterion.
    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let
        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)
    be the proportion of class k observations in node m.
    The Gini Index is then defined as:
        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """

    cdef void node_axes_impurities(
        self,
        double* rows_impurity, 
        double* cols_impurity, 
    ) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the Gini criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        cdef:
            double gini = 0.0
            double wj = 1.0
            double sq_count
            double count_k
            SIZE_t k, c, j

        for k in range(self.n_node_cols):
            j = self._node_col_indices[k]
            sq_count = 0.0

            if self.col_weights is not None:
                wj = self.col_weights[j]

            for c in range(self.n_classes[j]):
                count_k = self.sum_total[k, c]
                sq_count += count_k * count_k

            gini += wj * sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)

        rows_impurity[0] = 1.0 - gini / self.weighted_n_node_cols
        cols_impurity[0] = self.weighted_n_node_cols / self.weighted_n_cols  # FIXME

    # TODO: cache?
    cdef void children_axes_impurities(
            self,
            double* rows_impurity_left,
            double* rows_impurity_right,
            double* cols_impurity_left,
            double* cols_impurity_right,
        ) noexcept nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]) using the Gini index.
        Parameters
        ----------
        impurity_left : double pointer
            The memory address to save the impurity of the left node to
        impurity_right : double pointer
            The memory address to save the impurity of the right node to
        """
        cdef:
            double gini_left = 0.0
            double gini_right = 0.0
            double sq_count_left
            double sq_count_right
            double count_k
            double wj = 1.0
            SIZE_t k, c, j

        for k in range(self.n_node_cols):
            j = self._node_col_indices[k]
            sq_count_left = 0.0
            sq_count_right = 0.0

            if self.col_weights is not None:
                wj = self.col_weights[j]

            for c in range(self.n_classes[j]):
                count_k = self.sum_left[k, c]
                sq_count_left += count_k * count_k

                count_k = self.sum_right[k, c]
                sq_count_right += count_k * count_k

            gini_left += wj * sq_count_left / (self.weighted_n_left *
                                                self.weighted_n_left)

            gini_right += wj * sq_count_right / (self.weighted_n_right *
                                                  self.weighted_n_right)

        rows_impurity_left[0] = 1.0 - gini_left / self.weighted_n_node_cols
        rows_impurity_right[0] = 1.0 - gini_right / self.weighted_n_node_cols
        # FIXME: it grows till the end when it should not
        cols_impurity_left[0] = self.weighted_n_node_cols / self.weighted_n_cols  # FIXME
        cols_impurity_right[0] = self.weighted_n_node_cols / self.weighted_n_cols  # FIXME

# TODO:
#    cdef other_axis_children_impurity(SIZE_t other_axis_pos) nogil:
#        self.sum_total[k]
