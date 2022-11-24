# cython: boundscheck=True
from sklearn.tree._criterion cimport RegressionCriterion, Criterion
from libc.stdlib cimport malloc, calloc, free, realloc
from libc.string cimport memset
from libc.math cimport sqrt
import numpy as np
cimport numpy as cnp

np.import_array()

cdef DOUBLE_t NAN = np.nan

cdef class CriterionWrapper2D:
    """Abstract base class."""

    cdef int init(
        self, const DOUBLE_t[:, ::1] y_2D,
        DOUBLE_t* row_sample_weight,
        DOUBLE_t* col_sample_weight,
        double weighted_n_samples,
        SIZE_t[2] start, SIZE_t[2] end,
    ) nogil except -1:
        pass

    cdef int _node_reset_child_splitter(
            self,
            Splitter child_splitter,
            const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            SIZE_t start,
            SIZE_t end,
            DOUBLE_t* weighted_n_node_samples,
    ) nogil except -1:
        pass

    cdef void node_value(self, double* dest) nogil:
        pass

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ):
        pass

    cdef double impurity_improvement(
            self, double impurity_parent, double
            impurity_left, double impurity_right,
            SIZE_t axis,
    ) nogil:
        pass


cdef class RegressionCriterionWrapper2D(CriterionWrapper2D):
    def __cinit__(self, Splitter splitter_rows, Splitter splitter_cols):
        self.splitter_rows = splitter_rows
        self.splitter_cols = splitter_cols
        self.n_outputs = self.splitter_rows.criterion.n_outputs
        self.n_rows = self.splitter_rows.criterion.n_samples
        self.n_cols = self.splitter_cols.criterion.n_samples

        # Default values
        self.row_sample_weight = NULL
        self.col_sample_weight = NULL
        self.sq_sum_total = 0.0

        # total_row_sample_weight will correspond, for each row, to the weight
        # of the row times the total weight of all columns (i.e. the sum of all 
        # col_sample_weight's elements). If they were numpy arrays, it would be:
        #
        #       sample_weights * col_sample_weight.sum()
        #
        # NOTE: maybe we should use a [:, ::1] sample_weight matrix instead.
        # TODO: weight sum is stored in Splitter.weighted_n_sample
        self.total_row_sample_weight = \
            <DOUBLE_t*> malloc(self.n_rows * sizeof(DOUBLE_t))
        self.total_col_sample_weight = \
            <DOUBLE_t*> malloc(self.n_cols * sizeof(DOUBLE_t))

        if (self.total_row_sample_weight == NULL or
            self.total_col_sample_weight == NULL):
            raise MemoryError()

        self.start[0] = 0
        self.start[1] = 0
        self.end[0] = 0
        self.end[1] = 0

        # 1D criteria's only.
        # self.pos = 0
        # self.n_node_samples = 0

        self.weighted_n_node_samples = 0.0
        # self.weighted_n_left = 0.0
        # self.weighted_n_right = 0.0
        self.weighted_n_node_rows = 0.0
        self.weighted_n_node_cols = 0.0

        self.sum_total = np.zeros(self.n_outputs, dtype=np.float64)
        # self.sum_left = np.zeros(self.n_outputs, dtype=np.float64)
        # self.sum_right = np.zeros(self.n_outputs, dtype=np.float64)

        self.y_row_sums = np.zeros(
            (self.n_rows, self.n_outputs), dtype=np.float64)
        self.y_col_sums = np.zeros(
            (self.n_cols, self.n_outputs), dtype=np.float64)

    def __dealloc__(self):
        free(self.total_row_sample_weight)
        free(self.total_col_sample_weight)

    def __reduce__(self):
        return (type(self),
                (self.splitter_rows, self.splitter_cols),
                self.__getstate__())

    def __getstate__(self):
        return {}

    cdef int init(
            self, const DOUBLE_t[:, ::1] y_2D,
            DOUBLE_t* row_sample_weight,
            DOUBLE_t* col_sample_weight,
            double weighted_n_samples,
            SIZE_t[2] start, SIZE_t[2] end,
        ) nogil except -1:
        """This function adapts RegressionCriterion.init to 2D data."""
        # NOTE: A problem is sometimes n_outputs is actually treated the
        #       number of outputs, but sometimes it is just an alias for y.shape[1].
        #       In 1D, they have the same value, but now we have to discern them.
        #
        # FIXME: the way we deal with sample weights seems MSE-specific.

        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t p
        cdef SIZE_t q
        cdef DOUBLE_t y_ij
        cdef DOUBLE_t w_y_ij
        cdef DOUBLE_t w, wi, wj

        # Initialize fields
        self.y_2D = y_2D
        self.row_sample_weight = row_sample_weight
        self.col_sample_weight = col_sample_weight
        self.weighted_n_samples = weighted_n_samples
        self.row_samples = self.splitter_rows.samples
        self.col_samples = self.splitter_cols.samples
        self.start[0], self.start[1] = start[0], start[1]
        self.end[0], self.end[1] = end[0], end[1]
        self.sq_sum_total = 0.0

        self.weighted_n_node_samples = 0.0
        self.weighted_n_node_rows = 0.0
        self.weighted_n_node_cols = 0.0

        # TODO: implement multi-output.
        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))

        # Reset y axis means only where needed.
        for p in range(start[0], end[0]):
            i = self.row_samples[p]
            for q in range(start[1], end[1]):
                j = self.col_samples[q]
                self.y_row_sums[i, 0] = 0
                self.y_col_sums[j, 0] = 0


        # Compute y axis means.
        w = wi = wj = 1.0
        for p in range(start[0], end[0]):
            i = self.row_samples[p]

            if row_sample_weight != NULL:
                wi = row_sample_weight[i]

            for q in range(start[1], end[1]):
                j = self.col_samples[q]

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
            self.weighted_n_node_rows = end[0] - start[0]
        else:
            for p in range(start[0], end[0]):
                i = self.row_samples[p]
                self.weighted_n_node_rows += row_sample_weight[i]

        if self.col_sample_weight == NULL:
            self.weighted_n_node_cols = end[1] - start[1]
        else:
            for q in range(start[1], end[1]):
                j = self.col_samples[q]
                self.weighted_n_node_cols += col_sample_weight[j]

        # Build total_row[col]_sample_weight
        for p in range(start[0], end[0]):
            i = self.row_samples[p]

            if self.row_sample_weight != NULL:
                self.total_row_sample_weight[i] = \
                    self.row_sample_weight[i] * self.weighted_n_node_cols
            else:
                self.total_row_sample_weight[i] = self.weighted_n_node_cols

            # TODO: Multioutput
            self.y_row_sums[i, 0] = \
                self.y_row_sums[i, 0]/self.weighted_n_node_cols

        for q in range(start[1], end[1]):
            j = self.col_samples[q]

            if self.col_sample_weight != NULL:
                self.total_col_sample_weight[j] = \
                    self.col_sample_weight[j] * self.weighted_n_node_rows
            else:
                self.total_col_sample_weight[j] = self.weighted_n_node_rows

            # NOTE: divide to multiply after in Criterion.update().
            # Not the most efficient, but it was the only way I saw to keep
            # the sklearn class unmodified.
            # FIXME: the name should be y_col_means.
            # TODO: Multioutput
            self.y_col_sums[j, 0] = \
                self.y_col_sums[j, 0] / self.weighted_n_node_rows

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
        child_splitter.y = y
        child_splitter.sample_weight = sample_weight

        # TODO: it was being done in Splitter2D.init(), but it's now here for
        #       being criterion-specific. However, often there is no need for
        #       setting this on every node, and splitter.init is only called
        #       once, conveniently.
        child_splitter.weighted_n_samples = self.weighted_n_samples

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
    ):
        if axis == 0:
            self.splitter_rows.criterion.children_impurity(
                impurity_left, impurity_right)
        elif axis == 1:
            self.splitter_cols.criterion.children_impurity(
                impurity_left, impurity_right)

    cdef double impurity_improvement(
        self, double impurity_parent, double
        impurity_left, double impurity_right,
        SIZE_t axis,
    ) nogil:
        if axis == 0:
            return self.splitter_rows.criterion.impurity_improvement(
                impurity_parent, impurity_left, impurity_right)
        elif axis == 1:
            return self.splitter_cols.criterion.impurity_improvement(
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
    ): # TODO nogil: it breaks semi-supervised criteria
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).

        Is done here because sq_sum_* of children criterion is messed up, as
        they receive axis means as y.
        """
        cdef DOUBLE_t y_ij

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i, j, q, p, k
        cdef DOUBLE_t w = 1.0

        cdef DOUBLE_t weighted_n_left
        cdef DOUBLE_t weighted_n_right
        cdef RegressionCriterion criterion

        criterion = self._get_criterion(axis)

        cdef double[::1] sum_left = criterion.sum_left
        cdef double[::1] sum_right = criterion.sum_right

        cdef SIZE_t[2] end
        end[0], end[1] = self.end[0], self.end[1]

        sum_left = criterion.sum_left
        sum_right = criterion.sum_right
        weighted_n_left = criterion.weighted_n_left
        weighted_n_right = criterion.weighted_n_right
        end[axis] = criterion.pos

        with nogil:
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

    cdef Criterion _get_criterion(self, SIZE_t axis): 
        if axis == 1:
            return self.splitter_cols.criterion
        if axis == 0:
            return self.splitter_rows.criterion


# TODO: should work for classification as well.
cdef class PBCTCriterionWrapper(CriterionWrapper2D):
    """Applies Predictive Bi-Clustering Trees method.

    See [Pliakos _et al._, 2018](https://doi.org/10.1007/s10994-018-5700-x).
    """

    def __cinit__(self, Splitter splitter_rows, Splitter splitter_cols):
        self.splitter_rows = splitter_rows
        self.splitter_cols = splitter_cols
        self.criterion_rows = splitter_rows.criterion
        self.criterion_cols = splitter_cols.criterion
        self.n_rows = self.criterion_rows.n_samples
        self.n_cols = self.criterion_cols.n_samples
        self.n_outputs = self.n_rows + self.n_cols

        # Default values
        self.row_sample_weight = NULL
        self.col_sample_weight = NULL
        self.sq_sum_total = 0.0

        self.start[0] = 0
        self.start[1] = 0
        self.end[0] = 0
        self.end[1] = 0

        self.weighted_n_node_samples = 0.0
        self.weighted_n_node_rows = 0.0
        self.weighted_n_node_cols = 0.0

    def __dealloc__(self):
        free(self._node_value_aux)

    def __reduce__(self):
        return (type(self),
                (self.splitter_rows, self.splitter_cols),
                self.__getstate__())

    def __getstate__(self):
        return {}

    cdef int init(
            self, const DOUBLE_t[:, ::1] y_2D,
            DOUBLE_t* row_sample_weight,
            DOUBLE_t* col_sample_weight,
            double weighted_n_samples,
            SIZE_t[2] start, SIZE_t[2] end,
        ) nogil except -1:
        """This function adapts RegressionCriterion.init to 2D data."""
        cdef SIZE_t i, j, p, q
        cdef DOUBLE_t wi, wj

        # Initialize fields
        self.y_2D = y_2D
        self.row_sample_weight = row_sample_weight
        self.col_sample_weight = col_sample_weight
        self.weighted_n_samples = weighted_n_samples
        self.row_samples = self.splitter_rows.samples
        self.col_samples = self.splitter_cols.samples

        self.start[0], self.start[1] = start[0], start[1]
        self.end[0], self.end[1] = end[0], end[1]

        cdef double weighted

        self.criterion_rows.set_columns(
            start[1],
            end[1],
            &self.col_samples[0],
            self.col_sample_weight,
        )
        self.splitter_rows.node_reset(
            start[0], end[0], &self.weighted_n_node_rows,
        )

        self.criterion_cols.set_columns(
            start[0],
            end[0],
            &self.row_samples[0],
            self.row_sample_weight,
        )
        self.splitter_cols.node_reset(
            start[1], end[1], &self.weighted_n_node_cols,
        )

        self.weighted_n_node_samples = (
            self.weighted_n_node_rows * self.weighted_n_node_cols
        )
        return 0

    cdef void node_value(self, double* dest) nogil:
        """Copy the value (prototype) of node samples into dest.
        """
        cdef SIZE_t i
        for i in range(self.n_outputs):
            dest[i] = NAN

        self.splitter_cols.node_value(dest)
        self.splitter_rows.node_value(dest + self.n_rows)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node.

        In scikit-learn trees it is only used at the root node.
        """
        # Should yield equal result for splitter_cols
        return self.splitter_rows.node_impurity()

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right,
        SIZE_t axis,
    ):
        if axis == 0:
            self.criterion_rows.children_impurity(impurity_left, impurity_right)
        elif axis == 1:
            self.criterion_cols.children_impurity(impurity_left, impurity_right)
        else:
            raise ValueError(f"axis must be 1 or 0 ({axis} received)")

    cdef double impurity_improvement(
        self, double impurity_parent, double impurity_left, double impurity_right,
        SIZE_t axis,
    ) nogil:
        """The final value to express the split quality. 
        """
        if axis == 0:
            return self.criterion_rows.impurity_improvement(
                impurity_parent,
                impurity_left,
                impurity_right,
            )
        if axis == 1:
            return self.criterion_cols.impurity_improvement(
                impurity_parent,
                impurity_left,
                impurity_right,
            )
        with gil:
            raise ValueError(f"axis must be 1 or 0 ({axis} received)")
    
    cdef AxisRegressionCriterion _get_criterion(self, SIZE_t axis):
        if axis == 0:
            return self.splitter_rows.criterion
        if axis == 1:
            return self.splitter_cols.criterion
        raise ValueError(f"axis must be 1 or 0 ({axis} received)")

    cdef Splitter _get_splitter(self, SIZE_t axis):
        if axis == 0:
            return self.splitter_rows
        if axis == 1:
            return self.splitter_cols
        raise ValueError(f"axis must be 1 or 0 ({axis} received)")


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.
        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (self.sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

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

        for k in range(self.n_outputs):
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
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef DOUBLE_t y_ik

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs


cdef class AxisRegressionCriterion(RegressionCriterion):
    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
        self._columns_are_set = False
    
    cdef void set_columns(
        self,
        SIZE_t col_start,
        SIZE_t col_end,
        SIZE_t* col_samples,
        DOUBLE_t* col_sample_weight,
    ) nogil:
        self.col_start = col_start
        self.col_end = col_end
        self.col_samples = col_samples
        self.col_sample_weight = col_sample_weight
        self.n_node_cols = col_end - col_start
        self.weighted_n_node_cols = 0.

        cdef SIZE_t p
        cdef DOUBLE_t w = 1.0

        for p in range(col_start, col_end):
            if col_sample_weight != NULL:
                w = col_sample_weight[col_samples[p]]
            self.weighted_n_node_cols += w

        self._columns_are_set = True


cdef class GlobalMSE(AxisRegressionCriterion):
    def __init__(self, n_outputs=1, *args, **kwargs):
        if n_outputs != 1:
            raise ValueError(f"{type(self).__name__} only supports n_outputs=1")

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        cdef SIZE_t k, j
        cdef double wnns = (
            self.weighted_n_node_samples * self.weighted_n_node_cols
        )
        dest[0] = self.sum_total[0] / wnns  # self.n_outputs == 1

    # cdef double node_impurity(self) nogil:
    #     with gil:
    #         return MSE.node_impurity(<MSE>self)

    # cdef double proxy_impurity_improvement(self) nogil:
    #     with gil:
    #         return MSE.proxy_impurity_improvement(<MSE>self)

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.
        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef double impurity
        cdef SIZE_t k
        cdef double weighted_n_node_samples

        weighted_n_node_samples = (
            self.weighted_n_node_samples * self.weighted_n_node_cols
        )

        impurity = self.sq_sum_total / weighted_n_node_samples
        # for k in range(self.n_outputs):  # / self.n_outputs  # n_outputs == 1
        impurity -= (self.sum_total[0] / weighted_n_node_samples)**2.0

        return impurity  # / self.n_outputs  # n_outputs == 1

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

        for k in range(self.n_outputs):
            proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
            proxy_impurity_right += self.sum_right[k] * self.sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion.
        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].
        """
        if not self._columns_are_set:
            with gil:
                raise RuntimeError("BipartiteCriterion.set_columns() must be "
                                   "called before Criterion.init()")
        # Initialize fields
        self._original_y = y
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        # cdef int n = y.shape[0]
        # self._proxy_y = <DOUBLE_t[:n, :1:1]> malloc(sizeof(DOUBLE_t)*y.shape[0])
        cdef DOUBLE_t[:, ::1] proxy_y
        with gil:
            proxy_y = np.zeros_like(y, shape=(y.shape[0], 1))  # n_outputs == 1

        cdef DOUBLE_t* col_sample_weight = self.col_sample_weight
        cdef SIZE_t* col_samples = self.col_samples
        cdef SIZE_t col_start = self.col_start
        cdef SIZE_t col_end = self.col_end

        cdef SIZE_t i, j, p, q
        cdef DOUBLE_t y_ij
        cdef DOUBLE_t w_y_ij
        cdef DOUBLE_t wi = 1.0, wj = 1.0
        self.sq_sum_total = 0.0
        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))

        for p in range(start, end):
            i = samples[p]
            if sample_weight != NULL:
                wi = sample_weight[i]

            self.weighted_n_node_samples += wi

            for q in range(col_start, col_end):
                j = col_samples[q]
                if col_sample_weight != NULL:
                    wj = col_sample_weight[j]

                # for k in range(self.n_outputs):  # n_outputs == 1, k == 0
                y_ij = y[i, j]
                proxy_y[i, 0] += wj * y_ij
                w_y_ij = wi * wj * y_ij
                self.sum_total[0] += w_y_ij
                self.sq_sum_total += w_y_ij * y_ij

        self.y = proxy_y

        # Reset to pos=start
        self.reset()
        self._columns_are_set = False
        return 0
        
    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef DOUBLE_t* col_sample_weight = self.col_sample_weight
        cdef SIZE_t* col_samples = self.col_samples
        cdef SIZE_t col_start = self.col_start
        cdef SIZE_t col_end = self.col_end

        cdef DOUBLE_t y_ij
        cdef DOUBLE_t weighted_n_left
        cdef DOUBLE_t weighted_n_right

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i, j, p, q
        cdef DOUBLE_t wi = 1.0, wj = 1.0

        for p in range(start, pos):
            i = samples[p]
            if sample_weight != NULL:
                wi = sample_weight[i]

            for q in range(col_start, col_end):
                j = col_samples[q]
                if col_sample_weight != NULL:
                    wj = col_sample_weight[j]

                y_ij = self._original_y[i, j]
                sq_sum_left += wi * wj * y_ij * y_ij

        sq_sum_right = self.sq_sum_total - sq_sum_left

        weighted_n_left = self.weighted_n_left * self.weighted_n_node_cols
        weighted_n_right = self.weighted_n_right * self.weighted_n_node_cols

        impurity_left[0] = sq_sum_left / weighted_n_left
        impurity_right[0] = sq_sum_right / weighted_n_right

        # for k in range(self.n_outputs):  # n_outputs == 1, k == 0
        impurity_left[0] -= (self.sum_left[0] / weighted_n_left) ** 2.0
        impurity_right[0] -= (self.sum_right[0] / weighted_n_right) ** 2.0

        # impurity_left[0] /= self.n_outputs  # n_outputs == 1
        # impurity_right[0] /= self.n_outputs  # n_outputs == 1


cdef class LocalMSE(AxisRegressionCriterion):
    cdef DOUBLE_t sq_row_sums

    cdef double node_impurity(self) nogil:
        cdef double impurity = self.sq_sum_total

        impurity -= 0.5 * self.sq_row_sums / self.weighted_n_node_cols

        for k in range(self.n_outputs):
            impurity -= 0.5 * self.sum_total[k] ** 2 / self.weighted_n_node_samples

        impurity /= self.weighted_n_node_samples * self.weighted_n_node_cols

        return impurity

    cdef double proxy_impurity_improvement(self) nogil:
        return MSE.proxy_impurity_improvement(<MSE>self)

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        """Initialize the criterion.
        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].
        """
        if not self._columns_are_set:
            with gil:
                raise RuntimeError("BipartiteCriterion.set_columns() must be "
                                   "called before Criterion.init()")
        # Initialize fields
        self._original_y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.
        self.n_outputs = self.n_node_cols

        # self.y[:, :] = y[:, 0]  # FIXME: no need to copy, just allocate
        # cdef int n = y.shape[0]
        # self._proxy_y = <DOUBLE_t[:n, :1:1]> malloc(sizeof(DOUBLE_t)*y.shape[0])
        cdef DOUBLE_t[:, ::1] proxy_y
        with gil:
            proxy_y = np.zeros_like(y, shape=(y.shape[0], self.n_node_cols))
        cdef DOUBLE_t* col_sample_weight = self.col_sample_weight
        cdef SIZE_t* col_samples = self.col_samples
        cdef SIZE_t col_start = self.col_start
        cdef SIZE_t col_end = self.col_end

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t j
        cdef SIZE_t q
        cdef DOUBLE_t proxy_y_ij
        cdef DOUBLE_t y_ij
        cdef DOUBLE_t w_y_ij
        cdef DOUBLE_t wi = 1.0
        cdef DOUBLE_t wj = 1.0
        cdef DOUBLE_t sqrt_wj = 1.0
        cdef DOUBLE_t row_sum

        self.sq_row_sums = 0.0
        self.sq_sum_total = 0.0
        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(double))

        for p in range(start, end):
            i = samples[p]
            if sample_weight != NULL:
                wi = sample_weight[i]

            self.weighted_n_node_samples += wi
            row_sum = 0.0

            for q in range(col_start, col_end):
                j = col_samples[q]
                if col_sample_weight != NULL:
                    wj = col_sample_weight[j]
                    sqrt_wj = sqrt(wj)

                y_ij = y[i, j]
                row_sum += wj * y_ij
                proxy_y_ij = sqrt_wj * y_ij
                proxy_y[i, q - col_start] = proxy_y_ij
                w_y_ij = wi * proxy_y_ij

                self.sum_total[q - col_start] += w_y_ij
                self.sq_sum_total += w_y_ij * proxy_y_ij
            
            self.sq_row_sums += wi * row_sum * row_sum

        self.y = proxy_y

        # Reset to pos=start
        self.reset()
        self._columns_are_set = False
        return 0

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        cdef SIZE_t k, j

        for k in range(self.n_outputs):
            j = self.col_samples[self.col_start + k]
            dest[j] = self.sum_total[k] / self.weighted_n_node_samples

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes.
        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef DOUBLE_t* col_sample_weight = self.col_sample_weight
        cdef SIZE_t* col_samples = self.col_samples
        cdef SIZE_t col_start = self.col_start

        cdef DOUBLE_t proxy_y_ik
        cdef DOUBLE_t row_sum
        cdef DOUBLE_t sq_row_sums_left = 0.0
        cdef DOUBLE_t sq_row_sums_right

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i, k, p, q
        cdef DOUBLE_t wi = 1.0, wk = 1.0, sqrt_wk = 1.0

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                wi = sample_weight[i]

            row_sum = 0.0

            for k in range(self.n_outputs):
                if col_sample_weight != NULL:
                    wk = col_sample_weight[col_samples[k + col_start]]
                    sqrt_wk = sqrt(wk)

                proxy_y_ik = self.y[i, k]
                sq_sum_left += wi * proxy_y_ik * proxy_y_ik
                row_sum += sqrt_wk * proxy_y_ik  # wj * y_ij
            
            sq_row_sums_left += wi * row_sum * row_sum

        sq_sum_right = self.sq_sum_total - sq_sum_left
        sq_row_sums_right = self.sq_row_sums - sq_row_sums_left
        impurity_left[0] = sq_sum_left
        impurity_right[0] = sq_sum_right

        for k in range(self.n_outputs):
            impurity_left[0] -= 0.5 * self.sum_left[k] ** 2.0 / self.weighted_n_left
            impurity_right[0] -= 0.5 * self.sum_right[k] ** 2.0 / self.weighted_n_right
        
        impurity_left[0] -= 0.5 * sq_row_sums_left / self.weighted_n_node_cols
        impurity_right[0] -= 0.5 * sq_row_sums_right / self.weighted_n_node_cols

        impurity_left[0] /= self.weighted_n_node_cols * self.weighted_n_left
        impurity_right[0] /= self.weighted_n_node_cols * self.weighted_n_right
