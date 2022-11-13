# cython: boundscheck=False
from sklearn.tree._criterion cimport RegressionCriterion, Criterion
# from sklearn.tree._criterion import MSE
from time import time
from warnings import warn
from libc.stdlib cimport malloc, calloc, free, realloc
from libc.string cimport memset

import numpy as np
cimport numpy as cnp

cdef DOUBLE_t NAN = np.nan

np.import_array()
from copy import deepcopy

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
        self.row_samples = NULL
        self.col_samples = NULL

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
        self.weighted_n_row_samples = 0.0
        self.weighted_n_col_samples = 0.0

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
        self.row_samples = &self.splitter_rows.samples[0]
        self.col_samples = &self.splitter_cols.samples[0]
        self.start[0], self.start[1] = start[0], start[1]
        self.end[0], self.end[1] = end[0], end[1]
        self.sq_sum_total = 0.0

        self.weighted_n_node_samples = 0.0
        self.weighted_n_row_samples = 0.0
        self.weighted_n_col_samples = 0.0

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
            self.weighted_n_row_samples = end[0] - start[0]
        else:
            for p in range(start[0], end[0]):
                i = self.row_samples[p]
                self.weighted_n_row_samples += row_sample_weight[i]

        if self.col_sample_weight == NULL:
            self.weighted_n_col_samples = end[1] - start[1]
        else:
            for q in range(start[1], end[1]):
                j = self.col_samples[q]
                self.weighted_n_col_samples += col_sample_weight[j]

        # Build total_row[col]_sample_weight
        for p in range(start[0], end[0]):
            i = self.row_samples[p]

            if self.row_sample_weight != NULL:
                self.total_row_sample_weight[i] = \
                    self.row_sample_weight[i] * self.weighted_n_col_samples
            else:
                self.total_row_sample_weight[i] = self.weighted_n_col_samples

            # TODO: Multioutput
            self.y_row_sums[i, 0] = \
                self.y_row_sums[i, 0]/self.weighted_n_col_samples

        for q in range(start[1], end[1]):
            j = self.col_samples[q]

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
        self.n_rows = self.splitter_rows.criterion.n_samples
        self.n_cols = self.splitter_cols.criterion.n_samples
        self.row_samples = NULL
        self.col_samples = NULL

        self._aux_len = max(self.n_rows, self.n_cols)
        # Temporary storage to use in node_value()
        self._node_value_aux = <double*> malloc(
            sizeof(double) * self._aux_len
        )
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
        self.weighted_n_row_samples = 0.0
        self.weighted_n_col_samples = 0.0

        self.sum_total = np.zeros(self.n_outputs, dtype=np.float64)

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
        # NOTE: A problem is sometimes n_outputs is actually treated the
        # number of outputs, but sometimes it is just an alias for y.shape[1].
        # In 1D, they have the same value, but now we have to discern them.
        cdef SIZE_t i, j, p, q
        cdef DOUBLE_t wi, wj

        # Initialize fields
        self.y_2D = y_2D
        self.row_sample_weight = row_sample_weight
        self.col_sample_weight = col_sample_weight
        self.weighted_n_samples = weighted_n_samples
        self.row_samples = &self.splitter_rows.samples[0]
        self.col_samples = &self.splitter_cols.samples[0]

        # FIXME: does not work because of depth first-tree building
        # Use last split axis to avoid redundantly calculating node impurity
        # in self.impurity_improvement()
        if self.start[0] == start[0] and self.end[0] == end[0]:
            self.last_split_axis = 1
        elif self.start[1] == start[1] and self.end[1] == end[1]:
            self.last_split_axis = 0
        else:
            self.last_split_axis = -1

        self.start[0], self.start[1] = start[0], start[1]
        self.end[0], self.end[1] = end[0], end[1]
        self.sq_sum_total = 0.0

        # FIXME what to do with this? Get self.weighted_n_samples from it?
        cdef double[2] wnns  # will be discarded
        cdef SIZE_t n_node_rows = end[0] - start[0]
        cdef SIZE_t n_node_cols = end[1] - start[1]

        with gil:
            # FIXME: how to access composite semisupervised criterion?
            #        a gambiarra is used for now (they set n_outputs again).
            # HACK
            self.splitter_rows.criterion.n_outputs = n_node_cols
            self.splitter_cols.criterion.n_outputs = n_node_rows

            # self.splitter_rows.criterion.n_samples = n_node_rows
            # self.splitter_cols.criterion.n_samples = n_node_cols

            self.y_2D_rows = np.empty((self.n_rows, n_node_cols))
            self.y_2D_cols = np.empty((self.n_cols, n_node_rows))

        for p in range(n_node_rows):
            i = self.row_samples[p + start[0]]
            for q in range(n_node_cols):
                j = self.col_samples[q + start[1]]
                self.y_2D_rows[i, q] = self.y_2D_cols[j, p] = self.y_2D[i, j]

        # FIXME: this is actually MSE specific.
        if (self.row_sample_weight!=NULL) or (self.row_sample_weight!=NULL):
            wi = wj = 1.

            for p in range(n_node_rows):
                i = self.row_samples[p + start[0]]

                if row_sample_weight != NULL:
                    wi = row_sample_weight[i]

                for q in range(n_node_cols):
                    j = self.col_samples[q + start[1]]

                    if col_sample_weight != NULL:
                        wj = col_sample_weight[j]

                    self.y_2D_rows[i, q] *= wj ** .5
                    self.y_2D_cols[j, p] *= wi ** .5

        if -1 == self._node_reset_child_splitter(
            child_splitter=self.splitter_rows,
            y=self.y_2D_rows,
            sample_weight=self.row_sample_weight,
            start=start[0],
            end=end[0],
            weighted_n_node_samples=wnns,
        ):
            return -1

        if -1 == self._node_reset_child_splitter(
            child_splitter=self.splitter_cols,
            y=self.y_2D_cols,
            sample_weight=self.col_sample_weight,
            start=start[1],
            end=end[1],
            weighted_n_node_samples=wnns+1,
        ):
            return -1

        # FIXME: do we need self.sum_total?
        self.weighted_n_row_samples = wnns[0]
        self.weighted_n_col_samples = wnns[1]
        self.weighted_n_node_samples = wnns[0] * wnns[1]
        with gil: print('**** wnns', wnns)

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
        cdef int ret
        child_splitter.y = y
        # ### TODO: scaff
        # cdef:
        #     int k, i, p
        #     double sq_sum_total=0., w=1., y_ik, w_y_ik, impurity
        #     double sum_total[200], wnns

        # for k in range(child_splitter.criterion.n_outputs):
        #     sum_total[k] = 0.

        # for p in range(start, end):
        #     i = child_splitter.samples[p]

        #     if sample_weight != NULL:
        #         w = child_splitter.sample_weight[i]

        #     for k in range(child_splitter.criterion.n_outputs):
        #         y_ik = y[i, k]
        #         w_y_ik = w * y_ik
        #         sum_total[k] += w_y_ik
        #         sq_sum_total += w_y_ik * y_ik
        #     
        #     wnns += w

        # impurity = sq_sum_total / wnns
        # for k in range(child_splitter.criterion.n_outputs):
        #     impurity -= (sum_total[k] / wnns)**2.0

        # impurity /= child_splitter.criterion.n_outputs


        # ret = child_splitter.node_reset(start, end, weighted_n_node_samples)
        # with gil:
        #     print('*** sqsumtotal, start, end', sq_sum_total, start, end)
        #     print('*** impuyrity', impurity)
        ####

        return child_splitter.node_reset(start, end, weighted_n_node_samples)

    cdef void node_value(self, double* dest) nogil:
        """Copy the value (prototype) of node samples into dest.
        """
        cdef SIZE_t i, j, p, q

        for i in range(self.n_outputs):
            dest[i] = NAN

        self.splitter_cols.node_value(self._node_value_aux)

        # Copy each row's output to their corresponding positions of dest
        for q in range(self.start[0], self.end[0]):
            j = self.row_samples[q]
            dest[j] = self._node_value_aux[q-self.start[0]]
        
        self.splitter_rows.node_value(self._node_value_aux)

        # Copy each colum's output to their corresponding positions of dest
        for p in range(self.start[1], self.end[1]):
            i = self.col_samples[p]
            dest[i + self.n_rows] = self._node_value_aux[p-self.start[1]]

        # with gil:
        #     print('*** NODEVALUE NDCRIT PBCTCRIT')
        #     print('*** crit.weighted_n_node_samples')
        #     print(self.splitter_rows.criterion.weighted_n_node_samples)
        #     print(self.splitter_cols.criterion.weighted_n_node_samples)
        #     print('*** crit.n_out')
        #     print(self.splitter_rows.criterion.n_outputs)
        #     print(self.splitter_cols.criterion.n_outputs)
        #     print('***', end=' ')
        #     for p in range(self.start[0], self.end[0]):
        #         print(self.row_samples[p], end=' ')
        #     print()
        #     print('***', end=' ')
        #     for p in range(self.start[1], self.end[1]):
        #         print(self.col_samples[p], end=' ')
        #     print()
        #     print('***', end=' ')
        #     for i in range(self.n_rows + self.n_cols):
        #         print(dest[i], end=' ')
        #     print()
        #     # why are they different??
        #     print("*** partition (it's already resorted)")
        #     for i in range(self.start[0], self.end[0]):
        #         p = self.row_samples[i]
        #         for j in range(self.start[1], self.end[1]):
        #             q = self.col_samples[i]
        #             print(self.y_2D[p, q], end=' ')
        #         print()

        #     # why are not the outputs equal to y values when n_samples==1??
        #     # samples are equal?
        #     print("*** y crit rows")
        #     for i in range(self.start[0], self.end[0]):
        #         p = self.row_samples[i]
        #         for j in range(self.splitter_rows.criterion.n_outputs):
        #             print(self.splitter_rows.criterion.y[p, j], end='[')
        #             print(self.splitter_rows.criterion.samples[i], end='] ')
        #             # print(self.splitter_rows.criterion.sum_total[j], end=' ')
        #         print()
        #     print("*** y crit cols")
        #     for i in range(self.start[1], self.end[1]):
        #         p = self.col_samples[i]

        #         for j in range(self.splitter_cols.criterion.n_outputs):
        #             print(self.splitter_cols.criterion.y[p, j], end='[')
        #             print(self.splitter_cols.criterion.samples[i], end='] ')
        #             # print(self.splitter_cols.criterion.sum_total[j], end=' ')
        #         print()

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node.

        In scikit-learn trees it is only used at the root node.
        """
        with gil: print("***** ROOT IMP rows cols", self.splitter_rows.node_impurity(),
            self.splitter_cols.node_impurity())
        
        # Will be replaced by impurity improvement anyway. We define it here
        # just for the sake of semantics.
        return (self.splitter_rows.node_impurity()
                + self.splitter_cols.node_impurity()) / 2

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ):
        # if axis == 0:
        #     self.splitter_rows.criterion.children_impurity(
        #         impurity_left, impurity_right)
        # elif axis == 1:
        #     self.splitter_cols.criterion.children_impurity(
        #         impurity_left, impurity_right)
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
        """The final value to express the split quality. 
        """
        # TODO: An alternative to recalculating node impurity would be to
        #       always get the mean impurity among the two axes, yielding a
        #       symmetric and apparently more consistent and reasonable metric.
        #       However, obtaining the impurity along the axis other than the
        #       used for splitting is not trivial.
        with gil:
            # FIXME: wrong imp improvement (> 1)
            print("\n*** IMPIMP")
            print("*** last_split_axis", self.last_split_axis)
            print("*** weighted n rows/cols",
                self.splitter_rows.criterion.weighted_n_samples,
                self.splitter_cols.criterion.weighted_n_samples,
            )
            print("*** axis imp_parent left right", axis, impurity_parent, impurity_left, impurity_right)
            print("*** axis 0 nout wnleft wnright",
                self.splitter_rows.criterion.n_outputs,
                self.splitter_rows.criterion.weighted_n_left,
                self.splitter_rows.criterion.weighted_n_right,
            )
            print("*** axis 1 nout wnleft, wnright",
                self.splitter_cols.criterion.n_outputs,
                self.splitter_cols.criterion.weighted_n_left,
                self.splitter_cols.criterion.weighted_n_right,
            )
        if axis == 0:
            # FIXME: does not work because of depth first-tree building
            # Since row and col criteria yield different impurity (along rows'
            # or columns' axis), we recompute the node impurity here,
            # differently from what it is originally done (reusing children
            # impurity from the last split as the current's parent impurity).
            if self.last_split_axis != axis:
                impurity_parent = self.splitter_rows.criterion.node_impurity()
            with gil:
                print('*** recalc. impurity_parent', impurity_parent)
            return self.splitter_rows.criterion.impurity_improvement(
                impurity_parent, impurity_left, impurity_right)

        elif axis == 1:
            if self.last_split_axis != axis:
                impurity_parent = self.splitter_cols.criterion.node_impurity()
            with gil:
                print('*** recalc. impurity_parent', impurity_parent)
            return self.splitter_cols.criterion.impurity_improvement(
                impurity_parent, impurity_left, impurity_right)