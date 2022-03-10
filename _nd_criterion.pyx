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
    def __cinit__(self, list child_criteria):
        # cdef RegressionCriterion2D criterion
        cdef MSE2D criterion
        # for criterion in child_criteria:
        #     if criterion.n_outputs > 1:
        #         raise NotImplementedError(
        #             "Multi-output not implemented. Set n_outputs=1 for all"
        #             "Criterion objects.")

        self.child_criteria = child_criteria
        # Should be same as child_criteria[1].n_outputs
        criterion = self.child_criteria[0]
        self.n_outputs = criterion.n_outputs

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
        ) except -1:  # nogil TODO
        """This function adapts RegressionCriterion.init to 2D data."""
        # NOTE: A problem is sometimes n_outputs is actually treated the
        # number of outputs, but sometimes it is just an alias for y.shape[1].
        # In 1D, they have the same value, but now we have to discern them.

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
        #if self.total_row_sample_weight == NULL:
        ##if self.total_col_samples_weight == NULL:  # same.
        #    self.total_row_sample_weight = <DOUBLE_t*> malloc(n_rows * sizeof(DOUBLE_t))
        #    self.total_col_sample_weight = <DOUBLE_t*> malloc(n_cols * sizeof(DOUBLE_t))
        #    if (self.total_col_sample_weight == NULL or
        #        self.total_col_sample_weight == NULL):
        #        raise MemoryError()

        cdef int rc  # Return code
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t p
        cdef SIZE_t q
        cdef DOUBLE_t y_ij
        cdef DOUBLE_t w_y_ij
        cdef DOUBLE_t w=1.0, wi=1.0, wj=1.0

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

        # self.weighted_n_row_samples = 0.0
        # self.weighted_n_col_samples = 0.0
        weighted_n_row_samples = 0.0
        weighted_n_col_samples = 0.0

        # TODO: malloc/memset instead of np.
        # TODO: Since single output, use fortran contiguous? Does it make any
        # difference?
        # TODO: only zero where you need.
        # TODO: move these and shape to cinit.
        self.y_row_sums = np.zeros((n_rows, self.n_outputs), order='C')
        self.y_col_sums = np.zeros((n_cols, self.n_outputs), order='C')


        # TODO: implement multi-output.
        memset(self.sum_total, 0, self.n_outputs * sizeof(double))

        # TODO: only zero where you need.
        # memset(self.total_row_sample_weight, 0, n_rows * sizeof(DOUBLE_t))
        # memset(self.total_col_sample_weight, 0, n_cols * sizeof(DOUBLE_t))

        with nogil:
            # TODO: I think we can do this loop once per axis.
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

                    # TODO: if NULL sample_weight, you can simplify.
                    # self.total_row_sample_weight[i] += w
                    # self.total_col_sample_weight[j] += w

                    self.sum_total[0] += w_y_ij
                    self.sq_sum_total += w_y_ij * y_ij

                    self.weighted_n_node_samples += w

        # Set weighted axis n_samples.
        if self.row_sample_weight == NULL:
            weighted_n_row_samples = end[0] - start[0]
        else:
            for p in range(start[0], end[0]):
                i = row_samples[p]
                weighted_n_row_samples += row_sample_weight[i]

        if self.col_sample_weight == NULL:
            weighted_n_col_samples = end[1] - start[1]
        else:
            for q in range(start[1], end[1]):
                j = col_samples[q]
                weighted_n_col_samples += col_sample_weight[j]

        cdef MSE2D child0, child1
        child0 = self.child_criteria[0]
        child1 = self.child_criteria[1]

        rc = self._init_child_criterion(
            child0,
            self.y_row_sums,
            self.row_sample_weight,
            self.row_samples,
            self.start[0], self.end[0],
            weighted_n_col_samples,  # pass the opposite n_cols.
        )
        rc += self._init_child_criterion(
            child1,
            self.y_col_sums,
            self.col_sample_weight,
            self.col_samples,
            self.start[1], self.end[1],
            weighted_n_row_samples,
        )
        print(
            'ndcrit.wrapper2d.init:wnrs, wncs, ids',
            weighted_n_row_samples,
            weighted_n_col_samples,
            id(weighted_n_row_samples),
            id(weighted_n_col_samples),
        )
        print(
            'ndcrit.wrapper2d.init:children.wnc, ids',
            self.child_criteria[0].weighted_n_cols,
            self.child_criteria[1].weighted_n_cols,
            id(self.child_criteria[0].weighted_n_cols),
            id(self.child_criteria[1].weighted_n_cols),
        )
        print(
            'ndcrit.wrapper2d.init:children01.wnc, ids',
            child0.weighted_n_cols,
            child1.weighted_n_cols,
            id(child0.weighted_n_cols),
            id(child1.weighted_n_cols),
        )

        if rc:
            rc = -1
        return rc

    cdef int _init_child_criterion(
            self,
            MSE2D child_criterion,
            const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            SIZE_t* samples, SIZE_t start,
            SIZE_t end,
            DOUBLE_t weighted_n_cols,
    ) except -1:  # nogil
        """Initialize the child criterion.

        This initializes the criterion at node samples[start:end] and children
        samples[start:start] and samples[start:end].
        """
        ############# TODO: remove scaffold! ##############
        # cdef SIZE_t i
        # X = np.random.rand(end)
        # y = (X > .7).reshape((-1, 1)).astype(float, order='C')
        # asamples = X.argsort()

        # for i in range(end):
        #     samples[i] = asamples[i]

        # sample_weight = NULL

        # child_criterion.init(
        #     y, sample_weight, self.weighted_n_samples,
        #     samples, start, end)

        # # print("[child y]", child_criterion.y)

        # s = np.empty(end)
        # for i in range(end):
        #     s[i] = samples[i]
        # print(
        #     np.array(y), self.weighted_n_samples,
        #     s, start, end)

        ###################################################
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
        print("[child imp]", child_criterion.node_impurity())
        print("[child sum_left]", child_criterion.sum_left[0])
        print("[child sum_right]", child_criterion.sum_right[0])

        cdef DOUBLE_t wns = 1. * weighted_n_cols
        child_criterion.set_weighted_n_cols(deepcopy(wns))

        print(
            'ndcrit.wrapper2d._init_child:weighted_n_cols, child_crit.wnc, id',
            weighted_n_cols,
            child_criterion.weighted_n_cols,
            id(weighted_n_cols),
            id(child_criterion.weighted_n_cols),
        )
        print(
            'ndcrit.wrapper2d._init_child:child_crit, id',
            id(child_criterion),
        )

        return 0

    cdef void node_value(self, double* dest): # nogil
        """Copy the value (prototype) of node samples into dest."""
        # It should be the same as criterion_cols.node_values().
        cdef RegressionCriterion criterion = self.child_criteria[0]
        criterion.node_value(dest)

    cdef double node_impurity(self): # nogil
        """Return the impurity of the current node."""
        # It should be the same as criterion_cols.node_impurity().
        return self.child_criteria[0].node_impurity()

    cdef void children_impurity(
            self,
            double* impurity_left,
            double* impurity_right,
            SIZE_t axis,
    ):# nogil:
        cdef RegressionCriterion criterion
        criterion = self.child_criteria[axis]
        criterion.children_impurity(impurity_left, impurity_right)

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right,
                                     SIZE_t axis):
        cdef MSE2D criterion
        criterion = self.child_criteria[axis]
        return criterion.impurity_improvement(
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
    ):# nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (samples[start:pos]) and the
        impurity the right child (samples[pos:end]).
        """
        cdef RegressionCriterion criterion
        criterion = self.child_criteria[axis]

        cdef SIZE_t[2] end
        cdef SIZE_t pos = criterion.pos
        end[0], end[1] = self.end[0], self.end[1]
        print('chil_imp2d:intial start,end', self.start[0],
            self.start[1], end[0], end[1])
        end[axis] = pos
        criterion.reset()
        criterion.update(pos)

        cdef double* sum_left = criterion.sum_left
        cdef double* sum_right = criterion.sum_right
        cdef DOUBLE_t y_ij

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t q
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        print('chil_imp2d:start,end',
            self.start[0], self.start[1], end[0], end[1])

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
        print('_nd_crit:chil_imp(self.sq_sum_total, sq_sum_left, sq_sum_right)',
            self.sq_sum_total, sq_sum_left, sq_sum_right)
        print('_nd_crit:chil_imp(weighted_n_left, weighted_n_right)',
            criterion.weighted_n_left, criterion.weighted_n_right)
        print('_nd_crit:chil_impsum_total[0], sum_left[0], sum_right[0])',
            self.sum_total[0], sum_left[0], sum_right[0])
        print('_nd_crit:chil_imp:self.n_outputs', self.n_outputs)
        print('_nd_crit:chil_imp:crit.pos', criterion.pos)

        impurity_left[0] = sq_sum_left / criterion.weighted_n_left
        impurity_right[0] = sq_sum_right / criterion.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (sum_left[k] / criterion.weighted_n_left) ** 2.0
            impurity_right[0] -= (sum_right[k] / criterion.weighted_n_right) ** 2.0
        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs
        print('_nd_crit:chil_imp(imp_left, imp_right)',
             impurity_left[0], impurity_right[0])


#cdef class RegressionCriterion2D(RegressionCriterion):
cdef class MSE2D(RegressionCriterion):
    def __init__(self, SIZE_t n_outputs, SIZE_t n_samples):
        self.weighted_n_cols = 0.0
        self.testeee = 'testeee'

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef double* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t wnc = self.weighted_n_cols

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
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] += w * self.y[i, k]

                self.weighted_n_left += wnc * w  # Only change.
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] -= w * self.y[i, k]

                self.weighted_n_left -= wnc * w  # Only change.

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos
        return 0

    def set_weighted_n_cols(self, DOUBLE_t wnc):
        self.weighted_n_cols = wnc


# cdef class MSE2D(RegressionCriterion2D):
# # cdef class MSE(RegressionCriterion):  # Exact copy.
#     """Mean squared error impurity criterion.
# 
#         MSE = var_left + var_right
#     """
#     def __cinit__(self, SIZE_t n_outputs, SIZE_t n_samples):
#         self.weighted_n_cols = 0.0

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node.

        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of samples[start:end]. The smaller the impurity the
        better.
        """
        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

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
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        #with gil:
            # print ('proxy', proxy_impurity_left / self.weighted_n_left +
            #         proxy_impurity_right / self.weighted_n_right)
            # print('proxy_left/right', proxy_impurity_left, proxy_impurity_right)
            # print('wn_left/right', self.weighted_n_left, self.weighted_n_right)
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

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
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
            impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs


cdef class teste:
    def __cinit__(self):
        self.flo = 43.
