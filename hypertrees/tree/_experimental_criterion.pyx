# cython: bounds_check=False
from libc.math cimport fabs
from sklearn.tree._criterion cimport RegressionCriterion, DTYPE_t, SIZE_t


cdef double q2(double v, double own_mean, double others_mean) nogil:
    """
    own_mean: mean of the cluster to which v belongs
    others_mean: mean of the cluster to which v does not belong
    """
    cdef double own_diff = fabs(v - own_mean)
    cdef double others_diff = fabs(v - others_mean)
    cdef double maxdiff = own_diff if own_diff > others_diff else others_diff
    return (others_diff - own_diff) / maxdiff


cdef class UD3(RegressionCriterion):
    """UD3 criterion as defined by [1].
    """
    cdef double node_impurity(self) nogil:
        return 0.

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        impurity_left[0] = impurity_right[0] = 0.
    
    cdef double proxy_impurity_improvement(self) nogil:
        cdef:
            double mean_left, mean_right, first, last
            double w_first = 1.
            double w_last = 1.
            double impurity = 0.
            SIZE_t first_index = self.sample_indices[self.start]
            SIZE_t last_index = self.sample_indices[self.end - 1]
            SIZE_t k

        if self.sample_weight is None:
            w_first = self.sample_weight[first_index]
            w_last = self.sample_weight[last_index]

        for k in range(self.n_outputs):
            # IMPORTANT NOTE: we assume y is ordered! It's intended to be the
            #                 X's feature being used for splitting.
            # TODO: Test if actually min and max values.
            # TODO: Ensure getting min / max by using the already present loop
            #       at .init() (take the opportunity also to dismiss *sq_sum.).
            first = self.y[first_index, k] * w_first
            last = self.y[last_index, k] * w_last
            mean_left = self.sum_left[k] / self.weighted_n_left
            mean_right = self.sum_right[k] / self.weighted_n_right
            impurity += fabs(mean_right - mean_left) / (last - first)
        
        return impurity

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        return self.proxy_impurity_improvement() / self.n_outputs


cdef class UD35(RegressionCriterion):
    """UD3.5 criterion as defined by [2].
    
    2. [Loyola-González _et al._](DOI:10.1109/ACCESS.2020.2980581)
    """
    cdef double node_impurity(self) nogil:
        return 0.

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef:
            double mean_left, mean_right
            double first_left, last_left, first_right, last_right
            double _impurity_left = 0., _impurity_right = 0.
            double w_first_left, w_last_left, w_first_right, w_last_right
            SIZE_t first_index_left = self.sample_indices[self.start]
            SIZE_t last_index_left = self.sample_indices[self.pos - 1]
            SIZE_t first_index_right = self.sample_indices[self.pos]
            SIZE_t last_index_right = self.sample_indices[self.end - 1]
            SIZE_t k

        if self.sample_weight is None:
            w_first_left = self.sample_weight[first_index_left]
            w_last_left = self.sample_weight[last_index_left]
            w_first_right = self.sample_weight[first_index_right]
            w_last_right = self.sample_weight[last_index_right]
        else:
            w_first_left = w_last_left = w_first_right = w_last_right = 1.

        for k in range(self.n_outputs):
            # IMPORTANT NOTE: we assume y is ordered! It's intended to be the
            #                 X's feature being used for splitting.
            # TODO: Test if actually min and max values.
            # TODO: Ensure getting min / max by using the already present loop
            #       at .init() (take the opportunity also to dismiss *sq_sum.).
            first_left = self.y[first_index_left, k] * w_first_left
            last_left = self.y[last_index_left, k] * w_last_left
            first_right = self.y[first_index_right, k] * w_first_right
            last_right = self.y[last_index_right, k] * w_last_right

            mean_left = self.sum_left[k] / self.weighted_n_left
            mean_right = self.sum_right[k] / self.weighted_n_right

            # Negative because higher q2 is better.
            _impurity_left -= q2(first_left, mean_left, mean_right)
            _impurity_left -= q2(last_left, mean_left, mean_right)

            _impurity_right -= q2(first_right, mean_right, mean_left)
            _impurity_right -= q2(last_right, mean_right, mean_left)

        impurity_left[0] = _impurity_left / self.n_outputs / 4 + 1
        impurity_right[0] = _impurity_right / self.n_outputs / 4 + 1
