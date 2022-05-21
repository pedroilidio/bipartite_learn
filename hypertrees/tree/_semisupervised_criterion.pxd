from sklearn.tree._criterion cimport Criterion, RegressionCriterion
from sklearn.tree._criterion import MSE
from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters

cdef class SSRegressionCriterion(Criterion):
    cdef double[::1] sum_total
    cdef double[::1] sum_left
    cdef double[::1] sum_right
    pass

cdef class SSCompositeCriterion(SSRegressionCriterion):
    cdef RegressionCriterion supervised_criterion
    cdef RegressionCriterion unsupervised_criterion
    cdef const DOUBLE_t[:, ::1] X
    cdef SIZE_t n_features
    cdef double supervision

cdef class SSMSE(SSCompositeCriterion):
    pass
