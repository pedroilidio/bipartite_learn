from sklearn.tree._criterion cimport Criterion, RegressionCriterion
from sklearn.tree._criterion import MSE
from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters

cdef class SSRegressionCriterion(RegressionCriterion):
    pass

cdef class SSCompositeCriterion(SSRegressionCriterion):
    cdef RegressionCriterion supervised_criterion
    cdef RegressionCriterion unsupervised_criterion
    cdef const DOUBLE_t[:, ::1] X
    cdef SIZE_t n_features
    cdef double supervision

cdef class SSMSE(SSCompositeCriterion):
    pass

cdef class WeightedOutputsRegressionCriterion(RegressionCriterion):
    cdef DOUBLE_t* output_weights
    cdef void set_output_weights(self, DOUBLE_t* output_weights) nogil

cdef class WOMSE(WeightedOutputsRegressionCriterion):
    pass

cdef class SSMSE2(WOMSE):
    cdef double supervision
    cdef void set_supervision(self, double supervision) nogil