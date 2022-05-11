from sklearn.tree._criterion cimport Criterion, RegressionCriterion
from sklearn.tree._criterion import MSE
from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters

# cdef class WeightedMSE(RegressionCriterion, MSE):
#     cdef DOUBLE_t* output_weights
#     cdef void set_output_weights(self, DOUBLE_t output_weigths) nogil

cdef class SemisupervisedCriterion(Criterion):
    pass

cdef class SSCompositeCriterion(SemisupervisedCriterion):
    cdef Criterion supervised_criterion
    cdef Criterion unsupervised_criterion
    cdef const DOUBLE_t[:, ::1] X
    cdef SIZE_t n_features
    cdef double supervision

cdef class SSMSE(SSCompositeCriterion):
    pass