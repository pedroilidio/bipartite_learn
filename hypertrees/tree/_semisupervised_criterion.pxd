from sklearn.tree._splitter cimport Splitter
from sklearn.tree._criterion cimport Criterion, RegressionCriterion
from sklearn.tree._criterion import MSE
from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters
from ._nd_criterion cimport CriterionWrapper2D, RegressionCriterionWrapper2D


cdef class BaseDenseSplitter(Splitter):
    cdef const DTYPE_t[:, :] X

cdef class SemisupervisedCriterion(Criterion):
    """Base class for semantic purposes and future maintenance.
    """

# cdef class SSRegressionCriterion(SemisupervisedCriterion):
#     """Base class for semantic purposes and future maintenance.
#     """
#     cdef double[::1] sum_total
#     cdef double[::1] sum_left
#     cdef double[::1] sum_right

cdef class SSCompositeCriterion(SemisupervisedCriterion):
    cdef public RegressionCriterion supervised_criterion
    cdef public RegressionCriterion unsupervised_criterion
    cdef const DOUBLE_t[:, ::1] X
    cdef public SIZE_t n_features
    cdef public double supervision
    cdef public double original_supervision
    cdef void unpack_y(self, const DOUBLE_t[:, ::1] y) nogil
    cdef void update_supervision(self) nogil

cdef class SSMSE(SSCompositeCriterion):
    """Semi-supervised composite criterion with only MSE.
    """

cdef class SingleFeatureSSCompositeCriterion(SSCompositeCriterion):
    """Uses only the current feature as unsupervised data.
    """
    cdef public SIZE_t current_feature
    cdef public double current_node_impurity
    cdef const DOUBLE_t[:, ::1] full_X

cdef class BipartiteSemisupervisedCriterion(CriterionWrapper2D):
    cdef RegressionCriterion unsupervised_criterion_rows
    cdef RegressionCriterion unsupervised_criterion_cols
    cdef RegressionCriterionWrapper2D supervised_bipartite_criterion
    cdef SIZE_t n_row_features
    cdef SIZE_t n_col_features
    cdef public double supervision_rows
    cdef public double supervision_cols
    cdef public double _curr_supervision_rows
    cdef public double _curr_supervision_cols
    cdef object update_supervision  # callable
    cdef bint _supervision_is_dynamic