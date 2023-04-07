from sklearn.tree._splitter cimport Splitter
from sklearn.tree._criterion cimport Criterion, RegressionCriterion
from sklearn.tree._tree cimport DTYPE_t         # Type of X
from sklearn.tree._tree cimport DOUBLE_t        # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t          # Type for indices and counters
from ._bipartite_criterion cimport BipartiteCriterion, GMO
from ._axis_criterion cimport BaseComposableCriterion, AxisCriterion
from ._unsupervised_criterion cimport BaseUnsupervisedCriterion


cdef class BaseDenseSplitter(Splitter):
    cdef const DTYPE_t[:, :] X


cdef class BaseSemisupervisedAxisCriterion(AxisCriterion):
    """Abstract base class to facilitate future extension.

    This class should be used in type testing to allow future extensions
    to inherit from this class and still be accepted, not needing to subclass
    SSCompositeCriterion.
    """


cdef class SSCompositeCriterion(AxisCriterion):
    cdef public BaseComposableCriterion supervised_criterion
    cdef public BaseUnsupervisedCriterion unsupervised_criterion
    cdef const DOUBLE_t[:, ::1] X  # X data to initialize unuspervised criterion
    cdef SIZE_t n_features  # X.shape[1]

    # The supervision attribute is a float between 0 and 1 that weights
    # supervised and unsupervised impurities when calculating the total
    # semisupervised impurity:
    #
    #   final_impurity = (
    #     sup * self.supervised_criterion.node_impurity()
    #     / self._root_supervised_impurity
    #     + (1-sup) * self.unsupervised_criterion.node_impurity()
    #     / self._root_unsupervised_impurity
    #   )
    #
    # Its value can be dynamically controlled by a Python callable passed to
    # the constructor's update_supervision parameter, which should only receive
    # the current SSCompositeCriterion instance as argument and return a float.

    cdef double _root_supervised_impurity
    cdef double _root_unsupervised_impurity

    cdef double supervision           # first supervision value received
    cdef double _curr_supervision     # Current supervision amount
    cdef double _proxy_supervision    # Combined constants of proxy impurities
    cdef object update_supervision    # callable to update supervision

    cdef double _cached_u_impurity_left
    cdef double _cached_u_impurity_right
    cdef double _cached_s_impurity_left
    cdef double _cached_s_impurity_right

    # We test isinstance(self.supervised_criterion, AxisCriterion) only once at
    # __init__ to avoid requesting the GIL in methods that require that
    # condition.
    cdef bint _supervised_is_axis_criterion

    cpdef void set_X(self, const DOUBLE_t[:, ::1] X)
    cdef void set_root_impurities(self) nogil
    cdef void ss_children_impurities(
        self,
        double* u_impurity_left,
        double* u_impurity_right,
        double* s_impurity_left,
        double* s_impurity_right,
    ) nogil
    cdef inline void _copy_position_wise_attributes(self) noexcept nogil
    cdef int _update_supervision(self) except -1 nogil
    cdef int _set_proxy_supervision(self) except -1 nogil


# FIXME
cdef class SingleFeatureSSCompositeCriterion(SSCompositeCriterion):
    """Uses only the current feature as unsupervised data.
    """
    cdef SIZE_t current_feature
    cdef const DOUBLE_t[:, ::1] _full_X


cdef class BipartiteSemisupervisedCriterion(GMO):
    cdef public GMO bipartite_criterion
    cdef Criterion supervised_criterion_rows
    cdef Criterion supervised_criterion_cols
    cdef Criterion unsupervised_criterion_rows
    cdef Criterion unsupervised_criterion_cols

    cdef SIZE_t n_row_features
    cdef SIZE_t n_col_features

    cdef double _root_supervised_impurity
    cdef double _root_unsupervised_impurity_rows
    cdef double _root_unsupervised_impurity_cols

    cdef double _curr_supervision_rows
    cdef double _curr_supervision_cols

    cdef object update_supervision  # callable

    cpdef void set_X(
        self,
        const DOUBLE_t[:, ::1] X_rows,
        const DOUBLE_t[:, ::1] X_cols,
    )