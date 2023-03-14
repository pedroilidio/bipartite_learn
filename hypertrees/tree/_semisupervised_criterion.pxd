from sklearn.tree._splitter cimport Splitter
from sklearn.tree._criterion cimport Criterion, RegressionCriterion
from sklearn.tree._tree cimport DTYPE_t         # Type of X
from sklearn.tree._tree cimport DOUBLE_t        # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t          # Type for indices and counters
from ._nd_criterion cimport BipartiteCriterion, BipartiteRegressionCriterion


cdef class BaseDenseSplitter(Splitter):
    cdef const DTYPE_t[:, :] X

cdef class SemisupervisedCriterion(Criterion):
    """Base class for semantic purposes and future maintenance.
    """

cdef class SSCompositeCriterion(SemisupervisedCriterion):
    cdef public Criterion supervised_criterion
    cdef public Criterion unsupervised_criterion
    cdef const DOUBLE_t[:, ::1] X
    cdef public SIZE_t n_features

    # The supervision attribute is a float between 0 and 1 that weights
    # supervised and unsupervised impurities when calculating the total
    # semisupervised impurity:
    #
    #   final_impurity = (
    #     sup * self.supervised_criterion.node_impurity()
    #     + (1-sup) * self.unsupervised_criterion.node_impurity()
    #   )
    #
    # Its value can be dynamically controlled by a Python callable passed to
    # the constructor's update_supervision parameter, which should only receive
    # the current SSCompositeCriterion instance as argument and return a float.

    # TODO: setter method instead of public.
    cdef public double _curr_supervision     # Current supervision amount
    cdef public double supervision           # first supervision value received
    cdef public object update_supervision    # callable to update supervision

    # The _supervision_is_dynamic serves merely as a C-typed flag to
    # avoid asking for the GIL if update_supervision was not provided.
    cdef bint _supervision_is_dynamic

    # TODO: explain
    cdef SIZE_t _cached_pos
    cdef double _cached_u_impurity_left
    cdef double _cached_u_impurity_right
    cdef double _cached_s_impurity_left
    cdef double _cached_s_impurity_right

    # TODO: explain
    cdef void unpack_y(self, const DOUBLE_t[:, ::1] y) nogil

    cdef void ss_children_impurities(
        self,
        double* u_impurity_left,
        double* u_impurity_right,
        double* s_impurity_left,
        double* s_impurity_right,
    ) nogil

# FIXME
cdef class SingleFeatureSSCompositeCriterion(SSCompositeCriterion):
    """Uses only the current feature as unsupervised data.
    """
    cdef public SIZE_t current_feature
    cdef public double current_node_impurity
    cdef const DOUBLE_t[:, ::1] full_X

# Regression?
cdef class BipartiteSemisupervisedCriterion(BipartiteCriterion):
    cdef public Criterion unsupervised_criterion_rows
    cdef public Criterion unsupervised_criterion_cols
    cdef public BipartiteCriterion supervised_bipartite_criterion

    # TODO: we need to get access to the wrappers owned by the splitters to
    # dynamically change the supervision, since supervision is utilized in
    # proxy_impurity_improvement(). Ideally, we would like to drop this
    # dependency.
    cdef public SSCompositeCriterion ss_criterion_rows
    cdef public SSCompositeCriterion ss_criterion_cols

    # References to supervised_bipartite_criterion's components
    cdef public Criterion supervised_criterion_rows
    cdef public Criterion supervised_criterion_cols

    cdef SIZE_t n_row_features
    cdef SIZE_t n_col_features

    cdef public double supervision_rows
    cdef public double supervision_cols
    cdef double _curr_supervision_rows
    cdef double _curr_supervision_cols
    cdef object update_supervision  # callable
    cdef bint _supervision_is_dynamic
