cimport numpy as cnp
from sklearn.tree._tree cimport DTYPE_t         # Type of X
from sklearn.tree._tree cimport DOUBLE_t        # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t          # Type for indices and counters
from ._axis_criterion cimport AxisCriterion

import sklearn.tree._criterion
from . import _axis_criterion


cnp.import_array()


cdef class BaseUnsupervisedCriterion(BaseComposableCriterion):
    """ABC for unsupervised criteria used is semi-supervised compositions.
    """


cdef class UnsupervisedWrapperCriterion(BaseUnsupervisedCriterion):
    """ABC for extending existing criteria with composition."""
    def __cinit__(self, *args, **kwargs):
        self.criterion = None

    def __init__(self, Criterion criterion, *args, **kwargs):
        self.criterion = criterion

    cdef inline void _copy_node_wise_attributes(self) noexcept nogil:
        self.y = self.criterion.y
        self.sample_weight = self.criterion.sample_weight

        self.sample_indices = self.criterion.sample_indices
        self.start = self.criterion.start
        self.end = self.criterion.end

        self.n_outputs = self.criterion.n_outputs
        self.n_node_samples = self.criterion.n_node_samples
        self.weighted_n_samples = self.criterion.weighted_n_samples
        self.weighted_n_node_samples = self.criterion.weighted_n_node_samples

    cdef inline void _copy_position_wise_attributes(self) noexcept nogil:
        self.pos = self.criterion.pos
        self.weighted_n_left = self.criterion.weighted_n_left
        self.weighted_n_right = self.criterion.weighted_n_right

    cdef int init(
        self,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        SIZE_t[:] sample_indices,
        SIZE_t start,
        SIZE_t end,
    ) nogil except -1:
        self.criterion.init(
            y,
            sample_weight=sample_weight,
            weighted_n_samples=weighted_n_samples,
            sample_indices=sample_indices,
            start=start,
            end=end,
        )
        self._copy_node_wise_attributes()
        return 0

    cdef int reset(self) nogil except -1:
        self.criterion.reset()
        self._copy_position_wise_attributes()
        return 0

    cdef int reverse_reset(self) nogil except -1:
        self.criterion.reverse_reset()
        self._copy_position_wise_attributes()
        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        self.criterion.update(new_pos)
        self._copy_position_wise_attributes()
        return 0

    cdef double node_impurity(self) nogil:
        return self.criterion.node_impurity()

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right
    ) nogil:
        self.criterion.children_impurity(impurity_left, impurity_right)

    cdef void node_value(self, double* dest) nogil:
        self.criterion.node_value(dest)

    cdef double impurity_improvement(
        self,
        double impurity_parent,
        double impurity_left,
        double impurity_right
    ) nogil:
        return self.criterion.impurity_improvement(
            impurity_parent,
            impurity_left,
            impurity_right,
        )

    cdef double proxy_impurity_improvement(self) nogil:
        return self.criterion.proxy_impurity_improvement()

    cdef double _proxy_improvement_factor(self) noexcept nogil:
        """If improvement = proxy_improvement / a + b, this method returns a

        This is useful when defining proxy impurity improvements for
        compositions of Criterion objects.
        """
        return self.criterion.n_outputs * self.criterion.weighted_n_samples


cdef class PairwiseCriterion(UnsupervisedWrapperCriterion):
    """Unsupervision-focused criterion to use with pairwise data.

    It wraps an AxisCriterion instance and selects the columns corresponding
    to the rows selected with 'start', 'end' and 'sample_indices' during init(). 

    It is intended to pass an square (pairwise) X as the y argument of init().
    """
    def __init__(self, AxisCriterion criterion):
        self.criterion = criterion

    cdef int init(
        self,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        SIZE_t[:] sample_indices,
        SIZE_t start,
        SIZE_t end,
    ) nogil except -1:
        (<AxisCriterion>self.criterion).axis_init(
            y,
            sample_weight=sample_weight,
            col_weights=sample_weight,
            weighted_n_samples=weighted_n_samples,
            weighted_n_cols=weighted_n_samples,
            sample_indices=sample_indices,
            col_indices=sample_indices,
            start=start,
            end=end,
            start_col=start,
            end_col=end,
        )
        self._copy_node_wise_attributes()
        return 0

    cdef double _proxy_improvement_factor(self) noexcept nogil:
        """If improvement = proxy_improvement / a + b, this method returns a

        This is useful when defining proxy impurity improvements for
        compositions of Criterion objects.
        """
        return (<AxisCriterion>self.criterion)._proxy_improvement_factor()


cdef class PairwiseSquaredError(PairwiseCriterion):
    def __init__(self, SIZE_t n_outputs, SIZE_t n_samples):
        self.criterion = _axis_criterion.AxisSquaredError(
            n_outputs=n_outputs,
            n_samples=n_samples,
        )


cdef class PairwiseSquaredErrorGSO(PairwiseCriterion):
    def __init__(self, SIZE_t n_outputs, SIZE_t n_samples):
        self.criterion = _axis_criterion.AxisSquaredErrorGSO(
            n_outputs=n_outputs,
            n_samples=n_samples,
        )


cdef class PairwiseFriedman(PairwiseCriterion):
    def __init__(self, SIZE_t n_outputs, SIZE_t n_samples):
        self.criterion = _axis_criterion.AxisFriedmanGSO(
            n_outputs=n_outputs,
            n_samples=n_samples,
        )


cdef class PairwiseGini(PairwiseCriterion):
    def __init__(
        self,
        SIZE_t n_outputs,
        cnp.ndarray[SIZE_t, ndim=1] n_classes,
    ):
        self.criterion = _axis_criterion.AxisGini(
            n_outputs=n_outputs,
            n_classes=n_classes,
        )


cdef class PairwiseEntropy(PairwiseCriterion):
    def __init__(
        self,
        SIZE_t n_outputs,
        cnp.ndarray[SIZE_t, ndim=1] n_classes,
    ):
        self.criterion = _axis_criterion.AxisEntropy(
            n_outputs=n_outputs,
            n_classes=n_classes,
        )


cdef class UnsupervisedSquaredError(UnsupervisedWrapperCriterion):
    def __init__(self, SIZE_t n_outputs, SIZE_t n_samples):
        self.criterion = sklearn.tree._criterion.MSE(
            n_outputs=n_outputs,
            n_samples=n_samples,
        )


cdef class UnsupervisedFriedman(UnsupervisedWrapperCriterion):
    def __init__(self, SIZE_t n_outputs, SIZE_t n_samples):
        self.criterion = sklearn.tree._criterion.FriedmanMSE(
            n_outputs=n_outputs,
            n_samples=n_samples,
        )

    cdef double _proxy_improvement_factor(self) noexcept nogil:
        return (
            self.n_outputs
            * self.n_outputs
            * self.weighted_n_node_samples
        )


cdef class UnsupervisedGini(UnsupervisedWrapperCriterion):
    def __init__(
        self,
        SIZE_t n_outputs,
        cnp.ndarray[SIZE_t, ndim=1] n_classes,
    ):
        self.criterion = sklearn.tree._criterion.Gini(
            n_outputs=n_outputs,
            n_classes=n_classes,
        )


cdef class UnsupervisedEntropy(UnsupervisedWrapperCriterion):
    def __init__(
        self,
        SIZE_t n_outputs,
        cnp.ndarray[SIZE_t, ndim=1] n_classes,
    ):
        self.criterion = sklearn.tree._criterion.Entropy(
            n_outputs=n_outputs,
            n_classes=n_classes,
        )