# cython: boundscheck=False
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


# TODO
# cdef class UnsupervisedNumericCriterion(BaseUnsupervisedCriterion):
#     """ABC for unsupervised criteria that work on real-valued features.
#     Based on supervised regression criteria.
#     """


# cdef class UnsupervisedCategoricCriterion(BaseUnsupervisedCriterion):
#     """ABC for unsupervised criteria that work on categoric features.
#     Based on supervised classification criteria.
#     """


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
        self.n_samples = self.criterion.n_samples
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
        return self.criterion.weighted_n_samples


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


cdef class BaseKernelCriterion(BaseUnsupervisedCriterion):

    def __cinit__(self, SIZE_t n_samples, *args, **kwargs):
        """Initialize parameters for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted
        n_samples : SIZE_t
            The total number of samples to fit on
        """
        # Default values
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_samples = n_samples
        self.n_outputs = 0  # Unsupervised

        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sum_total = 0.0
        self.sum_left = 0.0
        self.sum_right = 0.0

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    cdef int init(
        self,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        const SIZE_t[:] sample_indices,
        SIZE_t start,
        SIZE_t end,
    ) except -1 nogil:
        """Initialize the criterion.
        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        """
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = (  # FIXME: - diag ** 2
            (weighted_n_samples - 1.0) * weighted_n_samples
        ) / 2
        self.weighted_n_node_samples = 0.0

        cdef:
            SIZE_t i, j, p, q
            DOUBLE_t y_ij, w_y_ij
            DOUBLE_t wi = 1.0, wj = 1.0

        self.sum_total = 0.0

        # Sum elements of the lower triangle of y
        for p in range(start + 1, end):
            i = sample_indices[p]

            if sample_weight is not None:
                wi = sample_weight[i]

            for q in range(start, p):  # Skip main diagonal
                j = sample_indices[q]
                if sample_weight is not None:
                    wj = sample_weight[j]

                y_ij = self.y[i, j]
                w_y_ij = wi * wj * y_ij
                self.sum_total += w_y_ij
                self.weighted_n_node_samples += wi * wj

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start."""
        self.sum_left = 0.0
        self.sum_right = self.sum_total

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end."""
        self.sum_left = self.sum_total
        self.sum_right = 0.0

        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0
        self.pos = self.end
        return 0

    cdef int update(self, SIZE_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""
        cdef:
            const DOUBLE_t[:] sample_weight = self.sample_weight
            const SIZE_t[:] sample_indices = self.sample_indices
            SIZE_t start = self.start
            SIZE_t pos = self.pos
            SIZE_t end = self.end
            SIZE_t i, j, p, q
            DOUBLE_t y_ij, w_y_ij
            DOUBLE_t wi = 1.0, wj = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    wi = sample_weight[i]

                for q in range(start, p):
                    j = sample_indices[q]

                    if sample_weight is not None:
                        wj = sample_weight[j]

                    self.sum_left += wi * wj * self.y[i, j]
                    self.weighted_n_left += wi * wj

                for q in range(p + 1, end):  # Skip main diagonal
                    j = sample_indices[q]

                    if sample_weight is not None:
                        wj = sample_weight[j]

                    self.sum_right -= wi * wj * self.y[i, j]
                    self.weighted_n_right -= wi * wj

        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    wi = sample_weight[i]

                for q in range(start, p):
                    j = sample_indices[q]

                    if sample_weight is not None:
                        wj = sample_weight[j]

                    self.sum_left -= wi * wj * self.y[i, j]
                    self.weighted_n_left -= wi * wj

                for q in range(p + 1, end):  # Skip main diagonal
                # for q in range(p, end):  # Skip main diagonal
                    j = sample_indices[q]

                    if sample_weight is not None:
                        wj = sample_weight[j]

                    self.sum_right += wi * wj * self.y[i, j]
                    self.weighted_n_right += wi * wj

        self.pos = new_pos
        return 0

    cdef double _proxy_improvement_factor(self) noexcept nogil:
        return self.weighted_n_samples
        
    cdef double node_impurity(self) noexcept nogil:
        pass

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right,
    ) noexcept nogil:
        pass
    

cdef class MeanDistance(BaseKernelCriterion):
    """Assumes y is a distance pairwise matrix."""

    cdef double node_impurity(self) noexcept nogil:
        if self.n_node_samples > 1:
            return self.sum_total / self.weighted_n_node_samples
        return 0.0

    cdef void children_impurity(
        self,
        double* impurity_left,
        double* impurity_right,
    ) noexcept nogil:
        if self.weighted_n_left > 0.0:
            impurity_left[0] = self.sum_left / self.weighted_n_left
        else:
            impurity_left[0] = 0.0
        if self.weighted_n_right > 0.0:
            impurity_right[0] = self.sum_right / self.weighted_n_right
        else:
            impurity_right[0] = 0.0


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

    cdef double _proxy_improvement_factor(self) noexcept nogil:
        return self.n_outputs * self.weighted_n_samples


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
