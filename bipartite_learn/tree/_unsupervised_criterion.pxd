from ._axis_criterion cimport BaseComposableCriterion
from sklearn.tree._criterion cimport Criterion

cdef class BaseUnsupervisedCriterion(BaseComposableCriterion):
    pass

cdef class UnsupervisedWrapperCriterion(BaseUnsupervisedCriterion):
    cdef Criterion criterion
    cdef inline void _copy_node_wise_attributes(self) noexcept nogil
    cdef inline void _copy_position_wise_attributes(self) noexcept nogil

cdef class PairwiseCriterion(UnsupervisedWrapperCriterion):
    pass

cdef class BaseKernelCriterion(BaseUnsupervisedCriterion):
    cdef double sum_total
    cdef double sum_left
    cdef double sum_right