from sklearn.tree._criterion cimport Criterion

cdef class BaseCriterionWrapper(Criterion):
    cdef Criterion criterion
    cdef inline void _copy_node_wise_attributes(self) noexcept nogil
    cdef inline void _copy_position_wise_attributes(self) noexcept nogil

cdef class PairwiseCriterion(BaseCriterionWrapper):
    pass