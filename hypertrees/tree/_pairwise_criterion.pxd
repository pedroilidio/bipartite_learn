from sklearn.tree._criterion cimport Criterion
from sklearn.tree._tree cimport DTYPE_t         # Type of X
from sklearn.tree._tree cimport DOUBLE_t        # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t          # Type for indices and counters
from ._axis_criterion cimport AxisCriterion

cdef class PairwiseCriterion(Criterion):
    cdef AxisCriterion criterion