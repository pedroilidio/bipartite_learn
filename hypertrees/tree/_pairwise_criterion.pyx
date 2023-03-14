# cython: boundscheck=False
from sklearn.tree._criterion cimport Criterion


cdef class PairwiseCriterion(Criterion):
    """Unsupervision-focused criterion to use with pairwise data.

    It wraps an AxisCriterion instance and selects the columns corresponding
    to the rows selected with 'start', 'end' and 'sample_indices' during init(). 

    It is intended to pass an square (pairwise) X as the y argument of init().
    """
    def __init__(self, AxisCriterion criterion):
        self.criterion = criterion
        self.n_outputs = criterion.n_outputs
        self.n_samples = criterion.n_samples

    cdef int init(
        self,
        const DOUBLE_t[:, ::1] y,
        const DOUBLE_t[:] sample_weight,
        double weighted_n_samples,
        SIZE_t[:] sample_indices,
        SIZE_t start,
        SIZE_t end,
    ) nogil except -1:
        self.criterion.init_columns(
            sample_weight, weighted_n_samples, sample_indices, start, end,
        )
        return self.criterion.init(
            y, sample_weight, weighted_n_samples, sample_indices, start, end,
        )

    cdef int reset(self) nogil except -1:
        return self.criterion.reset()

    cdef int reverse_reset(self) nogil except -1:
        return self.criterion.reverse_reset()

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        return self.criterion.update(new_pos)

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
