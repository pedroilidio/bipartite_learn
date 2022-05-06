from sklearn.tree._splitter cimport Splitter
from ._sklearn_splitter cimport (
    BestSplitter, 
#     BaseDenseSplitter, RandomSplitter,
#     BaseSparseSplitter, BestSparseSplitter, RandomSparseSplitter 
)
from sklearn.tree._criterion cimport Criterion

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters


cdef class SSCriterion(Criterion):
    cdef Criterion supervised_criterion,
    cdef Criterion unsupervised_criterion,
    cdef double supervision
    cdef const DOUBLE_t[:, ::1] X

    cdef int semisupervised_init(
            self, const DOUBLE_t[:, ::1] X, const DOUBLE_t[:, ::1] y,
            DOUBLE_t* sample_weight,
            double weighted_n_samples, SIZE_t* samples, SIZE_t start,
            SIZE_t end) nogil except -1


# cdef class SSSplitter(Splitter):
#     cdef const DOUBLE_t[:, ::1] X_targets
#     cdef SSCriterion sscriterion
# 
# cdef class BaseSSDenseSplitter(BaseDenseSplitter):
#     cdef const DOUBLE_t[:, ::1] X_targets
#     cdef SSCriterion sscriterion

cdef class SSBestSplitter(BestSplitter):
    cdef const DOUBLE_t[:, ::1] X_targets
    cdef SSCriterion sscriterion

# cdef class SSRandomSplitter(RandomSplitter):
#     cdef const DOUBLE_t[:, ::1] X_targets
#     cdef SSCriterion sscriterion
# 
# cdef class SSBaseSparseSplitter(BaseSparseSplitter):
#     cdef const DOUBLE_t[:, ::1] X_targets
#     cdef SSCriterion sscriterion
# 
# cdef class SSBestSparseSplitter(BestSparseSplitter):
#     cdef const DOUBLE_t[:, ::1] X_targets
#     cdef SSCriterion sscriterion
# 
# cdef class SSRandomSparseSplitter(RandomSparseSplitter):
#     cdef const DOUBLE_t[:, ::1] X_targets
#     cdef SSCriterion sscriterion