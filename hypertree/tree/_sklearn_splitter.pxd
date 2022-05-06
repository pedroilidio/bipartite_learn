import numpy as np
cimport numpy as np

from sklearn.tree._splitter cimport Splitter, SplitRecord

from sklearn.tree._tree cimport DTYPE_t          # Type of X
from sklearn.tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from sklearn.tree._tree cimport SIZE_t           # Type for indices and counters
from sklearn.tree._tree cimport INT32_t          # Signed 32 bit integer
from sklearn.tree._tree cimport UINT32_t         # Unsigned 32 bit integer


cdef class BaseDenseSplitter(Splitter):
    cdef SIZE_t n_total_samples
    cdef const DTYPE_t[:, :] X

cdef class BestSplitter(BaseDenseSplitter):
    pass

cdef class RandomSplitter(BaseDenseSplitter):
    pass

# cdef class BaseSparseSplitter(Splitter):
#     pass
# 
# cdef class BestSparseSplitter(BaseDenseSplitter):
#     pass
# 
# cdef class RandomSparseSplitter(BaseSparseSplitter):
#     pass