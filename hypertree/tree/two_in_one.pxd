from sklearn.tree._tree cimport SIZE_t

cdef SIZE_t encode(SIZE_t a, SIZE_t b) nogil

cdef void decode_unsigned(SIZE_t x, SIZE_t* res_a, SIZE_t* res_b) nogil

cdef void decode_signed(SIZE_t x, SIZE_t* res_a, SIZE_t* res_b) nogil
