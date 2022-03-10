# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
cdef SIZE_t HALF_N_BITS = sizeof(SIZE_t) * 4  # 8/2
cdef SIZE_t BITMASK = 0
for _ in range(HALF_N_BITS):
    BITMASK <<= 1
    BITMASK += 1
cdef SIZE_t BITMASKR = BITMASK << HALF_N_BITS


cdef SIZE_t encode(SIZE_t a, SIZE_t b) nogil:
    return ((a & BITMASK) << HALF_N_BITS) | (b & BITMASK)


cdef void decode_unsigned(SIZE_t x, SIZE_t* res_a, SIZE_t* res_b) nogil:
    res_a[0] = x >> HALF_N_BITS & BITMASK
    res_b[0] = x & BITMASK


cdef void decode_signed(SIZE_t x, SIZE_t* res_a, SIZE_t* res_b) nogil:
    decode_unsigned(x, res_a, res_b)

    if res_a[0] >> (HALF_N_BITS-1):
        res_a[0] |= BITMASKR
    if res_b[0] >> (HALF_N_BITS-1):
        res_b[0] |= BITMASKR


def test(a, b):
    cdef SIZE_t a2, b2
    print("a, b", a, b)
    print("sizeof(SIZE_t)", sizeof(SIZE_t))
    print("BITMASK", BITMASK, format(BITMASK, '0b'))
    r = encode(a, b)
    decode_unsigned(r, &a2, &b2)
    print(r, a2, b2)
    decode_signed(r, &a2, &b2)
    print(r, a2, b2)
