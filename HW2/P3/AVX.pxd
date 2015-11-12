# Cython Definition file for AVX.h

cdef extern from "AVX.h" nogil:
    ctypedef struct float8:
        pass
    ctypedef struct int8:
        pass

    # Create a vector of 8 floats
    #     from one value
    float8 float_to_float8(float)
    #     from 8 values
    float8 make_float8(float, float, float, float, float, float, float, float)

    # Arithmetic:
    #     Each of these operate on their 8 values independently, and return a
    #     new float8
    float8 add(float8, float8)
    float8 sub(float8, float8)
    float8 mul(float8, float8)
    float8 div(float8, float8)
    float8 sqrt(float8)
    float8 fmadd(float8 a, float8 b, float8 c)  # a * b + c
    float8 fmsub(float8 a, float8 b, float8 c)  # a / b + c

    # Comparisons:
    #     When comparing to vectors of float, the result is either 0.0 or -1.0,
    #     in each of the 8 locations.  Note that -1.0 is all 1s in its
    #     repesentation.  This can be useful with bitwise_and(), below.
    float8 less_than(float8 a, float8 b)     # (a < b) -> 0.0 or -1.0
    float8 greater_than(float8 a, float8 b)  # (a > b) -> 0.0 or -1.0

    # Bitwise AND:
    #     Note that 0.0 = all zeros,
    #              -1.0 == all 1s
    #     So, (val & 0.0) == 0.0
    #     and, (val & -1.0) == val
    float8 bitwise_and(float8, float8)
    # This version inverts its first argument before the and
    float8 bitwise_andnot(float8 mask, float8 val)

    # Helpers:
    #     This extracts the signs of each float into an 8-bit value.
    int signs(float8)

    #     This copies the contents of the float8 into a memory location
    void to_mem(float8, float *)