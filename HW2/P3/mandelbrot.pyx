import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter, k
       np.complex64_t c, z
       float out_vals[8]
       AVX.float8 avxcr, avxci, avxzr, avxzi, avxmag, avxmask, avxiter

       # To declare AVX.float8 variables, use:
       # cdef:
       #     AVX.float8 v1, v2, v3
       #
       # And then, for example, to multiply them
       #     v3 = AVX.mul(v1, v2)
       #
       # You may find the numpy.real() and numpy.imag() fuctions helpful.

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"
    '''
    with nogil:
        for i in prange(in_coords.shape[0], schedule="static", chunksize=1, num_threads=1):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter
    '''
    with nogil:
        for i in prange(in_coords.shape[0], schedule="static", chunksize=1, num_threads=4):
            for j in xrange(0,in_coords.shape[1],8):
                avxcr = AVX.make_float8(in_coords[i, j+7].real,
                                        in_coords[i, j+6].real,
                                        in_coords[i, j+5].real,
                                        in_coords[i, j+4].real,
                                        in_coords[i, j+3].real,
                                        in_coords[i, j+2].real,
                                        in_coords[i, j+1].real,
                                        in_coords[i, j+0].real)

                avxci = AVX.make_float8(in_coords[i, j+7].imag,
                                        in_coords[i, j+6].imag,
                                        in_coords[i, j+5].imag,
                                        in_coords[i, j+4].imag,
                                        in_coords[i, j+3].imag,
                                        in_coords[i, j+2].imag,
                                        in_coords[i, j+1].imag,
                                        in_coords[i, j+0].imag)

                # Keep track of the real part and imaginary part.
                avxzr = AVX.float_to_float8(0.0)
                avxzi = AVX.float_to_float8(0.0)

                # Keep track of the magnitude
                avxmag = AVX.float_to_float8(0.0)

                # Check if the magnitude is less than 4.0
                avxmask = AVX.less_than(avxmag,AVX.float_to_float8(4.0))
                avxiter = AVX.float_to_float8(0.0)

                for iter in range(max_iterations):
                    # Increment iteration
                    avxiter = AVX.add(avxiter, AVX.bitwise_and(AVX.float_to_float8(1.0), avxmask))

                    # break if all of the 8 values are greater than 8.
                    if (AVX.signs(avxmask) == 0):
                        break

                    # Update the real part and the imaginary part.
                    avxzr, avxzi = AVX.add(AVX.fmsub(avxzr, avxzr, AVX.mul(avxzi, avxzi)), avxcr), AVX.add(AVX.fmadd(avxzr, avxzi, AVX.mul(avxzr, avxzi)), avxci)
              
                    avxmag = AVX.fmadd(avxzr, avxzr, AVX.mul(avxzi, avxzi))
                    avxmask = AVX.less_than(avxmag, AVX.float_to_float8(4.0))
                    
                # Copy them to memory
                AVX.to_mem(avxiter, &(out_vals[0]))
                for k in range(8):
                    out_counts[i,j+k] = <np.uint32_t > out_vals[k]

# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval, tmp, mask
        float out_vals[8]
        float [:] out_view = out_vals

    assert values.shape[0] == 8

    # Note that the order of the arguments here is opposite the direction when
    # we retrieve them into memory.
    avxval = AVX.make_float8(values[7],
                             values[6],
                             values[5],
                             values[4],
                             values[3],
                             values[2],
                             values[1],
                             values[0])

    avxval = AVX.sqrt(avxval)

    # mask will be true where 2.0 < avxval
    mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)

    # invert mask and select off values, so should be 2.0 >= avxval
    avxval = AVX.bitwise_andnot(mask, avxval)

    AVX.to_mem(avxval, &(out_vals[0]))

    return np.array(out_view)
