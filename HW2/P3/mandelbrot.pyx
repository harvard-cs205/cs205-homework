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
       int i, j, iter
       np.complex64_t c, z

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

    with nogil:
        for i in prange(in_coords.shape[0], num_threads=4, schedule='static', chunksize=1):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter

# use instruction level
cdef void counts_to_output(AVX.float8 counts,
                           np.uint32_t [:, :] out_counts,
                           int i,
                           int j) nogil:
    cdef:
        float tmp_counts[8]
        int k

    AVX.to_mem(counts, &(tmp_counts[0]))

    for k in range(8):
        out_counts[i, j*8+k]=<unsigned int>tmp_counts[k]

cpdef mandelbrot_new(np.complex64_t [:, :] in_coords,
                     np.uint32_t [:, :] out_counts,
                     int max_iterations=511):
    cdef:
       int i, j, iter, k
       float [:, :] real, imag
       AVX.float8 cr, ci, zr, zr_n, zi, magz2, signs, counts

    real = np.real(in_coords)
    imag = np.imag(in_coords)


       # You may find the numpy.real() and numpy.imag() fuctions helpful.
    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    with nogil:
        for i in prange(in_coords.shape[0], num_threads=4, schedule='static', chunksize=1):
            for j in range(in_coords.shape[1]/8):
                cr = AVX.make_float8(real[i,j*8+7],
                                     real[i,j*8+6],
                                     real[i,j*8+5],
                                     real[i,j*8+4],
                                     real[i,j*8+3],
                                     real[i,j*8+2],
                                     real[i,j*8+1],
                                     real[i,j*8])

                ci = AVX.make_float8(imag[i,j*8+7],
                                     imag[i,j*8+6],
                                     imag[i,j*8+5],
                                     imag[i,j*8+4],
                                     imag[i,j*8+3],
                                     imag[i,j*8+2],
                                     imag[i,j*8+1],
                                     imag[i,j*8])

                zr = AVX.float_to_float8(0.0)
                zi = AVX.float_to_float8(0.0)
                counts = AVX.float_to_float8(0.0)

                for iter in range(max_iterations):

                    # @x: calculate the magnitude square
                    magz2 = AVX.add(AVX.mul(zr,zr), AVX.mul(zi,zi))

                    # @x: check if magnitude square of z less than 4, return 1 if true, return 0 if false
                    signs = AVX.bitwise_and(AVX.less_than(magz2, AVX.float_to_float8(4.0)),AVX.float_to_float8(1.0))

                    # @x: update inter counts
                    counts = AVX.add(counts, signs)

                    # @x: break if all values>4
                    if AVX.signs(AVX.less_than(magz2, AVX.float_to_float8(4.0)))==0:
                        break

                    # @x: calculate new zr value
                    zr_n = AVX.add(AVX.sub(AVX.mul(zr,zr), AVX.mul(zi,zi)), cr)

                    # @x: calculate new zi value
                    zi = AVX.fmadd(AVX.mul(AVX.float_to_float8(2.0),zr),zi,ci)

                    # @x: update zr value with zr_n
                    zr = zr_n


                counts_to_output(counts, out_counts, i, j)

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
