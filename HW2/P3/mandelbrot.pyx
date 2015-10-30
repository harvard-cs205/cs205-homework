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
       # np.complex64_t c, z
       AVX.float8 z_real, z_imag, c_real, c_imag, iters, fours, ones, newz_real, newz_imag, mask, z_magnitude_squared_avx

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

    # with nogil:
    #     for i in range(in_coords.shape[0]):
    #         for j in range(in_coords.shape[1]):
    #             c = in_coords[i, j]
    #             z = 0
    #             for iter in range(max_iterations):
    #                 if magnitude_squared(z) > 4:
    #                     break
    #                 z = z * z + c
    #             out_counts[i, j] = iter


    with nogil:
        for i in prange(in_coords.shape[0], num_threads=4, schedule='static', chunksize=1):
            for j in range(0, in_coords.shape[1], 8):
              c_real = AVX.make_float8(in_coords[i, j+7].real,
                                       in_coords[i, j+6].real,
                                       in_coords[i, j+5].real,
                                       in_coords[i, j+4].real,
                                       in_coords[i, j+3].real,
                                       in_coords[i, j+2].real,
                                       in_coords[i, j+1].real,
                                       in_coords[i, j].real)
              c_imag = AVX.make_float8(in_coords[i, j+7].imag,
                                       in_coords[i, j+6].imag,
                                       in_coords[i, j+5].imag,
                                       in_coords[i, j+4].imag,
                                       in_coords[i, j+3].imag,
                                       in_coords[i, j+2].imag,
                                       in_coords[i, j+1].imag,
                                       in_coords[i, j].imag)
              z_real = AVX.float_to_float8(0.0)
              z_imag = AVX.float_to_float8(0.0)

              iters = AVX.float_to_float8(0.0)
              fours = AVX.float_to_float8(4.0)
              ones = AVX.float_to_float8(1.0)
              for iter in range(max_iterations):
                  z_magnitude_squared_avx = AVX.fmadd(z_real, z_real, AVX.mul(z_imag, z_imag))
                  mask = AVX.less_than(z_magnitude_squared_avx, fours)
                  if AVX.signs(mask) == 0:
                      break
                  iters = AVX.add(iters, AVX.bitwise_and(mask, ones))
                  newz_real = AVX.fmsub(z_real, z_real, AVX.mul(z_imag, z_imag))
                  newz_imag = AVX.fmadd(z_real, z_imag, AVX.mul(z_real, z_imag))
                  z_real = AVX.add(newz_real, c_real)
                  z_imag = AVX.add(newz_imag, c_imag)
              counts_to_output(iters, out_counts, i, j)

                  
cdef void counts_to_output(AVX.float8 counts,
                      np.uint32_t [:, :] out_counts,
                      int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int val, simd

    AVX.to_mem(counts, &(tmp_counts[0]))

    simd = 8
    for val in range(simd):
        out_counts[i, j+val] = <np.uint32_t>tmp_counts[val]

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
