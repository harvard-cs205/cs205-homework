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
       int i, j, k, iter
       np.complex64_t c, z
       AVX.float8 z_imag, z_real, z_ri, z_sq, res, mask_gt4, mask_res0
       np.float32_t cur_iter
       np.float32_t [:, :] img = np.imag(in_coords)
       np.float32_t [:, :] real = np.real(in_coords)
       AVX.float8 avx_imag, avx_real
       float out_vals[8]

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

    # No instruction-level parallelism
    with nogil:
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=1):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter

    # with nogil:
    #     for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=1):
    #         for j in range(0, in_coords.shape[1], 8):
    #             avx_imag = AVX.make_float8(img[i, j+7], img[i, j+6], img[i,j+5], img[i,j+4], img[i,j+3], img[i,j+2], img[i,j+1],
    #                                        img[i,j+0])
    #             avx_real = AVX.make_float8(real[i, j+7], real[i, j+6], real[i,j+5], real[i,j+4], real[i,j+3], real[i,j+2], real[i,j+1],
    #                                        real[i,j+0])
    #
    #             z_imag = AVX.float_to_float8(0.)
    #             z_real = AVX.float_to_float8(0.)
    #
    #             res = AVX.float_to_float8(0.) # Stores the iteration counts
    #
    #             for iter in range(max_iterations):
    #                 z_sq = AVX.sub( AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag) )
    #                 z_ri = AVX.mul(z_real, z_imag)
    #                 cur_iter = iter
    #
    #                 z_real = AVX.add( z_sq, avx_real )
    #                 z_imag = AVX.add( AVX.add(z_ri, z_ri),  avx_imag )
    #
    #                 # Check if we have results > 4
    #                 mask_gt4 = AVX.less_than(AVX.float_to_float8(4.0), z_sq) # Mask for z_sq > 4.0
    #                 mask_res0 = AVX.greater_than(res, AVX.float_to_float8(0.)) # Mask for res > 0
    #                 res = AVX.add(res,
    #                               AVX.bitwise_and(
    #                                   AVX.bitwise_andnot(mask_res0, mask_gt4),
    #                                   AVX.float_to_float8(cur_iter) ) )
    #
    #                 # Exit loop if all results are > 0
    #                 if AVX.signs(mask_res0) == 255:
    #                     break
    #
    #                 if iter == max_iterations - 1: # On the last iteration so set any unset results
    #                     res = AVX.add(res, AVX.bitwise_andnot(mask_res0, AVX.float_to_float8(cur_iter)) )
    #
    #             # Update out_counts
    #             AVX.to_mem(res, &(out_vals[0]))
    #             for k in range(8):
    #                 out_counts[i, j + k] = <int> out_vals[k]

