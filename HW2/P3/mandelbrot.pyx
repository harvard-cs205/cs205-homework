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
    cdef:
        AVX.float8 avx_real, avx_imag, avx_zreal, avx_zimag, avx_iter
        AVX.float8 avx_mask, avx_mag, avx_zsqre_real, avx_zsqre_imag 
        AVX.float8 avx_mask_iter
        float out_vals[8]
    with nogil:
        #for i in range(in_coords.shape[0]):
        for i in prange(in_coords.shape[0], schedule="static", 
                        chunksize=1, num_threads=1):
            for j in range(0,in_coords.shape[1],8):
                avx_real = AVX.make_float8(in_coords[i, j+7].real,
                                           in_coords[i, j+6].real,
                                           in_coords[i, j+5].real,
                                           in_coords[i, j+4].real,
                                           in_coords[i, j+3].real,
                                           in_coords[i, j+2].real,
                                           in_coords[i, j+1].real,                  
                                           in_coords[i, j].real
                                          )
                avx_imag = AVX.make_float8(in_coords[i, j+7].imag,
                                           in_coords[i, j+6].imag,
                                           in_coords[i, j+5].imag,
                                           in_coords[i, j+4].imag,
                                           in_coords[i, j+3].imag,
                                           in_coords[i, j+2].imag,
                                           in_coords[i, j+1].imag,           
                                           in_coords[i, j].imag
                                          )
                avx_zreal = AVX.float_to_float8(0.0)
                avx_zimag = AVX.float_to_float8(0.0)
                avx_iter = AVX.float_to_float8(0.0)
                for iter in range(max_iterations):
                    avx_mag = AVX.add(AVX.mul(avx_zreal, avx_zreal), 
                                     AVX.mul(avx_zimag, avx_zimag) )
                    # mask will be true where 4.0 < avx_mag                
                    avx_mask = AVX.greater_than(AVX.float_to_float8(4.0), avx_mag)
                    avx_zsqre_real = AVX.sub(AVX.mul(avx_zreal, avx_zreal), 
                                AVX.mul(avx_zimag, avx_zimag) )
                    avx_zsqre_imag = AVX.mul(AVX.mul(avx_zreal, avx_zimag), 
                                             AVX.float_to_float8(2.0))
                    avx_zreal = AVX.add(avx_zsqre_real, avx_real)
                    avx_zimag = AVX.add(avx_zsqre_imag, avx_imag)
                    avx_mask_iter = AVX.bitwise_and(AVX.float_to_float8(1),
                                                avx_mask
                                                )
                    avx_iter = AVX.add(avx_iter, avx_mask_iter)
                AVX.to_mem(avx_iter, &(out_vals[0]))
                out_counts[i, j] = int(out_vals[0])
                out_counts[i, j+1] = int(out_vals[1])
                out_counts[i, j+2] = int(out_vals[2])
                out_counts[i, j+3] = int(out_vals[3])
                out_counts[i, j+4] = int(out_vals[4])
                out_counts[i, j+5] = int(out_vals[5])
                out_counts[i, j+6] = int(out_vals[6])
                out_counts[i, j+7] = int(out_vals[7])


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
