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
        for i in prange(in_coords.shape[0], num_threads=4):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter

    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrotAVX(np.float32_t [:, :] in_coords_real,
                    np.float32_t [:, :] in_coords_imag,
                    np.uint32_t [:, :] out_counts,
                    int max_iterations=511):

    cdef:
        int i, j, iter, k
        AVX.float8 cr, ci, zr, zi, tmpr, tmpi
        AVX.float8 iter_count
        AVX.float8 magnitude, mask, keep_iter
        int signcheck
        float out_vals[8]
        float [:] out_view = out_vals
        
        # To declare AVX.float8 variables, use:
        # cdef:
        #     AVX.float8 v1, v2, v3
        #
        # And then, for example, to multiply them
        #     v3 = AVX.mul(v1, v2)
        #
        # You may find the numpy.real() and numpy.imag() fuctions helpful.

    assert in_coords_real.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords_imag.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords_real.shape[0] == in_coords_imag.shape[0], "Real part and Imag part must be the same size"
    assert in_coords_real.shape[1] == in_coords_imag.shape[1], "Real part and Imag part must be the same size"
    assert in_coords_real.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords_real.shape[1] == out_counts.shape[1], "Input and output arrays must be the same size"
    assert in_coords_imag.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords_imag.shape[1] == out_counts.shape[1], "Input and output arrays must be the same size"

    with nogil:
        for i in prange(in_coords_real.shape[0], num_threads=1):
            for j in range(0,in_coords_real.shape[1],8):
                #cr = in_coords_real[i, j]
                cr = AVX.make_float8(in_coords_real[i, j+0], in_coords_real[i, j+1], in_coords_real[i, j+2], in_coords_real[i, j+3],
                                     in_coords_real[i, j+4], in_coords_real[i, j+5], in_coords_real[i, j+6], in_coords_real[i, j+7])
                #ci = in_coords_imag[i, j]
                ci = AVX.make_float8(in_coords_imag[i, j+0], in_coords_imag[i, j+1], in_coords_imag[i, j+2], in_coords_imag[i, j+3],
                                     in_coords_imag[i, j+4], in_coords_imag[i, j+5], in_coords_imag[i, j+6], in_coords_imag[i, j+7])
                #zr = 0
                zr = AVX.float_to_float8(0.0)
                #zi = 0
                zi = AVX.float_to_float8(0.0)
                iter_count = AVX.float_to_float8(0.0)
                keep_iter = AVX.float_to_float8(-1.0)
                for iter in range(max_iterations):
                    #tmpr = zr*zr - zi*zi + cr
                    tmpr = AVX.add( AVX.sub( AVX.mul(zr, zr), AVX.mul(zi,zi) ), cr )
                    #tmpi = 2.0*zr*zi + ci
                    tmpi = AVX.add( AVX.mul( AVX.float_to_float8(2.0), AVX.mul(zr,zi) ), ci )
                    zr = tmpr
                    zi = tmpi
                    
                    #if zr*zr + zi*zi <= 4: add iteration count
                    magnitude = AVX.add( AVX.mul(zr, zr), AVX.mul(zi,zi) )
                    mask = AVX.greater_than(magnitude, AVX.float_to_float8(4.0))
                    keep_iter = AVX.bitwise_andnot(mask, keep_iter) #mask for those elements still need iterate
                    iter_count = AVX.add( iter_count, AVX.bitwise_and(keep_iter, AVX.float_to_float8(1.0)) )
                    signcheck = AVX.signs(keep_iter)
                    #if zr*zr + zi*zi > 4
                    #    break
                    if signcheck == 0:
                        break
                    
                AVX.to_mem(iter_count, &(out_vals[0]))
                for k in range(8):
                    out_counts[i, j+k] = int(out_view[k])
 
    

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
