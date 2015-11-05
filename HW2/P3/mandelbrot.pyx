import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange, parallel

# function to compute complex number magnitude leveraging AVX
cdef AVX.float8 magnitude_squared_avx(AVX.float8 z_real, AVX.float8 z_imag) nogil:
    return AVX.add( AVX.mul( z_real, z_real ), AVX.mul( z_imag, z_imag ) )

@cython.boundscheck(False)
@cython.wraparound(False)

cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.float32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter, avx_check
       np.complex64_t c, z
       np.float32_t [:, :] in_coords_real
       np.float32_t [:, :] in_coords_imag

       # To declare AVX.float8 variables, use:
       cdef:
           AVX.float8 avx_c_real, avx_c_imag, avx_z_real, avx_z_imag, avx_z_real_temp, avx_comparator_constant, \
                      mag_z, avx_iter, mask, avx_iter_increment, avx_multi_constant, avx_barrier_constant

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    # extracting real and imaginary parts of in_coords for use with AVX
    in_coords_real = numpy.real(in_coords)
    in_coords_imag = numpy.imag(in_coords)
    
    with nogil, parallel(num_threads=4):
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1):
            for j in range(0, in_coords.shape[1], 8): # taking eight elements at a time from each row
                avx_c_real = AVX.make_float8(in_coords_real[i, j + 7],
                                             in_coords_real[i, j + 6],
                                             in_coords_real[i, j + 5],
                                             in_coords_real[i, j + 4],
                                             in_coords_real[i, j + 3],
                                             in_coords_real[i, j + 2],
                                             in_coords_real[i, j + 1],
                                             in_coords_real[i, j + 0])

                avx_c_imag = AVX.make_float8(in_coords_imag[i, j + 7],
                                             in_coords_imag[i, j + 6],
                                             in_coords_imag[i, j + 5],
                                             in_coords_imag[i, j + 4],
                                             in_coords_imag[i, j + 3],
                                             in_coords_imag[i, j + 2],
                                             in_coords_imag[i, j + 1],
                                             in_coords_imag[i, j + 0])
                
                # initializing z
                avx_z_real = AVX.float_to_float8( 0 )
                avx_z_imag = AVX.float_to_float8( 0 )
                # initializing supporting counters and constants
                avx_iter = AVX.float_to_float8( 0 )
                avx_iter_increment = AVX.float_to_float8( 1 )
                avx_barrier_constant = AVX.float_to_float8(4.0)
                avx_multi_constant = AVX.float_to_float8(2.0)
                avx_comparator_constant = AVX.float_to_float8(-1.0)

                for iter in range(max_iterations):
                    mag_z = magnitude_squared_avx( avx_z_real, avx_z_imag )
                    # ensure only values that satisfy magnitude barrier continue being iterated
                    mask = AVX.less_than( mag_z, avx_barrier_constant )
                    # if all eight values are past magnitude barrier, stop computation
                    avx_check = AVX.signs( AVX.greater_than( mask, avx_comparator_constant ) )
                    if avx_check == 255:
                        break
                    
                    avx_iter = AVX.add( avx_iter, AVX.bitwise_and(mask, avx_iter_increment) )
                    
                    # compute new values of z; use temp variable to ensure both imaginary and 
                    # real values are updated in lock step
                    avx_z_real_temp = AVX.add( AVX.sub( AVX.mul( avx_z_real, avx_z_real ), \
                                                   AVX.mul( avx_z_imag, avx_z_imag ) ), avx_c_real )
                    avx_z_imag = AVX.add( AVX.mul( AVX.mul( avx_z_real, avx_z_imag ), \
                                                        avx_multi_constant ), avx_c_imag )
                    avx_z_real = avx_z_real_temp
                    
                AVX.to_mem(avx_iter, &(out_counts[i, j]))
                

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
