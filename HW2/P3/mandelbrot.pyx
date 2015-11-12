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
       float [:, :] in_real, in_imag
       float bridge[8]
       AVX.float8 counts, c_real, c_imag, z_real, z_imag, magnitude, threshold, less_than_mask, tmp, ones

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

    in_real = np.real(in_coords) # real part of complex number
    in_imag = np.imag(in_coords) # imaginary part

    threshold = AVX.float_to_float8(4) # break threshold condition
    ones = AVX.float_to_float8(1)

    with nogil:
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=1):
            # loop over while processing 8 floats at a time
            for j in range(0, in_coords.shape[1], 8):
                counts = AVX.float_to_float8(0) # initial iteration counts

                # initialize real and imaginary part for c as float8 value
                c_real = AVX.make_float8(
                  in_real[i , j + 7], in_real[i , j + 6],
                  in_real[i , j + 5], in_real[i , j + 4],
                  in_real[i , j + 3], in_real[i , j + 2],
                  in_real[i , j + 1], in_real[i , j + 0])
                c_imag = AVX.make_float8(
                  in_imag[i , j + 7], in_imag[i , j + 6],
                  in_imag[i , j + 5], in_imag[i , j + 4],
                  in_imag[i , j + 3], in_imag[i , j + 2],
                  in_imag[i , j + 1], in_imag[i , j + 0])

                # initialize real and imaginary part for z as float8 value
                z_real = AVX.float_to_float8(0)
                z_imag = AVX.float_to_float8(0)

                for iter in range(max_iterations):
                    # check magnitude > 4 
                    # magnitude = z_real * z_real + z_imag * z_imag
                    magnitude = AVX.fmadd(z_real, z_real, AVX.mul(z_imag, z_imag))

                    # get a mask for each value in the float8 whether it is
                    # less than threshold = 4.
                    less_than_mask = AVX.less_than(magnitude, threshold)

                    # perform bitwise-and so that values which already pass threshold
                    # don't add to the iter counts any further.
                    counts = AVX.add(counts, AVX.bitwise_and(less_than_mask, ones))

                    if AVX.signs(less_than_mask) == 0:
                        break
                    
                    # compute z * z + c
                    # where z = z_real + i * z_imag
                    # where c = c_real + i * c_imag
                    tmp = AVX.add(AVX.fmsub(z_real, z_real, AVX.mul(z_imag, z_imag)), c_real)
                    z_imag = AVX.add(AVX.fmadd(z_real, z_imag, AVX.mul(z_real, z_imag)), c_imag)
                    z_real = tmp
                
                AVX.to_mem(counts, &(bridge[0]))
                for k in range(8):
                    out_counts[i, j + k] = <np.uint32_t>bridge[k]



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
