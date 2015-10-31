cimport numpy as np
cimport cython
import numpy as np
cimport AVX
from cython.parallel import prange

# Facilitate writing of counts to output
cdef void counts_to_output(AVX.float8 results, np.uint32_t[:, :] out_counts, int i, int j) nogil:
    cdef:
        float outValues[8]
        int idx
    AVX.to_mem(results, &(outValues[0]))
    for idx in range(8):
        out_counts[i,j+idx] = <np.uint32_t> outValues[idx]

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
        int i, j, iter
        AVX.float8 c_real, c_imag, z_real, z_imag, temp_real, temp_imag, magnitude, ones, fours, mask, results

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
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=1):
            for j in range(0,in_coords.shape[1],8):
                c_real = AVX.make_float8((in_coords[i,j+7]).real,
                                    (in_coords[i,j+6]).real,
                                    (in_coords[i,j+5]).real,
                                    (in_coords[i,j+4]).real,
                                    (in_coords[i,j+3]).real,
                                    (in_coords[i,j+2]).real,
                                    (in_coords[i,j+1]).real,
                                    (in_coords[i,j]).real)
                c_imag = AVX.make_float8((in_coords[i,j+7]).imag,
                                    (in_coords[i,j+6]).imag,
                                    (in_coords[i,j+5]).imag,
                                    (in_coords[i,j+4]).imag,
                                    (in_coords[i,j+3]).imag,
                                    (in_coords[i,j+2]).imag,
                                    (in_coords[i,j+1]).imag,
                                    (in_coords[i,j]).imag)
                
                # Mandelbrot values that are updated every iteration
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)

                # Iteration counts for each of the 8 values
                results = AVX.float_to_float8(0.0)

                # Some constant values for easy calculation
                ones = AVX.float_to_float8(1.0)
                fours = AVX.float_to_float8(4.0)

                for iter in range(max_iterations):
                    # Increment counts only when mask is at 1 which means 
                    results = AVX.add(results, AVX.bitwise_and(mask,ones))
                    # Mask is all 1's when magnitude is less than 4, all 0's when magnitude more than 4
                    magnitude = AVX.fmadd(z_real, z_real, AVX.mul(z_imag, z_imag))
                    mask = AVX.less_than(magnitude, fours)
                    
                    # Break when mask is all 0's
                    if (AVX.signs(mask) == 0):
                        break

                    # Update real and imaginary parts of z, as well as magnitude
                    temp_real = AVX.fmsub(z_real,z_real,AVX.mul(z_imag,z_imag))
                    temp_imag = AVX.fmadd(z_real,z_imag,AVX.mul(z_real,z_imag))
                    z_real  = AVX.add(temp_real,c_real)
                    z_imag  = AVX.add(temp_imag,c_imag)

                counts_to_output(results, out_counts, i, j)


















