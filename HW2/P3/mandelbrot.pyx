# Note: includes all my comments from reviewing the skeleton code
# AVX calculations with real/imaginary numbers adapted from https://github.com/skeeto/mandel-simd/blob/master/mandel_avx.c

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
cdef void counts_to_output(AVX.float8 counts,
                      np.uint32_t [:, :] out_counts,
                      int i, int j) nogil:
    cdef:
        float counts_temp[8]
        int step

    # Store result
    AVX.to_mem(counts, counts_temp)

    # Copy to output array
    for step in prange(8):
        out_counts[i, j + step] = <np.uint32_t> counts_temp[step]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot_without_avx(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter # Used to iterate over ranges
       np.complex64_t c, z # c = single point in in_coords; z = used in calculation

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    # Iterate over rows
    for i in prange(in_coords.shape[0], nogil=True, schedule='static', chunksize=1, num_threads=4):

        # Iterate over columns (will always be a multiple of 8 - checked above)
        for j in range(in_coords.shape[1]):

            # Identify coordinates for given row, column (i, j)
            c = in_coords[i, j]

            # Initialize z
            z = 0

            # Iterate until z**2 > 4
            for iter in xrange(max_iterations):
                
                # Test exit condition
                if magnitude_squared(z) > 4:
                    break

                # Update z
                z = z * z + c

            # Store result
            out_counts[i, j] = iter

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot_with_avx(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter # Used to iterate over ranges
       float counts_temp[8] # Used to convert result from AVX to np
       np.ndarray[np.float32_t, ndim=2] in_coords_real, in_coords_imag # Used to break up coords into real and imag components
       AVX.float8 c_real, c_imag, z_real, z_imag, z_real2, z_imag2, z_realimag, z2, counts, counts_add, exit_check, mask
       
    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    in_coords_real = np.real(in_coords)
    in_coords_imag = np.imag(in_coords)

    counts_add = AVX.float_to_float8(1.0)
    exit_check = AVX.float_to_float8(4.0)

    # Iterate over rows
    for i in prange(in_coords.shape[0], nogil=True, schedule='static', chunksize=1, num_threads=4):
        
        # Iterate over columns (will always be a multiple of 8 - checked above)
        for j in range(0, in_coords.shape[1], 8):

            # Identify coordinates for given row, column (i, j + 7)
            c_real = AVX.make_float8(in_coords_real[i, j+7], in_coords_real[i, j+6], in_coords_real[i, j+5], in_coords_real[i, j+4],
                                     in_coords_real[i, j+3], in_coords_real[i, j+2], in_coords_real[i, j+1], in_coords_real[i, j+0])

            c_imag = AVX.make_float8(in_coords_imag[i, j+7], in_coords_imag[i, j+6], in_coords_imag[i, j+5], in_coords_imag[i, j+4],
                                     in_coords_imag[i, j+3], in_coords_imag[i, j+2], in_coords_imag[i, j+1], in_coords_imag[i, j+0])
            
            # Initialize z
            z_real = AVX.float_to_float8(0.0)
            z_imag = AVX.float_to_float8(0.0)

            # Initialize counter
            counts = AVX.float_to_float8(1.0)

            # Iterate for max_iterations (does not attempt to stop at z**2 > 4)
            for iter in range(max_iterations):
                
                # Update z = z * z + c
                z_real2 = AVX.mul(z_real, z_real)
                z_imag2 = AVX.mul(z_imag, z_imag)
                z_realimag = AVX.mul(z_real, z_imag)
                z_real = AVX.add(AVX.sub(z_real2, z_imag2), c_real)
                z_imag = AVX.add(AVX.add(z_realimag, z_realimag), c_imag)

                # Check exit condition
                z2 = AVX.add(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))

                # Add to counter only if z**2 < 4
                mask = AVX.bitwise_and(AVX.less_than(z2, exit_check), counts_add)

                if AVX.signs(AVX.greater_than(mask, AVX.float_to_float8(0.0)))==0:
                    break

                counts = AVX.add(counts, mask)

            # Store result
            counts_to_output(counts, out_counts, i, j)

# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval, tmp, mask
        float out_vals[8] # Array with 8 values
        float [:] out_view = out_vals # Array with 8 values

    assert values.shape[0] == 8 # Check that the length is as expected

    # Convert np array into avx array
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

    # Apply (positive) square root to each element
    avxval = AVX.sqrt(avxval)

    # Mask will be true where 2.0 < avxval
    mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)

    # Invert mask and select off values, so should be 2.0 >= avxval
    avxval = AVX.bitwise_andnot(mask, avxval)

    # Convert from avx array to array
    AVX.to_mem(avxval, &(out_vals[0]))

    # Return as np array
    return np.array(out_view)
