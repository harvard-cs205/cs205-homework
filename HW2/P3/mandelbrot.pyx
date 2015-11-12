import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

# Stores iteration counts to memory
cdef void counts_to_output(AVX.float8 iterations, np.uint32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float temp_counts[8]
        int col
    AVX.to_mem(iterations, &(temp_counts[0]))
    for col in xrange(8):
        out_counts[i, j+col] = <np.uint32_t> temp_counts[col]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
        int i, j, iter
        AVX.float8 coords_real, coords_imag, z_real, z_imag, z2_real, z_real_imag, z2_imag, magnitude, bitmask, iterations, one_bits, max_mag
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
        # Set the limit of the mandelbrot function
        max_mag = AVX.float_to_float8(4.0)

        # Set one floats to add to iterations for coordinates
        one_bits = AVX.float_to_float8(1.0)

        for i in prange(in_coords.shape[0], num_threads=4, schedule='static', chunksize=1):
            for j in range(0, in_coords.shape[1], 8):
                # Grab imaginary coordinates 
                coords_imag = AVX.make_float8(in_coords[i,j+7].imag,
                                              in_coords[i,j+6].imag,
                                              in_coords[i,j+5].imag,
                                              in_coords[i,j+4].imag,
                                              in_coords[i,j+3].imag,
                                              in_coords[i,j+2].imag,
                                              in_coords[i,j+1].imag,
                                              in_coords[i,j].imag)
                
                # Grab real coordinates
                coords_real = AVX.make_float8(in_coords[i,j+7].real,
                                              in_coords[i,j+6].real,
                                              in_coords[i,j+5].real,
                                              in_coords[i,j+4].real,
                                              in_coords[i,j+3].real,
                                              in_coords[i,j+2].real,
                                              in_coords[i,j+1].real,
                                              in_coords[i,j].real)
                
                # Store count of iterations for each coordinate
                iterations = AVX.float_to_float8(0.0)

                # Store magnitude of each coordinate
                magnitude = AVX.float_to_float8(0.0)

                # Record which coordinates have reached their magnitude
                bitmask = AVX.float_to_float8(0.0)

                # Record the z values, separated by real and imaginary forms
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)

                for iter in range(max_iterations):
                    # Add to iterations count if magnitude is less than 4
                    iterations = AVX.add(iterations, AVX.bitwise_and(bitmask, one_bits))

                    # Calculate real and imaginary z values (z = z^2 + c)
                    z2_real = AVX.mul(z_real, z_real)
                    z2_imag = AVX.mul(z_imag, z_imag)
                    z_real_imag = AVX.add(AVX.mul(z_real, z_imag), AVX.mul(z_real, z_imag))
                    z_real = AVX.add(AVX.sub(z2_real, z2_imag), coords_real)
                    z_imag = AVX.add(z_real_imag, coords_imag)

                    # Calculate magnitude
                    magnitude = AVX.add(z2_real, z2_imag)

                    # Check if magnitudes are less than four
                    bitmask = AVX.less_than(magnitude, max_mag)

                    # If all magnitudes are >= 4, break and continue to next 8 coordinates
                    if not AVX.signs(bitmask):
                        break
                
                # Output iterations
                counts_to_output(iterations, out_counts, i, j)

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
