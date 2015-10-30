import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

# A function that returns the magnitude squared of complex numbers 
# The inputs are the real and imaginary components of 8 complex numbers 
# stored as separate AVX.float8 values
cdef AVX.float8 magnitude_squared_float8(AVX.float8 zreal, AVX.float8 zimag) nogil:
    return AVX.fmadd(zreal, zreal, AVX.mul(zimag, zimag))

# Stores an AVX.float8 to a specified location in a 2D array of ints
cdef void counts_to_output(AVX.float8 iter_counts, np.uint32_t[:,:] out_counts, int i, int j_start) nogil:
    cdef:
        float tmp_vals[8]
        int j
    AVX.to_mem(iter_counts, &(tmp_vals[0]))
    for j in range(0, 8):
        out_counts[i, j_start + j] = <int>tmp_vals[j] + 1
        
#cdef void a_print(AVX.float8 avxval) nogil:
#    cdef:
#        float out_vals[8]
#    AVX.to_mem(avxval, &(out_vals[0]))
#    with gil:
#        print out_vals

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):

    # Define all the variables we wil need
    cdef:
        int i, j, iter
        AVX.float8 creal, cimag, zreal, zimag, zreal_tmp
        AVX.float8 zeros, ones, twos, fours
        AVX.float8 mask, iters, mags_squared
        
    # Initialize some AVX.float8s of specific integer floats
    zeros = AVX.float_to_float8(0.0)
    ones = AVX.float_to_float8(1.0)
    twos = AVX.float_to_float8(2.0)
    fours = AVX.float_to_float8(4.0)

    # Assertion checks on the inputs
    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    # Original code
    #with nogil:
    #    for i in range(in_coords.shape[0]):
    #        for j in range(in_coords.shape[1]):
    #            c = in_coords[i, j]
    #            z = 0
    #            for iter in range(max_iterations):
    #                if magnitude_squared(z) > 4:
    #                    break
    #                z = z * z + c
    #            out_counts[i, j] = iter
    
    with nogil:
        # Rows using Multithreading (run in parallel)
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=4):
            # Columns using AVX (run 8 simultanesouly using SIMD-AVX)
            for j in range(in_coords.shape[1]/8):
                
                # Intialize the iteration counts and zreal and zimag to zeros
                iters, zreal, zimag = zeros, zeros, zeros
                
                # Store the real components of 8 c's
                creal = AVX.make_float8((in_coords[i, (j*8 + 7)]).real,
                             (in_coords[i, (j*8 + 6)]).real,
                             (in_coords[i, (j*8 + 5)]).real,
                             (in_coords[i, (j*8 + 4)]).real,
                             (in_coords[i, (j*8 + 3)]).real,
                             (in_coords[i, (j*8 + 2)]).real,
                             (in_coords[i, (j*8 + 1)]).real,
                             (in_coords[i, (j*8 + 0)]).real)
                                
                # Store the imaginary components of 8 c's
                cimag = AVX.make_float8((in_coords[i, (j*8 + 7)]).imag,
                             (in_coords[i, (j*8 + 6)]).imag,
                             (in_coords[i, (j*8 + 5)]).imag,
                             (in_coords[i, (j*8 + 4)]).imag,
                             (in_coords[i, (j*8 + 3)]).imag,
                             (in_coords[i, (j*8 + 2)]).imag,
                             (in_coords[i, (j*8 + 1)]).imag,
                             (in_coords[i, (j*8 + 0)]).imag)
                
                for iter in range(max_iterations):
                    # Square and sum the real and imaginary components of z
                    mags_squared = magnitude_squared_float8(zreal, zimag)
                    
                    # Create a mask of the values of the squared magnitudes that are greater than four
                    mask = AVX.greater_than(fours, mags_squared)
                    
                    # Increment the iteration counts of pixels whose magnitude squared < 4
                    iters = AVX.add(iters, AVX.bitwise_and(mask, ones))
                    
                    # If all of the values in mags_squared > 4, then short-circuit 
                    if AVX.signs(mask) == 0:
                        break
                   
                    # Original:
                    #    z = z * z + c
                    # Complex number multiplication with AVX:
                    #    (a + bi)^2 = (a^2 - b^2) + (2ab)i
                    zreal_tmp = AVX.add(AVX.fmsub(zreal, zreal, AVX.mul(zimag, zimag)), creal)
                    zimag = AVX.fmadd(twos, AVX.mul(zreal, zimag), cimag)
                    zreal = zreal_tmp
                    
                # Write the iteration counts to the image
                counts_to_output(iters, out_counts, i, j*8)

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