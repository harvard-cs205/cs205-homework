import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

# helper function to map AVX values in counts to the appropriate positions in out_counts
cdef void counts_to_output(AVX.float8 counts, np.uint32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int k
    AVX.to_mem(counts, &(tmp_counts[0]))
    for k in range(8):
        out_counts[i, j+k] = <unsigned int>tmp_counts[k]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       AVX.float8 c_real, c_imag, z_real, z_imag, z_temp, iters, curr

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    with nogil:
        # multithreading over all rows
        for i in prange(in_coords.shape[0], num_threads=4, schedule='static', chunksize=1):
            # looping through every 8 columns (instruction-level parallelism)
            for j in xrange(0,in_coords.shape[1],8):
                # split c into real and imaginary parts
                c_real = AVX.make_float8((in_coords[i, j+7]).real,(in_coords[i, j+6]).real,(in_coords[i, j+5]).real,(in_coords[i, j+4]).real,(in_coords[i, j+3]).real,(in_coords[i, j+2]).real,(in_coords[i, j+1]).real,(in_coords[i, j]).real)
                c_imag = AVX.make_float8((in_coords[i, j+7]).imag,(in_coords[i, j+6]).imag,(in_coords[i, j+5]).imag,(in_coords[i, j+4]).imag,(in_coords[i, j+3]).imag,(in_coords[i, j+2]).imag,(in_coords[i, j+1]).imag,(in_coords[i, j]).imag)

                # make z into a float8 vector with value 0 across
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)

                # initialize AVX iterate counts to 0.0 and curr to 0.0
                iters = AVX.float_to_float8(0.0)
                		
                # iterate until loop is broken or max_iterations is reached
                for iter in range(max_iterations):
                    # create AVX register containing which elements have |z|^2 < 4
                    curr = AVX.less_than(AVX.add(AVX.mul(z_real, z_real),AVX.mul(z_imag, z_imag)), AVX.float_to_float8(4.0))
		    # increase iters for the appropriate elements
                    iters = AVX.add(iters, AVX.bitwise_and(curr, AVX.float_to_float8(1.0)))
		    
                    # check whether all signs are positive (i.e., all elements have been updated)
                    if AVX.signs(curr) == 0:
                        # break if all elements are updated 
                        break

                    # update z for the next iteration		
                    z_temp = z_real
                    z_real = AVX.sub(AVX.fmadd(z_real, z_real, c_real),AVX.mul(z_imag, z_imag))
                    z_imag = AVX.add(AVX.mul(AVX.float_to_float8(2.0),AVX.mul(z_temp, z_imag)), c_imag)

                # transfer values to out_counts
                counts_to_output(iters, out_counts, i, j)

