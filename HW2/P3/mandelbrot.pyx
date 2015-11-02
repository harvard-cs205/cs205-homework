import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

######################
#
# Submission by Kendrick Lo (Harvard ID: 70984997) for
# CS 205 - Computing Foundations for Computational Science (Prof. R. Jones)
# 
# Homework 2 - Problem 3
#
######################

@cython.boundscheck(False)
@cython.wraparound(False)
#############
#
# helper function added to update out_counts array
#
#############
cdef np.uint32_t [:, :] update_oc(np.uint32_t [:, :] out_counts, int i, 
                                  int j, AVX.float8 vals) nogil:
    cdef: 
        int k
        float tmp_counts[8]

    AVX.to_mem(vals, &(tmp_counts[0]))
    for k in range(8):
        out_counts[i, j+k] = <int>tmp_counts[k]

    return out_counts

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       int n_threads  # added to allow setting of number of threads

       np.float32_t [:, :] r_coords, i_coords  # store real and imag parts
       int n_8cols  # number of 8 column chunks

       AVX.float8 cr, ci, zr, zi  # c and z 
       AVX.float8 iter_ind  # iteration counter for individual pixels
       AVX.float8 magz, newzr, newzi  # temporary values associated with z
       AVX.float8 mask, tmp, new_iter_ind

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    # separate real and imaginary components
    r_coords = np.real(in_coords)
    i_coords = np.imag(in_coords)

    with nogil:

        n_threads = 4
        n_8cols = in_coords.shape[1] / 8  # break into chunks of 8 columns

        for i in prange(in_coords.shape[0], num_threads=n_threads,
                        schedule='static', chunksize=1):

            for j in range(n_8cols):

                cr = AVX.make_float8(r_coords[i, j*8+7],
                                     r_coords[i, j*8+6],
                                     r_coords[i, j*8+5],
                                     r_coords[i, j*8+4],
                                     r_coords[i, j*8+3],
                                     r_coords[i, j*8+2],
                                     r_coords[i, j*8+1],
                                     r_coords[i, j*8])

                ci = AVX.make_float8(i_coords[i, j*8+7],
                                     i_coords[i, j*8+6],
                                     i_coords[i, j*8+5],
                                     i_coords[i, j*8+4],
                                     i_coords[i, j*8+3],
                                     i_coords[i, j*8+2],
                                     i_coords[i, j*8+1],
                                     i_coords[i, j*8])

                zr = AVX.float_to_float8(0.0)
                zi = AVX.float_to_float8(0.0)
                iter_ind = AVX.float_to_float8(0.0)

                for iter in range(max_iterations):
                    
                    # calculate |Z|
                    magz = AVX.mul(zi, zi)
                    magz = AVX.fmadd(zr, zr, magz)  # zr * zr + (zi * zi)

                    # mask will be true where |Z| < 4
                    mask = AVX.less_than(magz, AVX.float_to_float8(4.0))
                    if AVX.signs(mask)==0:
                        break # no values less than 4 (i.e. all values >= 4)

                    # calculate potential changed values for z
                    newzr = AVX.fmadd(zr, zr, cr)
                    tmp = AVX.mul(zi, zi)
                    newzr = AVX.sub(newzr, tmp)  # zr * zr + cr - zi * zi
                    newzi = AVX.mul(AVX.float_to_float8(2.0), zr)
                    newzi = AVX.fmadd(newzi, zi, ci)  # 2 * zr * zi + ci

                    # increment iteration counter
                    new_iter_ind = AVX.add(iter_ind, AVX.float_to_float8(1.0))

                    #########
                    #
                    # apply mask
                    # (i.e. flow through new values for zr, zi, and 
                    #  iteration counter if masked, otherwise keep old values)
                    #
                    #########

                    tmp = AVX.bitwise_and(mask, newzr)
                    zr = AVX.bitwise_andnot(mask, zr)
                    zr = AVX.add(zr, tmp)  

                    tmp = AVX.bitwise_and(mask, newzi)
                    zi = AVX.bitwise_andnot(mask, zi)
                    zi = AVX.add(zi, tmp) 

                    tmp = AVX.bitwise_and(mask, new_iter_ind)
                    iter_ind = AVX.bitwise_andnot(mask, iter_ind)
                    iter_ind = AVX.add(iter_ind, tmp) 

                # update master grid with the iteration counts
                update_oc(out_counts, i, j*8, iter_ind)
