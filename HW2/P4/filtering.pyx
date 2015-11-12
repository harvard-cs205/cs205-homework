# Note: includes all my comments from reviewing the skeleton code

#cython: boundscheck=False, wraparound=False

cimport numpy as np
import numpy as np
from libc.math cimport sqrt
cimport cython
from cython.parallel import prange, parallel
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks

# Useful types
ctypedef np.float32_t FLOAT

cdef extern from "median9.h":
    FLOAT median9(FLOAT s0, FLOAT s1, FLOAT s2,
                  FLOAT s3, FLOAT s4, FLOAT s5,
                  FLOAT s6, FLOAT s7, FLOAT s8) nogil

# clamped pixel fetch
cdef inline FLOAT GETPIX(FLOAT[:, :] im, int i, int j) nogil:
    if i < 0:
        i = 0
    if i >= im.shape[0]:
        i = im.shape[0] - 1
    if j < 0:
        j = 0
    if j >= im.shape[1]:
        j = im.shape[1] - 1
    return im[i, j]

# Median filtering
cpdef median_3x3(FLOAT[:, :] input_image,
                 FLOAT[:, :] output_image,
                 int offset, unsigned int step):
    cdef:
        int i, j

    # Check that dimensions of input and output images match
    assert input_image.shape[0] == output_image.shape[0], "median requires same size images for input and output"
    assert input_image.shape[1] == output_image.shape[1], "median requires same size images for input and output"

    with nogil:
        
        # Starting point based on number of threads
        i = offset

        # Loop through all relevant rows (starting point based on # of threads)
        while i < input_image.shape[0]:

            # Loop through columns
            for j in range(input_image.shape[1]):

                # Calculate median of 3x3 block surrounding pixel
                output_image[i, j] = median9(GETPIX(input_image, i-1, j-1), GETPIX(input_image, i-1, j), GETPIX(input_image, i-1, j+1),
                                             GETPIX(input_image, i,   j-1), GETPIX(input_image, i,   j), GETPIX(input_image, i,   j+1),
                                             GETPIX(input_image, i+1, j-1), GETPIX(input_image, i+1, j), GETPIX(input_image, i+1, j+1))
            
            # Increment based on number of threads
            i += step
