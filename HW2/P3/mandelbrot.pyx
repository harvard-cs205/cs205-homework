import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

cdef void counts_to_output(AVX.float8 counts,
                      np.uint32_t[:, :] out_counts,
                      int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int k
        
    AVX.to_mem(counts, &(tmp_counts[0]))
    for k in range(8):
        out_counts[i,j+k] = int(tmp_counts[k])


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       AVX.float8 c_real, c_imag, z_real, z_imag, counts, eight_ones, z_squared, mask, limit, eight_ones_masked, z_real_temp, z_imag_temp
       np.ndarray[np.float32_t, ndim=2] in_coords_real, in_coords_imag,
       float tmp_counts[8]

    # split complex numbers in real and imaginary parts
    in_coords_real = np.real(in_coords)
    in_coords_imag = np.imag(in_coords)

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    with nogil:
        for i in prange(in_coords.shape[0], num_threads=2, schedule='static', chunksize=1):
            for j in range(0, in_coords.shape[1], 8):

                # initialize variables
                c_real = AVX.make_float8(in_coords_real[i, j+7], in_coords_real[i, j+6],
                                         in_coords_real[i, j+5], in_coords_real[i, j+4],
                                         in_coords_real[i, j+3], in_coords_real[i, j+2],
                                         in_coords_real[i, j+1], in_coords_real[i, j+0])

                c_imag = AVX.make_float8(in_coords_imag[i, j+7], in_coords_imag[i, j+6],
                                         in_coords_imag[i, j+5], in_coords_imag[i, j+4],
                                         in_coords_imag[i, j+3], in_coords_imag[i, j+2],
                                         in_coords_imag[i, j+1], in_coords_imag[i, j+0])

                z_real = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
                z_imag = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
                counts = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
                eight_ones = AVX.make_float8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                limit = AVX.make_float8(4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0)

                for iter in range(max_iterations):
                    
                    # calculate z^2
                    z_squared = AVX.add(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))

                    # get mask
                    mask = AVX.less_than(z_squared, limit)

                    # check when to break out of the loop
                    if (AVX.signs(mask) == 0): break

                    # apply mask on array of eight ones
                    eight_ones_masked = AVX.bitwise_and(mask, eight_ones)

                    # update counts 
                    counts = AVX.add(counts, AVX.bitwise_and(mask, eight_ones_masked)) 

                    # get new z_real and z_imag
                    z_real_temp = AVX.sub(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))
                    z_imag_temp= AVX.add(AVX.mul(z_real, z_imag), AVX.mul(z_real, z_imag))
                    z_real = AVX.add(z_real_temp, c_real)
                    z_imag = AVX.add(z_imag_temp, c_imag)

                counts_to_output(counts, out_counts, i, j)
