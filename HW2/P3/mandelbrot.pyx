import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


#Converts 8 floats to ints and writes them to output
cdef void counts_to_output(AVX.float8 counts,
                      np.uint32_t[:, :] out_counts,
                      int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int k
    AVX.to_mem(counts, &tmp_counts[0])
    for k in range(8):
        out_counts[i, j+k] = <int>tmp_counts[k]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int threads, 
                 int max_iterations):
    cdef:
       int i, j, signs_sum
       AVX.float8 four, two, one, z_real, z_imag, c_real, c_imag, iters, max_iters8, z_real_sq, z_imag_sq, mag_sq, less_than_4_mask, increment
       float [:, :] in_coords_real, in_coords_imag

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

    #Separate out real and imaginary parts of array
    in_coords_real = np.asarray(in_coords).real
    in_coords_imag = np.asarray(in_coords).imag
    
    with nogil:
        #Values used repeatedly in AVX calculations
        four = AVX.float_to_float8(4.0)
        two = AVX.float_to_float8(2.0)
        one = AVX.float_to_float8(1.0)
        max_iters8 = AVX.float_to_float8(max_iterations*1.0)
        #Iterate over rows in parallel
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=threads):
            #Iterate over columns using AVX
            for j in range(0, in_coords.shape[1], 8):
                c_real = AVX.make_float8(in_coords_real[i, j+7], in_coords_real[i, j+6], in_coords_real[i, j+5], in_coords_real[i, j+4], in_coords_real[i, j+3], in_coords_real[i, j+2], in_coords_real[i, j+1], in_coords_real[i, j])
                c_imag = AVX.make_float8(in_coords_imag[i, j+7], in_coords_imag[i, j+6], in_coords_imag[i, j+5], in_coords_imag[i, j+4], in_coords_imag[i, j+3], in_coords_imag[i, j+2], in_coords_imag[i, j+1], in_coords_imag[i, j])
                #Initialize z and iteration count
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)
                iters = AVX.float_to_float8(0.0)
                #Breaks when any iteration reaches max
                while AVX.signs(AVX.less_than(iters, max_iters8)) == 255:
                    #Check if magnitude of z >= 4
                    z_real_sq = AVX.mul(z_real, z_real)
                    z_imag_sq = AVX.mul(z_imag, z_imag)
                    mag_sq = AVX.add(z_real_sq, z_imag_sq)
                    less_than_4_mask = AVX.greater_than(four, mag_sq)
                    #Returns true when all values >= 4
                    signs_sum = AVX.signs(less_than_4_mask)
                    if signs_sum == 0:
                        break
                    # (a + bi)(a + bi) + (c + di) = a^2 + b^2*i^2 + 2abi + c + di
                    # = a^2 - b^2 + c + i(2ab + d)
                    z_imag = AVX.add(AVX.mul(AVX.mul(z_real, z_imag), two), c_imag)
                    z_real = AVX.add(AVX.sub(z_real_sq, z_imag_sq), c_real)
                    #Increment only those always and currently < 4
                    increment = AVX.bitwise_and(less_than_4_mask, one)
                    iters = AVX.add(iters, increment)
                #Write to output
                counts_to_output(iters, out_counts, i, j)


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
