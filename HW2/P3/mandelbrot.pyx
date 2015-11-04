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
# this is original
cpdef mandelbrot_original(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       np.complex64_t c, z  
       
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
        for i in range(in_coords.shape[0]):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter




# this will be AVX version 
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, k, it
       np.float32_t [:, :] in_co_real, in_co_imag
       AVX.float8 real_c, imag_c, real_z, imag_z, tmp_real_z, tmp_imag_z, iters
       AVX.float8 mag_sq, comp_mask, fours, twos, ones, neg_ones
       # store the results 
       float out[8]

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    in_co_real = np.real(in_coords)
    in_co_imag = np.imag(in_coords)
    # vector of fours that we use in comparisons 
    fours = AVX.float_to_float8(4.0)
    twos = AVX.float_to_float8(2.0)
    ones = AVX.float_to_float8(1.0)
    neg_ones = AVX.float_to_float8(-1.0)
    with nogil:
        # parallelize
        for i in prange(in_coords.shape[0], num_threads=1, schedule='static', chunksize=1):
            # take every 8th j 
            for j in range(0, in_coords.shape[1], 8):
                real_c = AVX.make_float8(in_co_real[i, j+7],
                                          in_co_real[i, j+6],
                                          in_co_real[i, j+5],
                                          in_co_real[i, j+4],
                                          in_co_real[i, j+3],
                                          in_co_real[i, j+2],
                                          in_co_real[i, j+1],
                                          in_co_real[i, j])

                imag_c = AVX.make_float8(in_co_imag[i, j+7],
                                          in_co_imag[i, j+6],
                                          in_co_imag[i, j+5],
                                          in_co_imag[i, j+4],
                                          in_co_imag[i, j+3],
                                          in_co_imag[i, j+2],
                                          in_co_imag[i, j+1],
                                          in_co_imag[i, j])
                real_z = AVX.float_to_float8(0.0)
                imag_z = AVX.float_to_float8(0.0)
                # iteration for each i 
                iters = AVX.float_to_float8(0.0)
                for it in range(max_iterations):
                  # mag_sq = real^2 + imag^2
                  # = real* real + imag^2
                  mag_sq = AVX.mul(imag_z, imag_z)
                  mag_sq = AVX.fmadd(real_z, real_z, mag_sq)
                  # all 0000 > 0
                  # all 1111 < -1
                  comp_mask = AVX.less_than(mag_sq, fours)
                  # if it is less than four, we want to add one to the iteratiorns 
                  # if it is greater than four, add 0 
                  iters = AVX.add(iters, AVX.bitwise_and(comp_mask, ones))
                  if AVX.signs(comp_mask) == 0: 
                    break 

                  # update z 
                  # z = (a + bi)(a + bi) = (a^2 - b^2) + 2(ab)i 
                  tmp_real_z = AVX.mul(real_z, real_z)
                  tmp_real_z = AVX.sub(tmp_real_z, AVX.mul(imag_z, imag_z))
                  tmp_real_z = AVX.add(tmp_real_z, real_c)
                  tmp_imag_z = AVX.add(AVX.mul(twos, AVX.mul(real_z, imag_z)), imag_c)
                  real_z = tmp_real_z
                  imag_z = tmp_imag_z

                # copy the iters to out 
                AVX.to_mem(iters, &out[0])
                # copy out to out_counts 
                for k in range(8):
                  out_counts[i, j + k] = <int> out[k] 


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
