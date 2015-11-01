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
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter, k
       np.complex64_t c, z
       AVX.float8 c_real_avxval, c_imag_avxval, z_real_avxval, z_imag_avxval, magnitude_squared_z, mask, filtered_avxval, counts_avxval, all_one, z_square_real, z_square_imag
       float temp_counts[8] # for out_counts type casting

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
        for i in prange(in_coords.shape[0], num_threads=4, schedule='static', chunksize=1):
        #for i in range(in_coords.shape[0]):
            for j in xrange(0, in_coords.shape[1], 8):
                #c = in_coords[i, j]
                c_real_avxval = AVX.make_float8((in_coords[i, j+7]).real,
                                                (in_coords[i, j+6]).real,
                                                (in_coords[i, j+5]).real,
                                                (in_coords[i, j+4]).real,
                                                (in_coords[i, j+3]).real,
                                                (in_coords[i, j+2]).real,
                                                (in_coords[i, j+1]).real,
                                                (in_coords[i, j]).real)
                c_imag_avxval = AVX.make_float8((in_coords[i, j+7]).imag,
                                                (in_coords[i, j+6]).imag,
                                                (in_coords[i, j+5]).imag,
                                                (in_coords[i, j+4]).imag,
                                                (in_coords[i, j+3]).imag,
                                                (in_coords[i, j+2]).imag,
                                                (in_coords[i, j+1]).imag,
                                                (in_coords[i, j]).imag)
                #z = 0
                z_real_avxval = AVX.float_to_float8(0.0)
                z_imag_avxval = AVX.float_to_float8(0.0)
                all_one = AVX.float_to_float8(1.0)
                counts_avxval = AVX.float_to_float8(0.0)
                

                for iter in range(max_iterations):
                #    if magnitude_squared(z) > 4:
                #        break
                #    z = z * z + c
                    
                    magnitude_squared_z = AVX.add(AVX.mul(z_real_avxval, z_real_avxval), AVX.mul(z_imag_avxval, z_imag_avxval))
                    mask = AVX.less_than(magnitude_squared_z, AVX.float_to_float8(4.0))
                    
                    if (AVX.signs(mask) == 0) :
                        break
                    
                    counts_avxval = AVX.add(counts_avxval, AVX.bitwise_and(mask, all_one))

                    # compute z = z * z + c
                    # if z = a + bi, c = x + yi
                    # z * z = (a^2 - b^2) + (2ab)i
                    # z_square_real = (a^2 - b^2)
                    # z_square_imag = (2ab)
                    z_square_real = AVX.sub(AVX.mul(z_real_avxval, z_real_avxval), AVX.mul(z_imag_avxval, z_imag_avxval))
                    z_square_imag = AVX.add(AVX.mul(z_real_avxval, z_imag_avxval), AVX.mul(z_real_avxval, z_imag_avxval))
                    
                    # z_real = (a^2 - b^2) + x
                    # z_imag = (2ab) + y
                    z_real_avxval = AVX.add(z_square_real,c_real_avxval)
                    z_imag_avxval = AVX.add(z_square_imag,c_imag_avxval)


                #out_counts[i, j] = iter
                # Since we cannot write to out_counts directly,
                # we have to cast the type to (uint32_t)
                AVX.to_mem(counts_avxval, &(temp_counts[0]))
                for k in range(8):
                    out_counts[i, j + k] = <np.uint32_t>temp_counts[k]




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
