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
                 int num_threads,
                 int max_iterations=511):
    cdef:
       int i, j, iter, x
       np.complex64_t c, z
       AVX.float8 four = AVX.float_to_float8(4.0)
       AVX.float8 two = AVX.float_to_float8(2.0)
       AVX.float8 one = AVX.float_to_float8(1.0)
       AVX.float8 zero = AVX.float_to_float8(0.0)
       AVX.float8 iter_counts = AVX.float_to_float8(1.0)
       AVX.float8 c_real, c_imag, z_real, z_imag, old_z_real, old_z_imag, under_four, magnitude
       float out_vals[8]

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

        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=num_threads):
            
            # we want every eighth number
            for j in xrange(0, in_coords.shape[1], 8):
                # reset iteration counts
                iter_counts = AVX.float_to_float8(1.0)

                # we're computing values eight at a time, so we create a float8
                # with the current and next 7 values. one float8 will hold the
                # 8 real parts, and one will hold the 8 imaginary parts
                
                c_real = AVX.make_float8(in_coords[i, (j + 7)].real,
                                         in_coords[i, (j + 6)].real,
                                         in_coords[i, (j + 5)].real,
                                         in_coords[i, (j + 4)].real,
                                         in_coords[i, (j + 3)].real,
                                         in_coords[i, (j + 2)].real,
                                         in_coords[i, (j + 1)].real,
                                         in_coords[i, (j + 0)].real)

                c_imag = AVX.make_float8(in_coords[i, (j + 7)].imag,
                                         in_coords[i, (j + 6)].imag,
                                         in_coords[i, (j + 5)].imag,
                                         in_coords[i, (j + 4)].imag,
                                         in_coords[i, (j + 3)].imag,
                                         in_coords[i, (j + 2)].imag,
                                         in_coords[i, (j + 1)].imag,
                                         in_coords[i, (j + 0)].imag)


                z_real, z_imag = AVX.float_to_float8(0.0), AVX.float_to_float8(0.0)


                for iter in range(max_iterations):


                    # store the old values for use in computing the new ones
                    old_z_real = z_real
                    old_z_imag = z_imag

                    # update z
                    # z = a + bi
                    # c = c + ei
                    # z = (a + bi)(a + bi) + (c + di)
                    # z = a^2 + 2(a * bi) + (bi)^2 + c + di
                    # z_real = a^2 - b^2 + c
                    # z_imag = 2(a * bi) + di

                    z_real = AVX.add(AVX.sub(AVX.mul(old_z_real, old_z_real), AVX.mul(old_z_imag, old_z_imag)), c_real)
                    z_imag = AVX.add(AVX.mul(AVX.mul(old_z_real, old_z_imag), two), c_imag)

                    # magnitude = z_real^2 + z_imag^2
                    magnitude = AVX.add(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))

                    # returns a float8, where each position is a 0 if magnitude >= 4 and -1 if magnitude < 4
                    under_four = AVX.less_than(magnitude, four)

                    # and these values so that all -1's are 1's and 0's are 0's, and then add them to iter_counts
                    # i.e., add 1 to iter_counts where magnitude < 4
                    iter_counts = AVX.add(iter_counts, AVX.bitwise_and(under_four, one))

                    # if all under_four are 0's, then everything is over 4 and we should stop
                    if AVX.signs(under_four) == 0:
                        break

                AVX.to_mem(iter_counts, &(out_vals[0]))

                for x in xrange(8):                    
                    out_counts[i, j + x] = <int> out_vals[x]



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
