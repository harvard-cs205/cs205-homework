import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

cdef void counts_to_output(AVX.float8 answer, 
                           np.uint32_t [:, :] out_counts, 
                           int i, 
                           int j) nogil:
    cdef:
        float out_vals[8]
        int idx

    AVX.to_mem(answer, &(out_vals[0]))
    for idx in range(8):
        out_counts[i, j + idx] = <np.uint32_t> out_vals[idx]

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
        int i, j, iter
        np.complex64_t c, z
        float [:, :] in_coords_real, in_coords_imag
        AVX.float8 c_real, c_imag, z_real, z_imag, temp_real, temp_imag, magnitude, mask, one, two, four, ans

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

    in_coords_real = np.real(in_coords)
    in_coords_imag = np.imag(in_coords)

    with nogil:
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=4):
            for j in xrange(0, in_coords.shape[1], 8):

                c_real = AVX.make_float8(in_coords_real[i, j + 7],
                                         in_coords_real[i, j + 6],
                                         in_coords_real[i, j + 5],
                                         in_coords_real[i, j + 4],
                                         in_coords_real[i, j + 3],
                                         in_coords_real[i, j + 2],
                                         in_coords_real[i, j + 1],
                                         in_coords_real[i, j])

                c_imag = AVX.make_float8(in_coords_imag[i, j + 7],
                                         in_coords_imag[i, j + 6],
                                         in_coords_imag[i, j + 5],
                                         in_coords_imag[i, j + 4],
                                         in_coords_imag[i, j + 3],
                                         in_coords_imag[i, j + 2],
                                         in_coords_imag[i, j + 1],
                                         in_coords_imag[i, j])

                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)
                one = AVX.float_to_float8(1.0)
                two = AVX.float_to_float8(2.0)
                four = AVX.float_to_float8(4.0)

                ans = AVX.float_to_float8(0.0)

                for iter in range(max_iterations):

                    # add to ans if magnitude < 4
                    ans = AVX.add(ans, AVX.bitwise_and(mask, one))

                    #Compute magnitude of z and check whether it passes threshold
                    magnitude = AVX.add(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))
                    mask = AVX.less_than(magnitude, four)

                    if AVX.signs(mask) == 0:
                        break

                    # # if AVX.signs(AVX.greater_than(ans, AVX.float_to_float8(0.0))) == 255:
                    #     # break

                    temp_real = AVX.sub(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))
                    temp_imag = AVX.add(AVX.mul(z_real, z_imag), AVX.mul(z_real, z_imag))

                    z_real  = AVX.add(temp_real,c_real)
                    z_imag  = AVX.add(temp_imag,c_imag)

                counts_to_output(ans, out_counts, i, j)


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
