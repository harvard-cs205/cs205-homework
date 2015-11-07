import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef AVX.float8 magnitude_squared(AVX.float8 z_real, AVX.float8 z_imag) nogil:
    return AVX.add(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))

cdef void counts_to_output(AVX.float8 counts,
                           np.uint32_t [:, :] out_counts,
                           int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int idx

    AVX.to_mem(counts, &(tmp_counts[0]))
    for idx in range(8):
        out_counts[i, j*8 + idx] = <np.uint32_t>tmp_counts[idx]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int num_threads,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       np.complex64_t c, z
       AVX.float8 counts, c_real, c_imag, z_real, z_real_tmp, z_imag, mask
       int nt = num_threads

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

    for i in prange(in_coords.shape[0], nogil=True, num_threads=nt, schedule='static', chunksize=1):
        for j in range(in_coords.shape[1]/8):
            # start all the counts at zero
            counts = AVX.float_to_float8(0.)

            # load real & imaginary parts into separate AVX float8s
            c_real = AVX.make_float8(in_coords[i, j*8 + 7].real,
                                     in_coords[i, j*8 + 6].real,
                                     in_coords[i, j*8 + 5].real,
                                     in_coords[i, j*8 + 4].real,
                                     in_coords[i, j*8 + 3].real,
                                     in_coords[i, j*8 + 2].real,
                                     in_coords[i, j*8 + 1].real,
                                     in_coords[i, j*8].real)
            c_imag = AVX.make_float8(in_coords[i, j*8 + 7].imag,
                                     in_coords[i, j*8 + 6].imag,
                                     in_coords[i, j*8 + 5].imag,
                                     in_coords[i, j*8 + 4].imag,
                                     in_coords[i, j*8 + 3].imag,
                                     in_coords[i, j*8 + 2].imag,
                                     in_coords[i, j*8 + 1].imag,
                                     in_coords[i, j*8].imag)

            # start z at zero
            z_real = AVX.float_to_float8(0.)
            z_imag = AVX.float_to_float8(0.)

            # continue to max iteration count
            for iter in range(max_iterations):
                # if z magnitude is below threshold, increase count
                mask = AVX.less_than(magnitude_squared(z_real, z_imag),
                                     AVX.float_to_float8(4.))
                counts = AVX.add(counts,
                                 AVX.bitwise_and(mask,
                                                 AVX.float_to_float8(1.)))

                # if none of the 8 are below threshold, break
                if AVX.signs(mask) == 0:
                    break

                # calculate new z
                z_real_tmp = AVX.add(AVX.sub(AVX.mul(z_real, z_real),
                                             AVX.mul(z_imag, z_imag)),
                                     c_real)
                z_imag = AVX.add(AVX.mul(AVX.float_to_float8(2.),
                                         AVX.mul(z_real, z_imag)),
                                 c_imag)
                z_real = z_real_tmp

            # write counts to output array
            counts_to_output(counts, out_counts, i, j)



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
