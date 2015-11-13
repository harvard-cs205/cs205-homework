import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange
cdef: 
    int NUM_THREADS = 4

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot_thread(np.complex64_t [:, :] in_coords,
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
        for i in prange(in_coords.shape[0], num_threads=NUM_THREADS, schedule='static', chunksize=1):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot_instruction(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
        int i, j, k, iter
        np.complex64_t c, z
        AVX.float8 c_real, c_imag, z_real, z_imag, z_real_sq, z_imag_sq, tmp_real, tmp_imag
        AVX.float8 magnitudes_squared, mask, iter_count
        float out_vals[8]
        float [:] out_view = out_vals

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
        for i in prange(in_coords.shape[0], num_threads=NUM_THREADS, schedule='static', chunksize=1):
            for j in range(0, in_coords.shape[1], 8):
                iter_count = AVX.float_to_float8(0.)
                # convert data into float8
                c_real = AVX.make_float8(in_coords[i, j].real,
                                         in_coords[i, j+1].real,
                                         in_coords[i, j+2].real,
                                         in_coords[i, j+3].real,
                                         in_coords[i, j+4].real,
                                         in_coords[i, j+5].real,
                                         in_coords[i, j+6].real,
                                         in_coords[i, j+7].real)
                c_imag = AVX.make_float8(in_coords[i, j].imag,
                                         in_coords[i, j+1].imag,
                                         in_coords[i, j+2].imag,
                                         in_coords[i, j+3].imag,
                                         in_coords[i, j+4].imag,
                                         in_coords[i, j+5].imag,
                                         in_coords[i, j+6].imag,
                                         in_coords[i, j+7].imag)
                z_real = AVX.float_to_float8(0.)
                z_imag = AVX.float_to_float8(0.)
                for iter in range(max_iterations):
                    # compute abs of float8
                    z_real_sq = AVX.mul(z_real, z_real)
                    z_imag_sq = AVX.mul(z_imag, z_imag)
                    magnitudes_squared = AVX.add(z_real_sq, z_imag_sq)
                    # check if abs of float8 is less greater than 4
                    mask = AVX.less_than(magnitudes_squared, AVX.float_to_float8(4.))
                    # if float8 all greater than 4
                    if not AVX.signs(mask): 
                        break
                    # increment counts where magnitude_squared is less than 4
                    iter_count = AVX.add(iter_count, AVX.bitwise_and(mask, AVX.float_to_float8(1.)))
                    # z = z * z + c
                    # z_real = z_real**2 - z_imag**2 + c_real
                    # z_imag = 2 * z_real * z_imag * i + c_imag * i
                    tmp_real = AVX.add(AVX.sub(z_real_sq, z_imag_sq), c_real)
                    tmp_imag = AVX.add(AVX.mul(AVX.float_to_float8(2.), AVX.mul(z_real, z_imag)), c_imag)
                    z_real, z_imag = tmp_real, tmp_imag
                
                AVX.to_mem(iter_count, &(out_vals[0]))
                for k in range(8):
                    out_counts[i, j+k] = int(out_view[k])

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
