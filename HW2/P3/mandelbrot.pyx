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

# the original mandelbrot, with parallel.prange, added new argument n_threads
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511,
                 int num_threads = 1):
    cdef:
        int i, j, iter
        np.complex64_t c, z
        int n_threads = num_threads
 
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
        # here prange
        for i in prange(in_coords.shape[0], schedule = 'static', chunksize = 1, num_threads = n_threads):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter

# the function we paste our iteration results to a numpy array
cdef void counts_to_outcounts(AVX.float8 counts,
                 np.uint32_t [:, :] out_counts,
                 int i,
                 int j) nogil:
    cdef:
        float [8] result
        int ii

    AVX.to_mem(counts, &(result[0]))
    
    for ii in range(8):
         out_counts[i, j*8 + ii] = int(result[7 - ii])

# the float_8 version, added new argument num_threads
cpdef mandelbrot_float8(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511,
                 int num_threads = 1):
    cdef:
        int i, j, ii, iter
        int n_threads = num_threads
        float [8] result
        np.complex64_t c, z
        np.float32_t [:, :] in_coords_real = np.real(in_coords), in_coords_imag = np.imag(in_coords)
        AVX.float8 c_avxvecreal, c_avxvecimag, z_avxvecreal, z_avxvecimag, temp_avxvecreal, temp_avxvecimag
        AVX.float8 alltrues = AVX.float_to_float8(-1.0), allones = AVX.float_to_float8(1.0), counts = AVX.float_to_float8(0.0)
        AVX.float8 notfinish

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
        for i in prange(in_coords.shape[0], schedule = 'static', chunksize = 1, num_threads = n_threads):
            # devide the columns by 8
            for j in range(in_coords.shape[1] / 8):
                # create a float8 variable
                c_avxvecreal = AVX.make_float8(in_coords_real[i, j*8 + 0], in_coords_real[i, j*8 + 1], in_coords_real[i, j*8 + 2], in_coords_real[i, j*8 + 3], \
                                               in_coords_real[i, j*8 + 4], in_coords_real[i, j*8 + 5], in_coords_real[i, j*8 + 6], in_coords_real[i, j*8 + 7])
                c_avxvecimag = AVX.make_float8(in_coords_imag[i, j*8 + 0], in_coords_imag[i, j*8 + 1], in_coords_imag[i, j*8 + 2], in_coords_imag[i, j*8 + 3], \
                                               in_coords_imag[i, j*8 + 4], in_coords_imag[i, j*8 + 5], in_coords_imag[i, j*8 + 6], in_coords_imag[i, j*8 + 7])
                # the real part and imaginary part of z
                z_avxvecreal = AVX.float_to_float8(0.0)
                z_avxvecimag = AVX.float_to_float8(0.0)

                for iter in range(max_iterations):
                    # not finish is a float 8 vector use as the indicator of stop
                    notfinish = AVX.less_than(AVX.add(AVX.mul(z_avxvecreal, z_avxvecreal), AVX.mul(z_avxvecimag, z_avxvecimag)), AVX.float_to_float8(4.0))
                    # if all False, we stop
                    if AVX.signs(notfinish) & 255 == 0:
                         break
                    # otherwise we calculate z * z + c in two parts
                    temp_avxvecreal = AVX.add(AVX.add(AVX.mul(z_avxvecreal, z_avxvecreal), AVX.mul(AVX.mul(z_avxvecimag, z_avxvecimag), AVX.float_to_float8(-1.0))), c_avxvecreal)
                    temp_avxvecimag = AVX.add(AVX.mul(AVX.mul(z_avxvecreal, z_avxvecimag), AVX.float_to_float8(2.0)), c_avxvecimag)
                    z_avxvecreal = temp_avxvecreal
                    z_avxvecimag = temp_avxvecimag
                    # add a count, but with mask according to not finish
                    counts = AVX.add(counts, AVX.bitwise_and(notfinish, allones))
                # use the function defined above to paste back to out_counts
                counts_to_outcounts(counts, out_counts, i, j)
                # re-initialize counts for next iteration
                counts = AVX.float_to_float8(0.0)


# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval
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

    AVX.to_mem(avxval, &(out_vals[0]))

    return np.array(out_view)
