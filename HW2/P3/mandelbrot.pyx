import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import parallel, prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef void counts_to_output(AVX.float8 counts,
                           np.uint32_t [:, :] out_counts,
                           int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int k

    AVX.to_mem(counts, &(tmp_counts[0]))
    for k in range(8):
        out_counts[i, 8 * j + k] = <unsigned int>(tmp_counts[k])

cdef void print_test(AVX.float8 test) nogil:
    cdef:
        float tmp_test[8]

    AVX.to_mem(test, &(tmp_test[0]))
    with gil:
        print tmp_test

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511, int num_threads=4):
    cdef:
       int i, j, iter
       np.complex64_t c, z

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    with nogil, parallel(num_threads=num_threads):
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1):
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
cpdef mandelbrot_avx(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511, int num_threads=4):
    cdef:
       int i, j, iter
       AVX.float8 avx_magnitude_reference, avx_real_z, avx_imag_z, avx_real, avx_imag, avx_ones, avx_count, avx_real_square, avx_imag_square, avx_magnitude, avx_comp, avx_twos
       np.float32_t[:, :] real, imag
       np.complex64_t [:] c

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    real = np.real(in_coords)
    imag = np.imag(in_coords)
    avx_magnitude_reference = AVX.float_to_float8(4)
    avx_ones = AVX.float_to_float8(1)
    avx_twos = AVX.float_to_float8(2)

    with nogil, parallel(num_threads=num_threads):
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1):
            for j in range(in_coords.shape[1] / 8):
                avx_real = AVX.make_float8(real[i, 8 * j + 7],
                                           real[i, 8 * j + 6],
                                           real[i, 8 * j + 5],
                                           real[i, 8 * j + 4],
                                           real[i, 8 * j + 3],
                                           real[i, 8 * j + 2],
                                           real[i, 8 * j + 1],
                                           real[i, 8 * j + 0])
                avx_imag = AVX.make_float8(imag[i, 8 * j + 7],
                                           imag[i, 8 * j + 6],
                                           imag[i, 8 * j + 5],
                                           imag[i, 8 * j + 4],
                                           imag[i, 8 * j + 3],
                                           imag[i, 8 * j + 2],
                                           imag[i, 8 * j + 1],
                                           imag[i, 8 * j + 0])
                avx_real_z = AVX.float_to_float8(0)
                avx_imag_z = AVX.float_to_float8(0)
                avx_count = AVX.float_to_float8(0)
                for iter in range(max_iterations):
                    avx_real_square = AVX.mul(avx_real_z, avx_real_z)
                    avx_imag_square = AVX.mul(avx_imag_z, avx_imag_z)

                    avx_magnitude = AVX.add(avx_real_square,
                                            avx_imag_square)
                    avx_comp = AVX.less_than(avx_magnitude,
                                             avx_magnitude_reference)
                    # Final criterion
                    if AVX.signs(avx_comp) == 0:
                        break

                    # Incrementing the count only for the magnitude below 4
                    avx_count = AVX.add(avx_count, AVX.bitwise_and(avx_comp,
                                        avx_ones))
                    # Updating the value only for the magnitude below 4
                    avx_imag_z = AVX.add(AVX.mul(AVX.mul(avx_real_z,
                                                 avx_imag_z), avx_twos),
                                         avx_imag)
                    avx_real_z = AVX.add(AVX.sub(avx_real_square,
                                         avx_imag_square), avx_real)
                counts_to_output(avx_count, out_counts, i, j)


# An example using AVX instructions
cpdef example_sqrt_8(np.complex64_t[:] values):
    cdef:
        AVX.float8 avxval, avx_magnitude_reference, avx_ones
        float out_vals[8]
        float [:] out_view = out_vals
        np.float32_t [:] real
        np.uint32_t [:, :] out_counts

    assert values.shape[0] == 8

    # Note that the order of the arguments here is opposite the direction when
    # we retrieve them into memory.
    real = np.real(values)
    out_counts = np.zeros((1, 8)).astype(np.uint32)
    with nogil:
        avx_magnitude_reference = AVX.float_to_float8(4)
        avx_ones = AVX.float_to_float8(1)
        avxval = AVX.make_float8(real[7],
                                 real[6],
                                 real[5],
                                 real[4],
                                 real[3],
                                 real[2],
                                 real[1],
                                 real[0])
        counts_to_output(avxval, out_counts, 0, 0)

    avxval = AVX.bitwise_and(AVX.less_than(avxval, avx_magnitude_reference), avx_ones)
    AVX.to_mem(avxval, &(out_vals[0]))
    print AVX.signs(avxval)
    print AVX.signs(AVX.float_to_float8(-1))

    print('out_vals is {}'.format(out_vals))
    print('out_view is {}'.format(out_view))

    print('out_counts is {}'.format(out_counts))
    return np.array(out_view)
