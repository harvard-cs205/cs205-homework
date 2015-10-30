cimport cython
import numpy as np
cimport numpy as np
cimport AVX
from cython.parallel import prange, threadid, parallel

#cdef extern from "AVX.h" nogil:
#    AVX.int8 _mm256_ctvps_epi32(AVX.float8)
#    void _mm256_storeu_si256 (AVX.int8* mem, AVX.int8 a)

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, k, iter
       int signs_int
       np.complex64_t c, z
       float[:, :] array_real, array_imag
       #float[:, :] out_counts_float = np.zeros_like(out_counts, dtype=np.float)
       float[:, :] out_counts_float
       float temp[4][8]
       int thread_id
       AVX.float8 c_real, c_imag, z_real, z_imag, temp_real, temp_imag, mag
       AVX.float8 b, b2, fours, twos, ones
       AVX.float8 iter_val, iter_now
       AVX.int8 iter_int8

       # To declare AVX.float8 variables, use:
       # cdef:
       #     AVX.float8 v1, v2, v3
       #
       # And then, for example, to multiply them
       #     v3 = AVX.mul(v1, v2)
       #
       # You may find the numpy.real() and numpy.imag() fuctions helpful.

    #out_counts_float = np.zeros_like(out_counts, dtype=np.float32)
    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    array_real = np.real(in_coords)
    array_imag = np.imag(in_coords)

    ones = AVX.float_to_float8(1)
    twos = AVX.float_to_float8(2)
    fours = AVX.float_to_float8(4)

    with nogil, parallel(num_threads = 4):
        for i in prange(in_coords.shape[0],schedule='static', chunksize=1):
            thread_id = cython.parallel.threadid()
            for j in range(0, in_coords.shape[1], 8):
                c_real = AVX.make_float8(array_real[i, j + 7],
                                    array_real[i, j + 6],
                                    array_real[i, j + 5],
                                    array_real[i, j + 4],
                                    array_real[i, j + 3],
                                    array_real[i, j + 2],
                                    array_real[i, j + 1],
                                    array_real[i, j + 0])

                c_imag = AVX.make_float8(array_imag[i, j + 7],
                                    array_imag[i, j + 6],
                                    array_imag[i, j + 5],
                                    array_imag[i, j + 4],
                                    array_imag[i, j + 3],
                                    array_imag[i, j + 2],
                                    array_imag[i, j + 1],
                                    array_imag[i, j + 0])
                iter_now = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
                z_real = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
                z_imag = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
                for iter in range(max_iterations):
                    # 1. calc mag
                    temp_real = AVX.mul(z_real, z_real)
                    temp_imag = AVX.mul(z_imag, z_imag)
                    mag = AVX.add(temp_real, temp_imag)
                    # 2. decide if continue
                    b = AVX.greater_than(fours, mag)
                    signs_int = AVX.signs(b)
                    if signs_int == 0:
                        break
                    # 3. calculate new z: z = z * z + c
                    temp_real = AVX.fmsub(z_imag, z_imag, c_real)
                    temp_real = AVX.fmsub(z_real, z_real, temp_real)
                    temp_imag = AVX.mul(z_real, z_imag)
                    z_imag = AVX.fmadd(twos, temp_imag, c_imag)
                    z_real = temp_real
                    # 4. update iter
                    iter_val = AVX.bitwise_and(b, ones)
                    iter_now = AVX.add(iter_now, iter_val)
                # 5. save to memory
                AVX.to_mem(iter_now, &(temp[thread_id][0]))
                # 6. type cast, cost a lot of time
                for k in range(8):
                    out_counts[i, j + k] = <int>(temp[thread_id][k])

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
