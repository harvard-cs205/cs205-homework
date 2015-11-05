import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange
from libc.stdlib cimport malloc, free


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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot_mul(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511,
                 int num_t=1):
    cdef:
        int i, j, iter
        np.complex64_t c, z
        int num_threads = num_t

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    for i in prange(in_coords.shape[0], nogil=True, num_threads=num_t, schedule='static', chunksize=1):
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
cpdef mandelbrot_opt(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511,
                 int num_t=1):
    cdef:
        int i, j, k, iter
        AVX.float8 c_real, c_imag, mask, iters
        AVX.float8 z_real, z_real_tmp
        AVX.float8 z_imag 

        np.float32_t [:, :] r_incoords = np.real(in_coords)
        np.float32_t [:, :] i_incoords = np.imag(in_coords)
        float *out_val = <float *> malloc(8 * sizeof(float))

        int num_threads = num_t

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"
    
    for i in prange(in_coords.shape[0], nogil=True, num_threads=num_threads, schedule='static', chunksize=1):
        for j in range(0, in_coords.shape[1], 8):
            z_real = AVX.float_to_float8(0.0)
            z_imag = AVX.float_to_float8(0.0)
            iters = AVX.float_to_float8(0.0)
            c_real = AVX.make_float8(r_incoords[i, j+7],
                                   r_incoords[i, j+6],
                                   r_incoords[i, j+5],
                                   r_incoords[i, j+4],
                                   r_incoords[i, j+3],
                                   r_incoords[i, j+2],
                                   r_incoords[i, j+1],
                                   r_incoords[i, j])
            c_imag = AVX.make_float8(i_incoords[i, j+7],
                                   i_incoords[i, j+6],
                                   i_incoords[i, j+5],
                                   i_incoords[i, j+4],
                                   i_incoords[i, j+3],
                                   i_incoords[i, j+2],
                                   i_incoords[i, j+1],
                                   i_incoords[i, j])
        
            for iter in range(max_iterations):
                mask = AVX.less_than(AVX.fmadd(z_real, z_real, AVX.mul(z_imag, z_imag)), AVX.float_to_float8(4.0) )
                if not AVX.signs(mask): break
                z_real_tmp = AVX.add(AVX.sub(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag)), c_real)
                z_imag = AVX.add(AVX.add(AVX.mul(z_real, z_imag), AVX.mul(z_real, z_imag)), c_imag)
                z_real = z_real_tmp

                iters = AVX.add(iters, AVX.bitwise_and(mask, AVX.float_to_float8(1.0)))

            AVX.to_mem(iters, &out_val[0])
            for k in range(8):
                out_counts[i, j+k] = <int>out_val[k]

    free(<void *> out_val)


cdef void print_complex_AVX(AVX.float8 real,
                             AVX.float8 imag) nogil:
    cdef:
        float real_parts[8]
        float imag_parts[8]
        int i

    AVX.to_mem(real, &(real_parts[0]))
    AVX.to_mem(imag, &(imag_parts[0]))
    with gil:
        for i in range(8):
            print("    {}: {}, {}".format(i, real_parts[i], imag_parts[i]))

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
