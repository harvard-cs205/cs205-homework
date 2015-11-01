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
                 np.float32_t [:, :] out_counts,
                 int max_iterations=511):
    #np.uint32_t [:, :] out_counts,
    cdef:
       int i, j, iter
       int width = in_coords.shape[1]
       int height = in_coords.shape[0]
       np.complex64_t c, z
       AVX.float8 c_r, c_i, real_part, img_part, counter_avx, mask_avx, criteria_avx, increment_avx, z_i, z_r, mask_avx2
       float [:, :] real, img

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    real = numpy.real(in_coords)
    img = numpy.imag(in_coords)

    with nogil:
        for i in prange(height, schedule='static', chunksize=1, num_threads=8):
            for j in range(width/8):

                z_r = AVX.float_to_float8(0.0)
                z_i = AVX.float_to_float8(0.0)
                c_r = AVX.make_float8(real[i, j*8],
                                           real[i, j*8+1],
                                           real[i, j*8+2],
                                           real[i, j*8+3],
                                           real[i, j*8+4],
                                           real[i, j*8+5],
                                           real[i, j*8+6],
                                           real[i, j*8+7])
                c_i = AVX.make_float8(img[i, j*8],
                                          img[i, j*8+1],
                                          img[i, j*8+2],
                                          img[i, j*8+3],
                                          img[i, j*8+4],
                                          img[i, j*8+5],
                                          img[i, j*8+6],
                                          img[i, j*8+7])
                counter_avx = AVX.float_to_float8(0)
                criteria_avx = AVX.float_to_float8(4)

                for iter in range(max_iterations):

                    real_part = AVX.add(AVX.sub(AVX.mul(z_r, z_r), AVX.mul(z_i, z_i)), c_r)
                    img_part = AVX.add(AVX.mul(AVX.mul(z_r, z_i), AVX.float_to_float8(2)), c_i)
                    mask_avx = AVX.less_than(AVX.add(AVX.mul(real_part, real_part), AVX.mul(img_part, img_part)), 
                                            AVX.float_to_float8(4))
                    if AVX.signs(mask_avx)==0:
                        break
                    increment_avx = AVX.bitwise_and(mask_avx, AVX.float_to_float8(1))
                    counter_avx = AVX.add(counter_avx, increment_avx)
                    z_r = real_part
                    z_i = img_part
                AVX.to_mem(counter_avx, &(out_counts[i, j*8]))






cpdef mandelbrot_origin(np.complex64_t [:, :] in_coords,
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
