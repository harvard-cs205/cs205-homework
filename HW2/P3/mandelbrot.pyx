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
       int i, j, iter
       np.complex64_t c, z
       AVX.float8 real_avx, img_avx, real_part, img_part, counter_avx, mask_avx, criteria_avx, increment_avx, c1, c2

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

    temp_out_counts = np.zeros_like(out_counts).astype(float)


    real = numpy.real(in_coords)
    img = numpy.imag(in_coords)

    print real.shape
    with nogil:
        for i in prange(4000, schedule='static', chunksize=1, num_threads=2):
            for j in prange(4000/8, schedule='static', chunksize=1, num_threads=2):
                with gil:
                    print(i)
                    print(j)
                c1 = AVX.float_to_float8(numpy.real(in_coords[i, j]))
                c2 = AVX.float_to_float8(numpy.imag(in_coords[i, j]))
                real_avx = AVX.make_float8(real[i, j*8],
                                           real[i, j*8+1],
                                           real[i, j*8+2],
                                           real[i, j*8+3],
                                           real[i, j*8+4],
                                           real[i, j*8+5],
                                           real[i, j*8+6],
                                           real[i, j*8+7])
                img_avx = AVX.make_float8(img[i, j*8],
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
                    real_part = AVX.add(AVX.sub(AVX.mul(real_avx, real_avx), AVX.mul(real_avx, real_avx)), c1)
                    img_part = AVX.add(AVX.mul(AVX.mul(real_avx, img_avx), AVX.float_to_float8(2)), c2)
                    mask_avx = AVX.less_tan(AVX.add(AVX.mul(real_part, real_part), AVX.mul(img_part, img_part)), 
                                            AVX.float_to_float8(4))
                    if AVX.sign(mask_avx)==0.0:
                        break
                    increment_avx = AVX.bitwise_andnot(AVX.float_to_float8(1), mask_avx)
                    counter_avx = AVX.add(counter_avx, increment_avx)
                    real_avx = real_part
                    img_avx = img_part

                AVX.to_mem(counter_avx, &(temp_out_counts[i, j*8]))
    out_counts = temp_out_counts.astype(int)






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
