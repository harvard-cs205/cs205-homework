import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef void print_complex_AVX(AVX.float8 real,
                             AVX.float8 imag) nogil:
    cdef:
        float real_parts[8]
        float imag_parts[8]
        int i65

    AVX.to_mem(real, &(real_parts[0]))
    AVX.to_mem(imag, &(imag_parts[0]))
    with gil:
        for i in range(8):
            print("    {}: {}, {}".format(i, real_parts[i], imag_parts[i]))

cdef void counts_to_output(AVX.float8 counts, np.uint32_t[:, :] out_counts, int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int idx
    AVX.to_mem(counts, &(tmp_counts[0]))
    for idx in range(8):
        out_counts[i,j+idx] = <np.uint32_t> tmp_counts[idx]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       np.complex64_t c, z
       AVX.float8 c_real, c_imag, z_real, z_imag, mask_to_update, counts, mask_current, mask, z_real_temp
       float in_coords_real[8], in_coords_imag[8]

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
        for i in prange(in_coords.shape[0], num_threads = 4, schedule = 'static', chunksize = 1):
            for j in range(0, in_coords.shape[1], 8):
                c_real = AVX.make_float8(in_coords[i, j+7].real,
                                         in_coords[i,j+6].real,
                                         in_coords[i,j+5].real,
                                         in_coords[i,j+4].real,
                                         in_coords[i,j+3].real,
                                         in_coords[i,j+2].real,
                                         in_coords[i,j+1].real,
                                         in_coords[i,j].real)

                c_imag = AVX.make_float8(in_coords[i][j+7].imag,
                                         in_coords[i,j+6].imag,
                                         in_coords[i,j+5].imag,
                                         in_coords[i,j+4].imag,
                                         in_coords[i,j+3].imag,
                                         in_coords[i,j+2].imag,
                                         in_coords[i,j+1].imag,
                                         in_coords[i,j].imag)

                z_real = AVX.float_to_float8(0)
                z_imag = AVX.float_to_float8(0)
                counts = AVX.float_to_float8(0)
                for iter in range(max_iterations):
                    # break condition
                    if AVX.signs(AVX.greater_than(counts, AVX.float_to_float8(0))) == 255:
                        break

                    # update counts
                    # pixels eligible to be updated in current round

                    mask_current = AVX.less_than(AVX.fmadd(z_real, z_real, AVX.mul(z_imag, z_imag)), AVX.float_to_float8(4.0))
                    # pixels which haven't been updated
                    mask_to_update = AVX.less_than(counts, AVX.float_to_float8(0.5))
                    mask = AVX.bitwise_andnot(mask_current, mask_to_update)

                    counts = AVX.add(counts, AVX.bitwise_and(mask, AVX.float_to_float8(iter)))

                    # update z
                    z_real_temp = z_real
                    z_real = AVX.add(AVX.sub(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag)), c_real)
                    z_imag = AVX.fmadd(AVX.float_to_float8(2), AVX.mul(z_real_temp, z_imag), c_imag)

                counts = AVX.add(counts, AVX.bitwise_and(AVX.float_to_float8(max_iterations-1),
                                    AVX.bitwise_andnot(AVX.less_than(AVX.float_to_float8(0), counts),
                                    AVX.greater_than(AVX.fmadd(z_real, z_real,
                                    AVX.mul(z_imag, z_imag)), AVX.float_to_float8(0)))))

                # print_complex_AVX(counts, counts)
                counts_to_output(counts, out_counts, i, j)



# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        int sign
        AVX.float8 avxval, tmp, mask
        float out_vals[8], mask_val[8]
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

    mask2 = AVX.less_than(AVX.float_to_float8(2.0), AVX.float_to_float8(1.0))
    mask3 = AVX.greater_than(AVX.float_to_float8(2.0), AVX.float_to_float8(1.0))

    print_complex_AVX(mask2, mask3)

    mask4 = AVX.bitwise_andnot(mask2, mask3)

    print_complex_AVX(mask4, AVX.bitwise_and(mask4, AVX.float_to_float8(5.0)))

    sign = AVX.signs(AVX.float_to_float8(-1))
    sign2 = AVX.signs(AVX.float_to_float8(0))
    sign3 = AVX.signs(AVX.float_to_float8(1))
    sign4 = AVX.signs(AVX.make_float8(1,2,3,4,-1,-2,-3,-4))

    # invert mask and select off values, so should be 2.0 >= avxval
    avxval = AVX.bitwise_andnot(mask, avxval)

    AVX.to_mem(avxval, &(out_vals[0]))
    AVX.to_mem(mask, &(mask_val[0]))

    return np.array(out_view), ('sign', sign),('sign2', sign2),('sign3', sign3),('sign4', sign4)