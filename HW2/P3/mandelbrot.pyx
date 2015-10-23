import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

cdef void counts_to_output(AVX.float8 results, np.uint32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float out_vals[8]
        np.uint32_t [:] out_np
        int idx
    AVX.to_mem(results, &(out_vals[0]))
    with gil:
        out_np = np.array(out_vals, dtype='uint32')
        for idx in range(out_np.shape[0]):
            out_counts[i, j*8 + idx] = out_np[idx]
            
cdef void print_AVX(AVX.float8 x, int index=7, int include_below = 1) nogil:
    cdef:
        float x_temp[8]
        int i

    AVX.to_mem(x, &(x_temp[0]))
    with gil:
        if include_below == 1:
            for i in range(index+1):
                print("{}: {}".format(i, x_temp[i]))
        else:
            print("{}: {}".format(index, x_temp[index]))

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
        int i, j, iter
        float [:, :] in_coords_real, in_coords_imag
        AVX.float8 c_real, c_imag, z_real, z_imag, z_temp, magnitude, avx_4, avx_0, avx_2, iter_mask, iter_count, results

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
        avx_4 = AVX.float_to_float8(4.0)
        avx_0 = AVX.float_to_float8(0.0)
        avx_2 = AVX.float_to_float8(2.0)
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=2):
            for j in range(in_coords.shape[1]/8):
                c_real = AVX.make_float8(in_coords_real[i,j*8+7],
                                         in_coords_real[i,j*8+6],
                                         in_coords_real[i,j*8+5],
                                         in_coords_real[i,j*8+4],
                                         in_coords_real[i,j*8+3],
                                         in_coords_real[i,j*8+2],
                                         in_coords_real[i,j*8+1],
                                         in_coords_real[i,j*8])
                c_imag = AVX.make_float8(in_coords_imag[i,j*8+7],
                                         in_coords_imag[i,j*8+6],
                                         in_coords_imag[i,j*8+5],
                                         in_coords_imag[i,j*8+4],
                                         in_coords_imag[i,j*8+3],
                                         in_coords_imag[i,j*8+2],
                                         in_coords_imag[i,j*8+1],
                                         in_coords_imag[i,j*8])
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)
                results = AVX.float_to_float8(0.0)
                for iter in range(max_iterations):
                    iter_count = AVX.float_to_float8(iter)
                    magnitude = AVX.add(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))
                    #get intersection of magnitude > 4 and values that are 0 in results
                    iter_mask = AVX.bitwise_andnot(AVX.less_than(avx_0, results), AVX.greater_than(magnitude, avx_4))
                    #update iter_count to have values where iter_max is 1
                    iter_count = AVX.bitwise_and(iter_count, iter_mask)
                    #update results with iter_count
                    results = AVX.add(iter_count, results)
                    #stop loop if all results > 0
                    if AVX.signs(AVX.greater_than(results, avx_0))==255:
                        break
                    #update z
                    z_temp = z_real
                    z_real = AVX.add(AVX.sub(AVX.mul(z_real, z_real),AVX.mul(z_imag, z_imag)), c_real)
                    z_imag = AVX.add(AVX.mul(AVX.mul(z_temp, z_imag),avx_2), c_imag)
                #set results with 0 to max_iterations-1
                iter_count = AVX.float_to_float8(max_iterations-1)
                iter_mask = AVX.bitwise_andnot(AVX.less_than(avx_0, results), AVX.greater_than(magnitude, avx_0))
                iter_count = AVX.bitwise_and(iter_count, iter_mask)
                results = AVX.add(iter_count, results)
                #write to out_counts
                counts_to_output(results, out_counts, i, j)



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
