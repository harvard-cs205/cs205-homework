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
       int i, j, p,iter, all_true = 255
       np.complex64_t c, z
       AVX.float8 c_r, c_i, z_r, z_i, mag_sq, z_r_prev, z_i_prev
       AVX.float8 prev_sign, curr_sign, iter_array, t
       float [:, :] in_real, in_imag
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
    in_real = np.real(in_coords)
    in_imag = np.imag(in_coords)
    # with nogil:
    #     for i in prange(in_coords.shape[0],schedule='static',chunksize=1,num_threads=4):
    #         for j in range(in_coords.shape[1]):
    #             c_r = AVX.float_to_float8(in_real[i,j])
    #             c_i = AVX.float_to_float8(in_imag[i,j])
    #             z_r = AVX.float_to_float8(0.0)
    #             z_i = AVX.float_to_float8(0.0)
    #             for iter in range(max_iterations):
    #                 mag_sq = AVX.add(AVX.mul(z_i,z_i),AVX.mul(z_r,z_r))
    #                 if AVX.signs(AVX.greater_than(mag_sq,AVX.float_to_float8(4.0))) == all_true:
    #                     break
    #                 z_r_prev = z_r
    #                 z_r = AVX.add(AVX.sub(AVX.mul(z_r_prev,z_r_prev),AVX.mul(z_i_prev,z_i_prev)),c_r)
    #                 z_i = AVX.add(AVX.mul(AVX.float_to_float8(2.0),AVX.mul(z_r_prev,z_i_prev)),c_i)
    #             out_counts[i, j] = iter
    with nogil:
        for i in prange(in_coords.shape[0],schedule='static',chunksize=1,num_threads=4):
            for j in range(in_coords.shape[1]):
                if j % 8 == 0:
                    c_r = AVX.make_float8(in_real[i,j], \
                                            in_real[i,j+1],\
                                            in_real[i,j+2],\
                                            in_real[i,j+3],\
                                            in_real[i,j+4],\
                                            in_real[i,j+5],\
                                            in_real[i,j+6],\
                                            in_real[i,j+7])
                    c_i = AVX.make_float8(in_imag[i,j], \
                                            in_imag[i,j+1],\
                                            in_imag[i,j+2],\
                                            in_imag[i,j+3],\
                                            in_imag[i,j+4],\
                                            in_imag[i,j+5],\
                                            in_imag[i,j+6],\
                                            in_imag[i,j+7])
                    z_r = AVX.float_to_float8(0.0)
                    z_i = AVX.float_to_float8(0.0)
                    prev_sign = AVX.float_to_float8(0.0)
                    iter_array = AVX.float_to_float8(0.0)
                    for iter in range(max_iterations):
                        mag_sq = AVX.add(AVX.mul(z_i,z_i),AVX.mul(z_r,z_r))
                        # curr_sign is the float8 array of >4's
                        curr_sign = AVX.greater_than(mag_sq,AVX.float_to_float8(4.0))
                        # if you've got at least one that's not 0
                        if AVX.signs(curr_sign) != 0:
                            t = AVX.bitwise_and(curr_sign,AVX.float_to_float8(1.0))
                            # take elements that have just changed to 1 and multiply then by iter and add them to iter
                            iter_array = AVX.add(AVX.mul(AVX.sub(t,prev_sign),AVX.float_to_float8(iter)),iter_array)
                            prev_sign = t
                            if AVX.signs(curr_sign) == all_true:
                                break
                        z_r_prev = z_r
                        z_r = AVX.add(AVX.sub(AVX.mul(z_r_prev,z_r_prev),AVX.mul(z_i,z_i)),c_r)
                        z_i = AVX.add(AVX.mul(AVX.float_to_float8(2.0),AVX.mul(z_r_prev,z_i)),c_i)
                    counts_to_output(iter_array, out_counts, i, j)
cdef void counts_to_output(AVX.float8 counts, np.uint32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int k

    AVX.to_mem(counts, &(tmp_counts[0]))
    for k in range(8):
        out_counts[i, j+k]=<int>tmp_counts[k]


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

cdef void print_AVX(AVX.float8 real, int iter) nogil:
    cdef:
        float real_parts[8]
        int i

    AVX.to_mem(real, &(real_parts[0]))
    with gil:
        print("ITER NUMBER {}".format(iter))
        for i in range(8):
            print("    {}: {}".format(i, real_parts[i]))

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
