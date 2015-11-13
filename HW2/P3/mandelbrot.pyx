import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free


# cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
#     return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)

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

cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
        int i, j, jj, iter
        np.complex64_t c, z
        AVX.float8 c_real, c_imag, z_real, z_imag,z_real_new, magnitude_squared, cont_case,avx_out_counts
        #thread local buffer for storing out counts in memory before populating 
        #out_counts array
        float *out_counts_temp
        int arrsize0=in_coords.shape[0]
        int arrsize1=in_coords.shape[1]
        np.float32_t [:,:] in_coords_real=np.zeros((arrsize0,arrsize1),dtype=np.float32)
        np.float32_t [:,:] in_coords_imag=np.zeros((arrsize0,arrsize1),dtype=np.float32)


    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"
    for i in range(in_coords.shape[0]):
        for j in range(in_coords.shape[1]):
            in_coords_real[i,j]=in_coords[i,j].real
            in_coords_imag[i,j]=in_coords[i,j].imag
    with nogil, parallel(num_threads=4):
        #setup thread local buffer
        out_counts_temp = <float *> malloc(sizeof(float) * 8)
        for i in prange(in_coords.shape[0],schedule='static',chunksize=1):           
            for j in range(in_coords.shape[1]/8):
                c_real = AVX.make_float8(in_coords_real[i,j*8+7],
                                    in_coords_real[i,j*8+6],
                                    in_coords_real[i,j*8+5],
                                    in_coords_real[i,j*8+4],
                                    in_coords_real[i,j*8+3],
                                    in_coords_real[i,j*8+2],
                                    in_coords_real[i,j*8+1],
                                    in_coords_real[i,j*8+0])
                c_imag = AVX.make_float8(in_coords_imag[i,j*8+7],
                                    in_coords_imag[i,j*8+6],
                                    in_coords_imag[i,j*8+5],
                                    in_coords_imag[i,j*8+4],
                                    in_coords_imag[i,j*8+3],
                                    in_coords_imag[i,j*8+2],
                                    in_coords_imag[i,j*8+1],
                                    in_coords_imag[i,j*8+0])
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)
                avx_out_counts=AVX.float_to_float8(0.0)
                for iter in range(max_iterations):
                    # #do the magnitude squared computation in two steps...
                    magnitude_squared=AVX.mul(z_real,z_real)
                    magnitude_squared=AVX.fmadd(z_imag,z_imag,magnitude_squared)

                    #mag^2<4, return 1. 
                    cont_case=AVX.less_than_equal(magnitude_squared,AVX.float_to_float8(4))

                    #if mag^2<4, cont_case will be -NaN, otherwise 0
                    #When all iterations for all 8 elements are done, 
                    #all bits of cont_case will be 0.
                    if AVX.signs(cont_case) == 0:
                        break

                    z_real_new = AVX.add(AVX.sub(AVX.mul(z_real,z_real),AVX.mul(z_imag,z_imag)),c_real)
                    z_imag = AVX.add(AVX.mul(AVX.mul(z_real,z_imag),AVX.float_to_float8(2)),c_imag)
                    z_real=z_real_new
                #add 1 to counts from previous iteration for elements that aren't finished. Do this by masking 1 with cont_case, 
                    avx_out_counts = AVX.add(avx_out_counts,AVX.bitwise_and(AVX.float_to_float8(1.0),cont_case))
                AVX.to_mem(avx_out_counts,out_counts_temp)
                for jj in range(8):
                    out_counts[i, 8*j+jj] = <int> out_counts_temp[jj]
        free(out_counts_temp)


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
