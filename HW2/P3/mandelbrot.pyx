import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

cdef void count_out(AVX.float8 results, np.uint32_t[:,:] out_counts, int i, int j) nogil:
    cdef:
        float output_vals[8]
        int idx
    #Storing the value in an size 8 array of floats
    AVX.to_mem(results,&(output_vals[0]))
    for idx in range(8):
        #Converting to unsigned int before assigning it to outcounts[i,j]
        out_counts[i,j*8+idx] = <unsigned int>output_vals[idx]



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

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter, n_threads, signs
       float c_real,c_imag,z_real,z_imag
       float[:,:] coords_real,coords_imag
       np.complex64_t c, z
       AVX.float8 c_real8, c_imag8, z_real8, z_imag8, mag, z_real8_int, z_imag8_int, iter8,\
               iter_mask, res, avx4, axv0
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
    
    n_threads = 2
    coords_real = np.real(in_coords)
    coords_imag = np.imag(in_coords)  
    with nogil:
        for i in prange(in_coords.shape[0],schedule='static',chunksize=1,num_threads=n_threads):
            for j in range(in_coords.shape[1]/8):
                c_real8 = AVX.make_float8(coords_real[i,j*8+7],\
                                          coords_real[i,j*8+6],\
                                          coords_real[i,j*8+5],\
                                          coords_real[i,j*8+4],\
                                          coords_real[i,j*8+3],\
                                          coords_real[i,j*8+2],\
                                          coords_real[i,j*8+1],\
                                          coords_real[i,j*8])
                c_imag8 = AVX.make_float8(coords_imag[i,j*8+7],\
                                        coords_imag[i,j*8+6],\
                                        coords_imag[i,j*8+5],\
                                        coords_imag[i,j*8+4],\
                                        coords_imag[i,j*8+3],\
                                        coords_imag[i,j*8+2],\
                                        coords_imag[i,j*8+1],\
                                        coords_imag[i,j*8])
                z_real8 = AVX.make_float8(0,0,0,0,0,0,0,0)
                z_imag8 = AVX.make_float8(0,0,0,0,0,0,0,0)
                #intializing results initially to zero
                res = AVX.make_float8(0,0,0,0,0,0,0,0)
                avx4 = AVX.float_to_float8(4.0) 
                for iter in range(max_iterations):
                    #using temporary variables for later swaps
                    z_real8_int = z_real8
                    z_imag8_int = z_imag8
                    #finding the magnitude
                    mag = AVX.add(AVX.mul(z_real8,z_real8),AVX.mul(z_imag8,z_imag8))
                    #creating an array of 1s
                    iter8 = AVX.float_to_float8(1.0)
                    #using a mask to find which values to update
                    iter_mask =AVX.less_than(mag, avx4)
                    #using bitwise_and to keeps 1 in those places where we
                    #need to update the result vector
                    iter8 = AVX.bitwise_and(iter8,iter_mask)
                    #incrementing iteration counts
                    res = AVX.add(res,iter8)
                    signs = AVX.signs(AVX.sub(avx4,mag))
                    #Breaking out of loop if all the values are negative
                    if signs == 255:
                        break 
                    #Updating Z 
                    z_real8 = AVX.add(AVX.sub(AVX.mul(z_real8_int,z_real8_int),\
                            AVX.mul(z_imag8_int,z_imag8_int)),c_real8)
                    z_imag8 = AVX.add(AVX.mul(AVX.mul(AVX.float_to_float8(2.0),\
                            z_imag8_int),z_real8_int),c_imag8)

                count_out(res, out_counts, i, j)
    '''
    with nogil:
        for i in prange(in_coords.shape[0],\
                schedule='static',chunksize=1,num_threads=n_threads):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter
    '''


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
