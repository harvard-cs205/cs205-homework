import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange
from cython.parallel import parallel
from libc.stdlib cimport abort, malloc, free

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
        int i, j, iter, k
        np.complex64_t c, z
        float* tmp_counts
        AVX.float8 coor_real, coor_imag, z_real, z_imag, new_z_imag, tmp, new_z_real
        AVX.float8 iterat, mask, add_it, test_z, stop, checker
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

    with nogil, parallel(num_threads = 1):
        tmp_counts =  <float *> malloc(sizeof(float) * 8)
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1):
            for j in range(in_coords.shape[1]/8):

                # Note that the order of the arguments here is opposite the direction when
                # we retrieve them into memory.
                coor_real = AVX.make_float8(in_coords[i, j*8+7].real,
                                         in_coords[i, j*8+6].real,
                                         in_coords[i, j*8+5].real,
                                         in_coords[i, j*8+4].real,
                                         in_coords[i, j*8+3].real,
                                         in_coords[i, j*8+2].real,
                                         in_coords[i, j*8+1].real,
                                         in_coords[i, j*8].real)



                coor_imag = AVX.make_float8(in_coords[i, j*8+7].imag,
                                         in_coords[i, j*8+6].imag,
                                         in_coords[i, j*8+5].imag,
                                         in_coords[i, j*8+4].imag,
                                         in_coords[i, j*8+3].imag,
                                         in_coords[i, j*8+2].imag,
                                         in_coords[i, j*8+1].imag,
                                         in_coords[i, j*8].imag)



                z_real = AVX.float_to_float8(0.0)

                z_imag = AVX.float_to_float8(0.0)

                iterat = AVX.float_to_float8(0.0)

                #Stop controls whether the iteration has been broken
                stop = AVX.float_to_float8(1.0)


                #tmp = AVX.float_to_float8(0.0)
                #new_z_imag = AVX.float_to_float8(0.0)
                #new_z_real = AVX.float_to_float8(0.0)
                #test_z = AVX.float_to_float8(0.0)
                #add_it = AVX.float_to_float8(0.0)
                #mask = AVX.float_to_float8(0.0)

                for iter in range(max_iterations):

                    #we have to do z = z*z +c
                    # let z = a+bi , c = x+yi
                    # doing complex expansion: we get 
                    # (a^2-b^2) + (2ab)i + (x + yi)
                    # = (a^2-b^2 + x) + (2ab + y)i   
                    # = (a^2-b^2 + x) + (a+a)b + y)i  
    
                    #this is doing imaginary part
                    tmp = AVX.add(z_real, z_real)
                    new_z_imag = AVX.fmadd(tmp, z_imag, coor_imag) 


                    #this is doing real part
                    tmp = AVX.mul(z_real, z_real)
                    new_z_real = AVX.mul(z_imag, z_imag)

                    #this is magnitude squared
                    test_z = AVX.add(tmp, new_z_real)

                    # mask will be true where 4.0 >= new_z_real
                    mask = AVX.greater_than(AVX.float_to_float8(4.0), test_z)

                    # invert mask and select off values, so should be 4.0 > test_z
                    add_it = AVX.bitwise_and(mask, AVX.float_to_float8(1.0))


                    #AND the mask with stop. Stop controls whether the iteration has been broken
                    stop = AVX.bitwise_and(add_it, stop)

                    #subtract one from stop to make it -1 for stop, 0 for add. This allow us to check the signs
                    checker = AVX.sub(stop, AVX.float_to_float8(1.0))
                    if AVX.signs(checker) == 255:
                        break


                    new_z_real = AVX.sub(tmp, new_z_real)

                    new_z_real = AVX.add(new_z_real, coor_real)

                    #update z
                    z_real = new_z_real
                    z_imag = new_z_imag


                    iterat = AVX.add(iterat, stop)

                AVX.to_mem(iterat, &(tmp_counts[0]))   
                
                for k in range(8):
                    out_counts[i, j*8+k] = int(tmp_counts[k])
        free(tmp_counts)


# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval, tmp, mask
        AVX.float8 zeros
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


    zeros = AVX.make_float8(1, 1, 1, 1, 1, 1, 1, 1)

    print AVX.signs(zeros)

    # mask will be true where 2.0 < avxval
    mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)

    # invert mask and select off values, so should be 2.0 >= avxval
    avxval = AVX.bitwise_andnot(mask, AVX.float_to_float8(1.0))

    AVX.to_mem(avxval, &(out_vals[0]))

    return np.array(out_view)
