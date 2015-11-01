import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

#######
#Don't forget to delete the function after testing
#this function was used for print AVX for debugging
#######

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
       np.complex64_t c, z
       #declare float8 variables
       AVX.float8 c_i, c_r, z_i, z_r, counters,magz,cond,z_r_temp

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

    #without AVX
    #for i in prange(in_coords.shape[0], nogil=True, schedule='static', chunksize = 1, num_threads = 4):
    #    for j in range(in_coords.shape[1]):
    #        c = in_coords[i, j]
    #        z = 0
    #        for iter in range(max_iterations):
    #            if magnitude_squared(z) > 4:
    #                break
    #            z = z * z + c
    #        out_counts[i, j] = iter



    #with AVX
    for i in prange(in_coords.shape[0], nogil=True, schedule='static', chunksize = 1, num_threads = 4):
        #divide y into 8 (start, end, step)
        for j in range(0,in_coords.shape[1],8):

            #get the real parts of in_coords and store them into cr
            c_r = AVX.make_float8((in_coords[i,j+7].real),
                                 (in_coords[i,j+6].real),
                                 (in_coords[i,j+5].real),
                                 (in_coords[i,j+4].real),
                                 (in_coords[i,j+3].real),
                                 (in_coords[i,j+2].real),
                                 (in_coords[i,j+1].real),
                                 (in_coords[i,j].real))

            #get the imaginary part if in_coords and store them into ci
            c_i = AVX.make_float8((in_coords[i,j+7].imag),
                                 (in_coords[i,j+6].imag),
                                 (in_coords[i,j+5].imag),
                                 (in_coords[i,j+4].imag),
                                 (in_coords[i,j+3].imag),
                                 (in_coords[i,j+2].imag),
                                 (in_coords[i,j+1].imag),
                                 (in_coords[i,j].imag))

            #create z for 8 parts, but all 0
            z_i =  AVX.float_to_float8(0.0)
            z_r = AVX.float_to_float8(0.0)

            #initialzie the counter for 8 registers
            counters = AVX.float_to_float8(0.0)



            #start the iterations
            for iter in range(max_iterations):

                #check if magnitude of z is less than 4, if not, break the program
                #comparison of AVX and assign values to z
                #using mask as comparison
                #calculate the magnitude of z
                magz = AVX.add(AVX.mul(z_r,z_r), AVX.mul(z_i,z_i))

                #create a condition(if z is smaller than 4)
                cond = AVX.less_than(magz, AVX.float_to_float8(4.0))

                #check the condition
                #if the condition is true, so do the calculation
                #and the iteration plus one
                #update the z
                #if they are all false, then we can break it
                if(AVX.signs(cond) == 0):
                    break
                #else we could update the z value
                else:
                    z_r_temp = z_r
                    z_r = AVX.add(AVX.sub(AVX.mul(z_r,z_r),AVX.mul(z_i,z_i)),c_r)
                    z_i = AVX.add(AVX.mul(AVX.mul(AVX.float_to_float8(2.0),z_r_temp),z_i),c_i)

                    #also update counters which has 1(which still satisfy out condition)
                    #we only need to increment certain counters which satisfy our conditions
                    counters = AVX.add(AVX.bitwise_and(cond,AVX.float_to_float8(1.0)),counters)

        #print_AVX(counters)
                counts_to_output(counters, out_counts, i, j)



# a function which put output the result(safe)
# no need to return anything because we are dealing with memory
cdef void counts_to_output(AVX.float8 counters,
                      np.uint32_t[:, :] out_counts,
                      int i, int j) nogil:
    cdef:
        float tmp_counters[8]
        int index
    AVX.to_mem(counters, &(tmp_counters[0]))
    for index in range(8):
        out_counts[i,j+index] = <unsigned int> tmp_counters[index]



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
