import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

#Adapted from Piazza
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

cdef void print_AVX(AVX.float8 real8) nogil:
    cdef:
        float real[8]
        int i

    AVX.to_mem(real8, &(real[0]))
    with gil:
        for i in range(8):
            print("     {}: {}".format(i, real[i]))

#Adapted from Piazza
cdef void write_out_tmp(AVX.float8 tmp_out8, np.uint32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float tmp_out[8]
        int k

    AVX.to_mem(tmp_out8, &(tmp_out[0]))
    for k in range(8):
        out_counts[i, j+k] = <int>tmp_out[k]


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef AVX.float8 magnitude_squared8(AVX.float8 z_reals8, AVX.float8 z_imags8) nogil:
    return AVX.add(AVX.mul(z_reals8,z_reals8), AVX.mul(z_imags8,z_imags8))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter, num_threads, j_idx
       np.complex64_t c, z
       np.float32_t [:,:] in_reals, in_imags
       AVX.float8 reals8, imags8, z_reals8, z_imags8, fours8, iter8, write_mask8, not_written_mask8, tmp_out8, new_counts8, zeros8, nans8, new_z_reals8
       
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

    # convert to complex part and real part
    in_reals = np.real(in_coords)
    in_imags = np.imag(in_coords)
    
    num_threads = 4
     
    # for reseting purposes
    zeros8 = AVX.float_to_float8(0.0)

    # used in comparison later
    fours8 = AVX.float_to_float8(4.0)
    nans8 = AVX.greater_than(fours8,zeros8)
   
    
    with nogil:
        for i in prange(in_coords.shape[0],num_threads=num_threads,schedule='static',chunksize=1):
            for j_idx in range(in_coords.shape[1]/8): # grab sets of 8 columns
                j = j_idx * 8 # the real offset
                
                # get list of the complex values in the 8 columns
                # and convert to float8
                # there must be a better way to do this
                reals8 = AVX.make_float8(in_reals[i,j+7], in_reals[i,j+6], in_reals[i,j+5], in_reals[i,j+4], in_reals[i,j+3], in_reals[i,j+2], in_reals[i,j+1], in_reals[i,j])
                imags8 = AVX.make_float8(in_imags[i,j+7], in_imags[i,j+6], in_imags[i,j+5], in_imags[i,j+4], in_imags[i,j+3], in_imags[i,j+2], in_imags[i,j+1], in_imags[i,j])
                
                #reset float8 vectors for z's
                z_reals8 = AVX.bitwise_and(zeros8, zeros8)
                z_imags8 = AVX.bitwise_and(zeros8, zeros8)
                #reset tmp buffer
                #this will store the vector of counts before we actually write it to out_counts
                tmp_out8 = AVX.bitwise_and(zeros8, zeros8)
                
                for iter in range(max_iterations):
                    #write_mask8 has -NaN's where magnitude_squared < 4 and 0's otherwise 
                    # thus write mask has 0's where the mag_squared > 4 --> write to tmp buffer
                    write_mask8 = AVX.less_than(magnitude_squared8(z_reals8, z_imags8),fours8)
                    # invert the write_mask
                    # now we have -NaNs where we should write to tmp buffer
                    write_mask8 = AVX.bitwise_andnot(write_mask8, nans8)
                    
                    #AVX.greater_than(tmp_out8, zeros8) has NaN's where tmp_out has already been written to
                    # so invert that  (--> 0's where we have already written) & (-NaN's where we want to write)
                    # --> result is -NaN's where mag_squared > 4 and we have not written yet
                    write_mask8 = AVX.bitwise_andnot(AVX.greater_than(tmp_out8, zeros8), write_mask8)
                    
                    # current iter float8 vector
                    iter8 = AVX.float_to_float8(float(iter))
                    # the result of the AND is then iter where we want to write and 0's everywhere else.
                    new_counts8 = AVX.bitwise_and(write_mask8, iter8)

                    # add the new counts to the tmp buffer
                    tmp_out8 = AVX.add(new_counts8, tmp_out8)
                    
                    # check signs of values in tmp_out and see if we can quit
                    # Note: we can quit if every value in tmp_out is > 0
                    #AVX.greater_than(tmp_out8, zeros8) has NaN's where tmp_out has already been written to
                    if AVX.signs(AVX.greater_than(tmp_out8, zeros8)) == 255: #2^8-1 = (binary) 11111111
                        #write it out to out_counts
                        break
                        
                    # now do z = z * z + c
                    new_z_reals8 = AVX.add(AVX.sub(AVX.mul(z_reals8, z_reals8), AVX.mul(z_imags8, z_imags8)), reals8)
                    z_imags8 = AVX.add(AVX.add(AVX.mul(z_reals8, z_imags8), AVX.mul(z_reals8, z_imags8)), imags8)
                    z_reals8 = new_z_reals8
                #fill empty spots in tmp_out8 with iter (== max iterations - 1)
                tmp_out8 = AVX.add(AVX.bitwise_andnot(AVX.greater_than(tmp_out8, zeros8), iter8), tmp_out8)
                #write out complete tmp buffer
                write_out_tmp(tmp_out8, out_counts, i, j)    



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