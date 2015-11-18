import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


#cdef np.float64_t mag_squared(np.complex64_t z) nogil:
#    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations = 511):
    cdef:
       # To declare AVX.float8 variables, use:
       # cdef:
       #     AVX.float8 v1, v2, v3
       #
       # And then, for example, to multiply them
       #     v3 = AVX.mul(v1, v2)
       #
       # You may find the numpy.real() and numpy.imag() fuctions helpful.
       int i, j, k, iter
       #np.complex64_t c, z
       AVX.float8 real_pic_c, imag_pic_c, real_pic_z, imag_pic_z, cz_pic,
       AVX.float8 mag, tmp_num_a, tmp_num_b, mask, mask2, real_pic_z_tmp, cnt_iteration, cnt_finally, cnt_iteration_filt 
       float out_vals[8] 

       

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    with nogil:
        for i in prange(in_coords.shape[0], schedule = 'static', chunksize = 1, num_threads = 4):
        #for i in range(in_coords.shape[0]):
            for j in range(0,in_coords.shape[1],8):
                #c = in_coords[i, j]
                #z = 0
                #from back to start 
                real_pic_c = AVX.make_float8( in_coords[i, j + 7].real, in_coords[i, j + 6].real,
                                              in_coords[i, j + 5].real, in_coords[i, j + 4].real,
                                              in_coords[i, j + 3].real, in_coords[i, j + 2].real,
                                              in_coords[i, j + 1].real, in_coords[i, j + 0].real )

                imag_pic_c = AVX.make_float8( in_coords[i, j + 7].imag, in_coords[i, j + 6].imag,
                                              in_coords[i, j + 5].imag, in_coords[i, j + 4].imag,
                                              in_coords[i, j + 3].imag, in_coords[i, j + 2].imag,
                                              in_coords[i, j + 1].imag, in_coords[i, j + 0].imag )

                real_pic_z = AVX.float_to_float8(0.0)
                imag_pic_z = AVX.float_to_float8(0.0)
                cz_pic = AVX.float_to_float8(2.0)
                cnt_finally = AVX.float_to_float8(0.0)
                
                for iter in range(max_iterations):
                #    if mag_squared(z) > 4:
                #        break
                #    z = z * z + c
                #out_counts[i, j] = iter
                    # set tmp_num_a
                    tmp_num_a = AVX.mul(real_pic_z, real_pic_z) 
                    # set tmp_num_b          
                    tmp_num_b = AVX.mul(imag_pic_z, imag_pic_z) 
                    # set magnitude          
                    mag  = AVX.sqrt( AVX.add( tmp_num_a, tmp_num_b ) ) 
                    # set mask   
                    mask = AVX.less_than(mag, cz_pic)
                    cnt_iteration = AVX.float_to_float8(1.0) 
                    cnt_iteration_filt = AVX.bitwise_and( mask, cnt_iteration )
                    cnt_finally = AVX.add(cnt_finally, cnt_iteration_filt) 
                    # set new temp_num_a and temp_num_b
                    tmp_num_a = AVX.mul(real_pic_z, real_pic_z)
                    tmp_num_b = AVX.mul(imag_pic_z, imag_pic_z)
                    # set real_pic_z_temp
                    real_pic_z_tmp = AVX.sub( tmp_num_a, tmp_num_b ) 
                    real_pic_z_tmp = AVX.add( real_pic_z_tmp, real_pic_c )
                    # set new temp_num_a and temp_num_b
                    tmp_num_a = AVX.mul(real_pic_z, imag_pic_z)
                    tmp_num_b = AVX.mul(imag_pic_z, real_pic_z)
                    # set imag_pic_z and real_pic_z                    
                    imag_pic_z = AVX.add( tmp_num_a, tmp_num_b )
                    imag_pic_z = AVX.add( imag_pic_z, imag_pic_c ) 
                    real_pic_z = real_pic_z_tmp
          
                counts_to_output(cnt_finally, out_counts, i, j)

cdef void counts_to_output(AVX.float8 counts, np.uint32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int k
    AVX.to_mem(counts, &(tmp_counts[0]))
    for k in range(8):
        out_counts[i,j + k] = <int>tmp_counts[k]



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