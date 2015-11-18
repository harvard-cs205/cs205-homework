import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


#cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
#    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, k,iter
       #np.complex64_t c, z
       AVX.float8 c_real, c_imag, z_real,z_imag,cp2,
       AVX.float8 magnitude,temp1,temp2,mask,mask2, z_real_temp, iter_count, final_count, iter_count_filtered
       float out_vals[8] 

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
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=4):
        #for i in range(in_coords.shape[0]):
            for j in range(0,in_coords.shape[1],8):
                #c = in_coords[i, j]
                #z = 0
                c_real = AVX.make_float8( in_coords[i,j+7].real, in_coords[i,j+6].real,
                                               in_coords[i,j+5].real, in_coords[i,j+4].real,
                                               in_coords[i,j+3].real, in_coords[i,j+2].real,
                                               in_coords[i,j+1].real, in_coords[i,j+0].real  )
                c_imag = AVX.make_float8( in_coords[i,j+7].imag, in_coords[i,j+6].imag,
                                               in_coords[i,j+5].imag, in_coords[i,j+4].imag,
                                               in_coords[i,j+3].imag, in_coords[i,j+2].imag,
                                               in_coords[i,j+1].imag, in_coords[i,j+0].imag  )
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)
                cp2    = AVX.float_to_float8(2.0)
                final_count = AVX.float_to_float8(0.0)
                
                for iter in range(max_iterations):
                #    if magnitude_squared(z) > 4:
                #        break
                #    z = z * z + c
                #out_counts[i, j] = iter
                    temp1 = AVX.mul(z_real,z_real)           
                    temp2 = AVX.mul(z_imag,z_imag)           
                    magnitude  = AVX.add( temp1,temp2 )      
                    magnitude  = AVX.sqrt( magnitude )


                    mask = AVX.less_than(magnitude,cp2)
                    if AVX.signs(mask)==0.0:
                        break
                    iter_count = AVX.float_to_float8(1.0) 
                    iter_count_filtered = AVX.bitwise_and(mask,iter_count )
                    final_count = AVX.add(final_count,iter_count_filtered) 

        
                    temp1 = AVX.mul(z_real,z_real)
                    temp2 = AVX.mul(z_imag,z_imag)
                    z_real_temp = AVX.sub( temp1,temp2) 
                    z_real_temp = AVX.add( z_real_temp, c_real )

                    temp1 = AVX.mul(z_real,z_imag)
                    temp2 = AVX.mul(z_imag,z_real)                    
                    z_imag = AVX.add( temp1,temp2)
                    z_imag = AVX.add( z_imag, c_imag ) 
                    z_real = z_real_temp
                
                #AVX.to_mem(final_count,&(out_vals[0]))
                #for k in range(8):
                #    out_counts[i,j+k] = <int>out_vals[k]
                counts_to_output(final_count, out_counts, i, j)

cdef void counts_to_output(AVX.float8 counts, np.uint32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int k
    AVX.to_mem(counts, &(tmp_counts[0]))
    for k in range(8):
        out_counts[i,j+k]=<int>tmp_counts[k]



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
