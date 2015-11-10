import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange



#Based on Thouis' recommendation on piazza:
cdef void counts_to_output(AVX.float8 counts, np.uint32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int idx
        
    AVX.to_mem(counts, &(tmp_counts[0]))
    for idx in range(8):
        out_counts[i, j*8 + idx] = <int> tmp_counts[idx]       

#Not used anymore:     
cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       float [:, :] ic_real, ic_imag
       AVX.float8 c_real, c_imag, z_real, z_imag, z_real2, counts, magn_z, mask, mask_not, z_imag_temp, z_real_temp, increm
       
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
    
    ic_real = np.real(in_coords)
    ic_imag = np.imag(in_coords)
    
    with nogil:
        #for i in prange(in_coords.shape[0], num_threads = 1, schedule = 'static', chunksize = 1):
        #for i in prange(in_coords.shape[0], num_threads = 2, schedule = 'static', chunksize = 1):
        for i in prange(in_coords.shape[0], num_threads = 4, schedule = 'static', chunksize = 1):

            #Selecting groups of 8:
            for j in range(in_coords.shape[1]/8):
            
                #Creating 2 AVX registars for the real and imaginary parts of the in_coords elements as suggested:
                
                c_real = AVX.make_float8(ic_real[i,j*8+7],
                                         ic_real[i,j*8+6],
                                         ic_real[i,j*8+5],
                                         ic_real[i,j*8+4],
                                         ic_real[i,j*8+3],
                                         ic_real[i,j*8+2],
                                         ic_real[i,j*8+1],
                                         ic_real[i,j*8])
                                     
                c_imag = AVX.make_float8(ic_imag[i,j*8+7],
                                         ic_imag[i,j*8+6],
                                         ic_imag[i,j*8+5],
                                         ic_imag[i,j*8+4],
                                         ic_imag[i,j*8+3],
                                         ic_imag[i,j*8+2],
                                         ic_imag[i,j*8+1],
                                         ic_imag[i,j*8])  
                
                #Initialising the z's:   
                                  
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)
                
                #Initialising the counts registrar tracking the iteration where each element of the in_coords registar satisfy magn(z)>4:
                
                counts = AVX.float_to_float8(0.0)
                
                for iter in range(max_iterations):
                    
                    #Evaluate the magnitude of z:
                    
                    magn_z = AVX.add( AVX.mul(z_real,z_real), AVX.mul(z_imag,z_imag))
                    
                    #Getting the mask summarising which z have magnitude greater or smaller than 4:
                
                    mask = AVX.less_than(AVX.float_to_float8(4.0), magn_z)
                    mask_not = AVX.greater_than(AVX.float_to_float8(4.0),magn_z)
                    
                    #If all z's in registar have magnitudes greater than 4 then break out of the loop
                    
                    if AVX.signs(mask_not) == 0:
                        break
                    
                    # Increment by 1 the counts that corresponds to z with magnitude less than 4:
                    increm = AVX.bitwise_andnot(mask,AVX.float_to_float8(1.0))
                    counts = AVX.add(counts,increm)
                    
                    #Updating the zs:
                    
                    z_real2 = z_real
                    z_real_temp = AVX.sub(AVX.mul(z_real,z_real), AVX.mul(z_imag,z_imag))
                    z_real_temp = AVX.add(z_real_temp,c_real)
                    z_imag_temp = AVX.mul(AVX.mul(z_real2,z_imag), AVX.float_to_float8(2.0))
                    z_imag_temp = AVX.add(z_imag_temp,c_imag)
                    
                    # Keeping z who already have magnitudes greater than 4 constant and transforming only those with magnitude less than 4:
                    
                    z_real = AVX.add(AVX.bitwise_and(mask, z_real),AVX.bitwise_and(mask_not,z_real_temp))
                    z_imag = AVX.add(AVX.bitwise_and(mask, z_imag),AVX.bitwise_and(mask_not,z_imag_temp))
                    
                #Adding the counts for this particular registar to the total counts:
                    
                counts_to_output(counts, out_counts, i, j)
                
                # It worked without having to return anything but I don't really understand why ! Is it some kind of cython magic
                # that returns the last object brought back from memory? Any comment appreciated !  

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
