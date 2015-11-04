import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange
        
@cython.boundscheck(False)
@cython.wraparound(False)
        
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
        int i, j, iter
        
        #wrote each variable individually to make sure none left out      
        AVX.float8 z_mag
        AVX.float8 z_real
        AVX.float8 z_imag 
        AVX.float8 z_real_new 
        AVX.float8 z_imag_new 
        AVX.float8 cr
        AVX.float8 ci 
        AVX.float8 mask 
        AVX.float8 const_one 
        AVX.float8 const_four
        AVX.float8 counts 
        AVX.float8 const_zero 
        
        float [:,:] real, imag
    
    #grab the real and im. parts of in_coords before we give up gil
    real = np.real(in_coords)
    imag = np.imag(in_coords)
    
    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"  

    with nogil:
        
        #Parallelize over 1,2, and 4 threads. 4 threads shown here
        for i in prange(in_coords.shape[0],num_threads=4,schedule='static',chunksize=1):
            for j in range(in_coords.shape[1]/8): #note: divide by 8 since we will grab 8 elements at a time 
                
                #create float8 constants to be used in operations below
                const_zero = AVX.float_to_float8(0.0)
                const_one = AVX.float_to_float8(1.0) 
                const_four = AVX.float_to_float8(4.0) 
                
                #Initialize counts which will be incremented each iteration below
                counts = AVX.float_to_float8(0.0)
                
                #initialize the real and imag. parts of z initially set to 0
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)
                
                #pull in 8 real parts of in_coords at a time in AVX
                cr = AVX.make_float8(real[i,7+8*j],
                                    real[i,6+8*j],
                                    real[i,5+8*j],
                                    real[i,4+8*j],
                                    real[i,3+8*j],
                                    real[i,2+8*j],
                                    real[i,1+8*j],
                                    real[i,0+8*j])                
                
                #pull in 8 imag parts of in_coords at a time in AVX
                ci = AVX.make_float8(imag[i,7+8*j],
                                    imag[i,6+8*j],
                                    imag[i,5+8*j],
                                    imag[i,4+8*j],
                                    imag[i,3+8*j],
                                    imag[i,2+8*j],
                                    imag[i,1+8*j],
                                    imag[i,0+8*j]) 
                                    
                for iter in range(max_iterations):
                    
                    #compute magnitude of z which is: (real^2 + imag^2)
                    z_mag = AVX.add(AVX.mul(z_real,z_real),AVX.mul(z_imag,z_imag))                                        
                    
                    #Check to see where in our 8 parts of z is the value < 4                    
                    mask = AVX.less_than(z_mag, const_four) 

                    #increase counts by 1 at indices where mag. of z < 4
                    counts = AVX.add(AVX.bitwise_and(mask,const_one),counts)                        
                     
                    #stop if all 8 magnitudes of z are less than 4 (mask = 0's) 
                    if AVX.signs(mask) == 0:
                        break
                        
                    #z*z+c rewritten as [(real^2 - imag^2) + cr] + [(2*real*imag) + ci]i
                    z_real_new = AVX.add(AVX.sub(AVX.mul(z_real,z_real),AVX.mul(z_imag,z_imag)),cr)
                    z_imag_new = AVX.add(AVX.mul(z_real,z_imag),AVX.add(AVX.mul(z_real,z_imag),ci))  
                
                    z_real = z_real_new
                    z_imag = z_imag_new
                
                #when done with iterations, call counts_to_output
                counts_to_output(counts,out_counts,i,j)
                
cdef void counts_to_output(AVX.float8 counts,
                    np.uint32_t [:, :] out_counts,
                    int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int x
        
    #call to _mem as instructed    
    AVX.to_mem(counts, &(tmp_counts[0]))
    
    #insert the set of 8 count values computed into our final out_counts array
    for x in range(8):
        out_counts[i,x + 8*j] = <int>tmp_counts[x]   
    