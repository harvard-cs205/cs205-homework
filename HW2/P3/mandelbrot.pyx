import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef void counts_to_output(AVX.float8 counts,
                      np.uint32_t [:, :] out_counts,
                      int i, int block) nogil:
    cdef:
        float tmp_counts[8]
        int idx
    AVX.to_mem(counts, &(tmp_counts[0]))
    for idx in range(8):
        out_counts[i, 8*block + idx] = <unsigned int> tmp_counts[0]

cdef void print_AVX(AVX.float8 reg) nogil:
    cdef:
        float regi[8]
        int i

    AVX.to_mem(reg, &(regi[0]))
    with gil:
        for i in range(8):
            print("    {}: {}".format(i, regi[i]))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
        int i, iter, block, blox, j
        float output[8]
        float [:, :] inco_real, inco_imag
        AVX.float8 avx0, avx1, avx2, avx4, magsquared, mask, zimag, zreal, zreal_inter, cimag, creal, itercounts 

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

# Without Instruction-level parallelism
#    with nogil:
 #       for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=4):
  #          for j in range(in_coords.shape[1]):
   #             c = in_coords[i, j]
    #            z = 0
     #           for iter in range(max_iterations):
      #              if magnitude_squared(z) > 4:
       #                 break
        #            z = z * z + c
         #       out_counts[i, j] = iter

    # Separate out the real and imaginary components of in_coords:
    inco_real = np.real(in_coords)
    inco_imag = np.imag(in_coords)

    # 8-bit registers of 0, 2, and 4 that I use below
    avx0 = AVX.float_to_float8(0)
    avx1 = AVX.float_to_float8(1)
    avx2 = AVX.float_to_float8(2)
    avx4 = AVX.float_to_float8(4)
    
    # With Instruction-level parallelism
    with nogil:
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=4): 
            # Break each row into blocks of 8

            # Need number of blocks on each row:
            blox = in_coords.shape[1]/8

            # Now I parallelize within each of those blocks.  
            for block in range(blox):              
                
                # Take the initial real components of c.
                creal = AVX.make_float8(inco_real[i, 8*block + 7],
                                        inco_real[i, 8*block + 6],
                                        inco_real[i, 8*block + 5],
                                        inco_real[i, 8*block + 4],
                                        inco_real[i, 8*block + 3],
                                        inco_real[i, 8*block + 2],
                                        inco_real[i, 8*block + 1],
                                        inco_real[i, 8*block + 0])
                
                # Take the initial imaginary components of c.  
                cimag = AVX.make_float8(inco_imag[i, 8*block + 7],
                                        inco_imag[i, 8*block + 6],
                                        inco_imag[i, 8*block + 5],
                                        inco_imag[i, 8*block + 4],
                                        inco_imag[i, 8*block + 3],
                                        inco_imag[i, 8*block + 2],
                                        inco_imag[i, 8*block + 1],
                                        inco_imag[i, 8*block + 0])
                
                # Starting values of zero for z and the iteration counts.
                zreal = avx0
                zimag = avx0              
                itercounts = avx0

                for iter in range(max_iterations):
                
                    # Magnitude squared (sum of squares of real and imaginary components).
                    magsquared = AVX.mul(AVX.mul(zreal,zreal),AVX.mul(zimag,zimag))
                     
                    # Determine whether the magnitude squared is less than 4
                    mask = AVX.less_than(magsquared, avx4)
                    
                    # If all elements are less than zero, write the iteration counts to the array.
                    # Also, if still iterating at the last iteration, write the maximum iteration count
                    #    to the array.  And break out of the loop.  
                    if AVX.signs(mask)==0 or iter==(max_iterations - 1):
                        counts_to_output(itercounts, out_counts, i, block)
                        break

                    # If the loop hasn't hit break, convert the mask to 1's (for those still iterating)
                    #     and 0's (for those that have completed),  
                    mask = AVX.bitwise_and(mask, avx1)

                    # Add mask to the iteration counts to keep a count of iterations for each element.
                    itercounts = AVX.add(itercounts, mask)
                    
                    # Update zreal and zimag, based on:
                    #    c = a + bj, z = e + fj
                    #    z*z = (e*e) + (2*e*f)*j + (f*f)*(j*j)
                    #        = (e*e) + (2*e*f)*j + (f*f)*(-1)
                    #        = (e*e - f*f) + (2*e*f)*j
                    # z*z + c= (e*e - f*f + a) + (2*e*f + b)*j

                    # Make an intermediate copy to update with later so that this iterations values    
                    #    for zreal can still be used to update zimag, and update both components.  
                    zreal_inter = AVX.add(creal, AVX.sub(AVX.mul(zreal,zreal),AVX.mul(zimag,zimag)))
                    zimag = AVX.add(cimag, AVX.mul(avx2,AVX.mul(zreal,zimag)))
                    zreal = zreal_inter
                       
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
