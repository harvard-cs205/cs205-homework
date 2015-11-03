import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
#from cython.parallel import prange
from cython.parallel import parallel, prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef void counts_to_output(AVX.float8 counts, np.uint32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int jj

    AVX.to_mem(counts, &(tmp_counts[0]))
    
    #with nogil:
    for jj in range(8):
        out_counts[i, j+jj]=<int> tmp_counts[jj]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       #np.complex64_t c, z
       AVX.float8 cRe, cIm, mask, zRe, zIm, zRe2, counts, counts_iter# stores the real part, img part, and the mask, iter_count change name
       float tmp_counts[8]
       int jj


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
        #for i in range(in_coords.shape[0]):
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=1):# num_threads=1, 2, 4
            #for j in range(in_coords.shape[1]):
            for j in range(0,in_coords.shape[1],8):# j increment 8 at each loop
                cRe = AVX.make_float8(in_coords[i, j+7].real, in_coords[i, j+6].real, in_coords[i, j+5].real, in_coords[i, j+4].real, in_coords[i, j+3].real, in_coords[i, j+2].real, in_coords[i, j+1].real, in_coords[i, j].real) # real part
                cIm = AVX.make_float8(in_coords[i, j+7].imag, in_coords[i, j+6].imag, in_coords[i, j+5].imag, in_coords[i, j+4].imag, in_coords[i, j+3].imag, in_coords[i, j+2].imag, in_coords[i, j+1].imag, in_coords[i, j].imag) # img part
                zRe = AVX.float_to_float8(0.0) # 8 zeros
                zIm = AVX.float_to_float8(0.0) # 8 zeros
                counts = AVX.float_to_float8(0.0)
                counts_iter=AVX.float_to_float8(1.0)
                
                for iter in range(max_iterations):
                    #mask will be true(all 1s) where magnitude_squared(z) < 4.0
                    mask = AVX.less_than(AVX.add(AVX.mul(zRe,zRe), AVX.mul(zIm,zIm)), AVX.float_to_float8(4.0))

                    # possible optimisation: if all true, break earlier
                    if AVX.signs(mask)== 0.0:
                        break

                    counts=AVX.add(counts, AVX.bitwise_and(mask, counts_iter))

                    # update rule for z: z = z * z + c
                    zRe2 = AVX.sub( AVX.mul(zRe,zRe), AVX.mul(zIm,zIm) )# complex mult: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                    zRe2 = AVX.add( zRe2, cRe )
                    zIm = AVX.add( AVX.mul(zRe,zIm), AVX.mul(zIm,zRe) )
                    zIm = AVX.add( zIm, cIm )
                    zRe = zRe2

                    # update out_counts
                counts_to_output(counts, out_counts, i, j) 
                     

# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        int i, jj
        AVX.float8 avxval, mask
        float tmp[8]
        float out_vals[16]
        float [:] out_view = out_vals

    assert values.shape[0]%8==0

    # Note that the order of the arguments here is opposite the direction when
    # we retrieve them into memory.
    #with nogil:
    for i in range(values.shape[0]/8): 
        avxval = AVX.make_float8(values[i+7],
                                 values[i+6],
                                 values[i+5],
                                 values[i+4],
                                 values[i+3],
                                 values[i+2],
                                 values[i+1],
                                 values[i+0])

        avxval = AVX.sqrt(avxval)
   
    # mask will be true where 2.0 < avxval
        mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)

    # invert mask and select off values, so should be 2.0 >= avxval
    #avxval = AVX.add(AVX.bitwise_andnot(mask, avxval), AVX.bitwise_and(mask, AVX.float_to_float8(1.0)))
        avxval = AVX.bitwise_andnot(mask, avxval)
        AVX.to_mem(avxval, tmp)
        for jj in range(8):
            out_vals[i+jj]=tmp[jj]

    return np.array(out_view)
