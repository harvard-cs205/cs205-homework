import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange, parallel, threadid

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter, num_threads, k, indFlag, val0
       np.complex64_t c, z
       np.float32_t [:,:] re 
       np.float32_t [:,:] im
       AVX.float8 cRe, cIm, re2, im2, reTemp, imTemp, mTemp, mag, thresh, flag, iterC, one8, count8
       float countTemp[4][8]
       float [:,:] out_view = countTemp
       np.uint32_t [:, :] out_counts2

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
    
    out_counts2 = np.copy(out_counts)
    num_threads = 1
    thresh = AVX.float_to_float8(4.0)
    one8 = AVX.make_float8(1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0)
    re = np.real(in_coords)
    im = np.imag(in_coords)
    print "num threads: ",num_threads
    with nogil, parallel(num_threads=num_threads):
        val0 = threadid()
        for i in prange(in_coords.shape[0], schedule='static', chunksize = 1):
            for j in range(in_coords.shape[1]/8):
                #if (i == 0) & (j == 0):
                    #with gil:
                        #print re[i,0],im[i,0]
                cRe = AVX.make_float8(re[i, 8*j],re[i, 8*j+1],re[i, 8*j+2],re[i, 8*j+3],re[i, 8*j+4],re[i, 8*j+5],re[i, 8*j+6],re[i, 8*j+7])
                cIm = AVX.make_float8(im[i, 8*j],im[i, 8*j+1],im[i, 8*j+2],im[i, 8*j+3],im[i, 8*j+4],im[i, 8*j+5],im[i, 8*j+6],im[i, 8*j+7])
                re2 = AVX.make_float8(0,0,0,0,0,0,0,0)
                im2 = AVX.make_float8(0,0,0,0,0,0,0,0)
                count8 = AVX.make_float8(0,0,0,0,0,0,0,0)
                for iter in range(max_iterations):
                    reTemp = AVX.mul(re2,re2)
                    imTemp = AVX.mul(im2,im2)
                    mTemp = AVX.mul(re2,im2)
                    mag = AVX.add(reTemp,imTemp)

                    flag = AVX.less_than(mag,thresh)
                    iterC = AVX.bitwise_and(flag, one8)
                    if AVX.signs(flag) == 0:
                        break
                    count8 = AVX.add(count8,iterC) 
               
                    re2 = AVX.sub(reTemp,imTemp)
                    re2 = AVX.add(re2,cRe)
                    im2 = AVX.add(mTemp,mTemp)
                    im2 = AVX.add(im2,cIm)
                    #if (i == 0) & (j == 0) & (iter < 52):
                        #AVX.to_mem(mag, &(countTemp[val0][0]))
                        #with gil:
                            #print countTemp[val0][0]#countTemp[val0][1],countTemp[val0][2],countTemp[val0][3],countTemp[val0][4],countTemp[val0][5],countTemp[val0][6],countTemp[val0][7]
                        #AVX.to_mem(count8, &(countTemp[val0][0]))
                        #with gil:
                            #print countTemp[val0][0],countTemp[val0][1],countTemp[val0][2],countTemp[val0][3],countTemp[val0][4],countTemp[val0][5],countTemp[val0][6],countTemp[val0][7]
                AVX.to_mem(count8, &(countTemp[val0][0])) 
                for indFlag in range(0,8):
                    out_counts[i,8*j+indFlag] = <int> countTemp[val0][indFlag]
                        
    """
    print "num threads: ", num_threads
    for i in prange(in_coords.shape[0], nogil=True, schedule='static', chunksize = 1, num_threads=num_threads):
        for j in range(in_coords.shape[1]):
            #if (i == 0) & (j == 0):
                #with gil:
                    #print in_coords[i,j]
            c = in_coords[i, j]
            z = 0
            for iter in range(max_iterations):
                #if (i == 0) & (j == 0):
                    #with gil:
                        #print magnitude_squared(z)
                if magnitude_squared(z) > 4:
                    break
                z = z * z + c
            out_counts[i, j] = iter
    """  
                      

# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval
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

    mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)
    avxval = AVX.bitwise_andnot(mask, avxval)

    AVX.to_mem(avxval, &(out_vals[0]))

    return np.array(out_view)
