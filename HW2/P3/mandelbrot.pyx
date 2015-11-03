cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, k, h #iter
       np.complex64_t c, z
       AVX.float8 vcreal,vcimag,vzreal,vzimag,tmpreal
       AVX.float8 threshold,iter_flag,iter,cmp,max_iter,mag_sq
       float out[8]
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
    
    with nogil, parallel(num_threads=2):
      # malloc space for buffer
      local_buf=<float *> malloc(sizeof(float)*8)
      if local_buf==NULL:
          abort
      for i in prange(in_coords.shape[0],schedule='static',chunksize=1):
          # try 8-way parallel 
           for j in range(0,in_coords.shape[1],8):
                # init AVX float8
                vcreal=AVX.make_float8(in_coords[i,j].real,in_coords[i,j+1].real,in_coords[i,j+2].real,in_coords[i,j+3].real,in_coords[i,j+4].real,in_coords[i,j+5].real,in_coords[i,j+6].real,in_coords[i,j+7].real)
                vcimag=AVX.make_float8(in_coords[i,j].imag,in_coords[i,j+1].imag,in_coords[i,j+2].imag,in_coords[i,j+3].imag,in_coords[i,j+4].imag,in_coords[i,j+5].imag,in_coords[i,j+6].imag,in_coords[i,j+7].imag)
                vzreal=AVX.float_to_float8(0)
                vzimag=AVX.float_to_float8(0)
                threshold=AVX.float_to_float8(4)
                iter_flag=AVX.float_to_float8(1)
                iter=AVX.float_to_float8(0)
                # start loop
                for h in range(max_iterations):
                    mag_sq=AVX.add(AVX.mul(vzreal,vzreal),AVX.mul(vzimag,vzimag))
                    cmp=AVX.less_eq_than(mag_sq,threshold)
                    iter_flag=AVX.bitwise_and(iter_flag,cmp)
                    if AVX.signs(cmp)==0:  #when all 8 numbers are greater than 4 
                        break;        
                    tmpreal=AVX.add(AVX.sub(AVX.mul(vzreal,vzreal),AVX.mul(vzimag,vzimag)),vcreal)
                    vzimag=AVX.add(AVX.add(AVX.mul(vzreal,vzimag),AVX.mul(vzreal,vzimag)),vcimag)
                    vzreal=tmpreal
                    iter=AVX.add(iter,iter_flag)
                
                AVX.to_mem(iter, &(local_buf[0]))
                for k in range(8):
                    out_counts[i, j+k]=int(local_buf[k])
      free(local_buf)
