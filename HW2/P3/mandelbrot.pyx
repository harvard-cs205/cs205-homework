###########################################################################
#Jaemin Cheun
#CS205, Fall 2015 Computing Foundations for Computer Science
#Nov 4, 2015
#mandelbrot.pyx
###########################################################################

import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


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
       np.float32_t [:,:] in_coords_r, in_coords_i
       AVX.float8 c_r, c_i, z_r, z_i, counts, mask, z_temp_r, z_temp_i

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

    # We first seperate the real and imaginary parts of the in_coords
    in_coords_r = np.real(in_coords)
    in_coords_i = np.imag(in_coords) 

    with nogil:
        for i in prange(in_coords.shape[0], schedule = 'static', chunksize = 1, num_threads = 4):
          for j in range(in_coords.shape[1] / 8):
            c_r= AVX.make_float8(in_coords_r[i,j*8+7],
              in_coords_r[i,j*8+6],
              in_coords_r[i,j*8+5],
              in_coords_r[i,j*8+4],
              in_coords_r[i,j*8+3],
              in_coords_r[i,j*8+2],
              in_coords_r[i,j*8+1],
              in_coords_r[i,j*8+0]) 
            c_i= AVX.make_float8(in_coords_i[i,j*8+7],
              in_coords_i[i,j*8+6],
              in_coords_i[i,j*8+5],
              in_coords_i[i,j*8+4],
              in_coords_i[i,j*8+3],
              in_coords_i[i,j*8+2],
              in_coords_i[i,j*8+1],
              in_coords_i[i,j*8+7])   

            #initialize z and counter
            z_r = AVX.float_to_float8(0.0)
            z_i = AVX.float_to_float8(0.0)
            counts = AVX.float_to_float8(0.0)


            for iter in range(max_iterations):
              # No need to stop computations for values > 4.0
              mask = AVX.less_than(AVX.add(AVX.mul(z_r,z_r),AVX.mul(z_i,z_i)),AVX.float_to_float8(4.0))

              # We stop when the values are all false (255), got help from the piazza post
              if AVX.signs(mask) == 0:
                break
              # We update the real and imaginary values of z
              z_temp_r = z_r
              z_temp_i = z_i
              z_r = AVX.add(AVX.sub(AVX.mul(z_temp_r,z_temp_r),AVX.mul(z_temp_i,z_temp_i)),c_r)
              z_i = AVX.add(AVX.mul(AVX.mul(z_temp_r,z_temp_i),AVX.float_to_float8(2.0)),c_i)

              # We only increase the counter when it passes the break
              counts = AVX.add(counts,AVX.bitwise_and(mask,AVX.float_to_float8(1.0)))


            counts_to_output(counts, out_counts, i, j)

# Followed the piazza post for this
cdef void counts_to_output(AVX.float8 counts, np.uint32_t[:,:] out_counts, int i, int j) nogil:
  cdef:
    float temp[8]
    int idx
  AVX.to_mem(counts, &(temp[0]))
  for idx in range(8):
    out_counts[i, j*8 + idx] = <np.uint32_t> temp[idx]

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
