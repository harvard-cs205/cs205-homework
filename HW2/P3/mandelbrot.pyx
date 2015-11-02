

import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag


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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       np.complex64_t c, z

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

    for i in prange(in_coords.shape[0], nogil=True, num_threads=8, schedule=static):
        for j in range(in_coords.shape[1]):
            c = in_coords[i, j]
            z = 0
            for iter in range(max_iterations):
                if magnitude_squared(z) > 4: #then his in AVX
                    break
                z = z * z + c #start with coding this in AVX
            out_counts[i, j] = iter






# An example using AVX instructions
cpdef mandelbrot_AVX(np.complex64_t [:,:] in_coords, np.uint32_t [:,:] out_counts):

'''
This is the mandelbrot calcualtion using AVX
'''

    cdef:
        AVX.float8 tmp_real, tmp_img, c, z, mask, xx, yy, real, img1
        AVX.float8 img, mag, new_ones, xx_mul, yy_mul, xy_sub, z_mag_real
        AVX.float8 z_mag_img, z_mag, z_real, z_img, xy_2_mul, two_xy_mul, xy_mul 
        AVX.float8 counts
        int i, j, it, k
        int max_iterations=511
        np.float32_t [:,:] coords_1 
        np.float32_t [:,:] coords_2 
        AVX.float8 zeros = AVX.float_to_float8(0)
        AVX.float8 thresh = AVX.float_to_float8(4)
        AVX.float8 two = AVX.float_to_float8(2)
        AVX.float8 ones = AVX.float_to_float8(1)
        float out_vals[8]
        float [:] out_view = out_vals

    coords_1 = np.real(in_coords)
    coords_2 = np.imag(in_coords)



    for j in prange(in_coords.shape[0], nogil=True, num_threads=8, schedule=static):
      for i in range(0, in_coords.shape[1], 8):

        #pull out the real components
        tmp_real = AVX.make_float8(coords_1[j, i],coords_1[j, i+1],coords_1[j, i+2],coords_1[j, i+3],coords_1[j, i+4],coords_1[j, i+5],coords_1[j, i+6],coords_1[j, i+7])

        #pull out the imaginary components
        tmp_img = AVX.make_float8(coords_2[j, i],coords_2[j, i+1],coords_2[j, i+2],coords_2[j, i+3],coords_2[j, i+4],coords_2[j, i+5],coords_2[j, i+6],coords_2[j, i+7])


        #initialize all of the vectors
        z_real = zeros
        z_img = zeros
        z_mag_real = zeros
        z_mag_img = zeros
        z = zeros
        counts = zeros
        z_mag = AVX.add(z_mag_real, z_mag_img)


        for it in range(max_iterations):

          #create the mask which acts as the if statement
          mask = AVX.less_than(z_mag, thresh) 

          if not AVX.signs(mask): #this will ensure that the loop breaks if the conditions are no longer met
              break

          new_ones = AVX.bitwise_and(ones, mask)#added not
          counts = AVX.add(new_ones, counts)

          ################overall computatations################
          #    Must execute the computations separately, 
          #    real and imaginary
          #    z_real = x^2 - y^2 + x
          #    z_img = 2xyi + y
          ######################################################

          #calculate the real poriton using AVX
          xx_mul = AVX.mul(z_real, z_real) # x^2
          yy_mul = AVX.mul(z_img, z_img) # y^2
          xy_sub = AVX.sub(xx_mul, yy_mul) # x^2 - y^2 

          #calculate the imaginary poriton using AVX
          xy_mul = AVX.mul(z_real, z_img) # xy
          two_xy_mul = AVX.mul(two, xy_mul) # 2xy

          #consolidate the the real and imaginary portions
          z_real = AVX.add(xy_sub, tmp_real) # x^2 - y^2 + x
          z_img = AVX.add(two_xy_mul, tmp_img) # xyi + y

          #consolidate real and imaginary for the magnitude calculation
          z_mag_real = AVX.mul(z_real, z_real) 
          z_mag_img = AVX.mul(z_img, z_img)
          z_mag = AVX.add(z_mag_real, z_mag_img)



        for k in range(8): #write the information out to memory 
          out_counts[j,i+k] = <np.uint32_t>((<np.float32_t*>&counts)[k])


        






