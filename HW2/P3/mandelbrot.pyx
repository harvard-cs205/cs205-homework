import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

from libc.stdio cimport printf


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, k, iter
       np.complex64_t c, z

       #used to store the real and imaginary parts of the image
       float[:, :] real_part, imaginary_part

       #to store the values in float8
       AVX.float8 real_c, imaginary_c, squared_real, squared_imaginary, temp_real, temp_imaginary, real_z, imaginary_z, magnitude, magnitude_four, new_iter_value, less_than_threshold

       #to convert the values back to numpy floats
       float float_array[8]

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    real_part = np.real(in_coords)
    imaginary_part = np.imag(in_coords)

    one_float =  AVX.float_to_float8(1)
    two_float =  AVX.float_to_float8(2)
    four_float = AVX.float_to_float8(4)

    with nogil:
        for i in prange(in_coords.shape[0], num_threads = 1, schedule="static", chunksize=1):
            #loop in steps of 8, since we are taking 8 instructions at a time
            for j in range(0, in_coords.shape[1], 8):
                
                #get the correct portions of the real and imaginary pixels
                real_c = AVX.make_float8(
                    real_part[i][j+7],
                    real_part[i][j+6],
                    real_part[i][j+5],
                    real_part[i][j+4],
                    real_part[i][j+3],
                    real_part[i][j+2],
                    real_part[i][j+1],
                    real_part[i][j],
                )

                imaginary_c = AVX.make_float8(
                    imaginary_part[i][j+7],
                    imaginary_part[i][j+6],
                    imaginary_part[i][j+5],
                    imaginary_part[i][j+4],
                    imaginary_part[i][j+3],
                    imaginary_part[i][j+2],
                    imaginary_part[i][j+1],
                    imaginary_part[i][j],
                )
                
                real_z = AVX.float_to_float8(0)
                imaginary_z = AVX.float_to_float8(0)
                new_iter_value = AVX.float_to_float8(0)
                
                for iter in range(max_iterations):                    
                    #calculate the new magnitude
                    #magnitude = real_z^2 + imaginary_z^2
                    squared_real = AVX.mul(real_z, real_z)
                    squared_imaginary = AVX.mul(imaginary_z, imaginary_z)
                    magnitude = AVX.add(squared_real, squared_imaginary)

                    #check if the magnitude is under the threshold
                    #we are really checking this for the 8 values, so we calculate the greater than
                    #less_than_threshold will be -1 where it is less than, and 0 where greater than
                    less_than_threshold = AVX.less_than(magnitude, four_float)

                    #if everything in less_than_threshold is 0 (all are greater than threshold)
                    #if none are under the threshold, break (signs converts all 8 floats to a single int)
                    if AVX.signs(less_than_threshold) == 0:
                        break

                    #update the real and imaginary portions
                    #z = z * z + c
                    #z = (real_z + imaginary_z * i) * (real_z + imaginary_z * i) + (real_c + imaginary_c * i)
                    #z = real_z^2 + 2 * real_z * imaginary_z * i - imaginary_z^2 + (real_c + imaginary_c * i)
                    #z = [real_z^2 - imaginary_z^2 + real_c] + [(2 * real_z * imaginary_z + imaginary_c) * i]

                    #real component of z = real_z^2 - imaginary_z^2 + real_c
                    temp_real = AVX.add(AVX.sub(squared_real, squared_imaginary), real_c)
                    
                    #imaginary component of z = 2 * real_z * imaginary_z + imaginary_c
                    temp_imaginary = AVX.add(AVX.mul(two_float, AVX.mul(real_z, imaginary_z)), imaginary_c)

                    #update the real and imaginary portions of z
                    real_z = temp_real
                    imaginary_z = temp_imaginary                                    

                    #convert every -1 in less_than_threshold to 1, and keep 0 as 0
                    less_than_threshold = AVX.bitwise_and(less_than_threshold, one_float)

                    #we will increase the out_count by 1 for each value that was under the threshold
                    new_iter_value = AVX.add(new_iter_value, less_than_threshold)

                #assigns the value of new_iter_value (a float8) at the memory address of float_array
                #we start by unpacking the float8 value into an array of 8 floats
                AVX.to_mem(new_iter_value, &(float_array[0]))
                
                #next we read each float from float_array and store it in out_counts
                for k in range(8):                                    
                    out_counts[i][j + k] = <int> float_array[k]                