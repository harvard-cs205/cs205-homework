import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf, stdout, fprintf

cimport openmp

cdef float RIDICULOUS = -1000.


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef AVX.float8 magnitude_squared_float8(AVX.float8 z_real_float8, AVX.float8 z_imag_float8) nogil:
    cdef AVX.float8 real_mag = AVX.mul(z_real_float8, z_real_float8)
    cdef AVX.float8 imag_mag = AVX.mul(z_imag_float8, z_imag_float8)

    return AVX.add(real_mag, imag_mag)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    cdef int num_cols_to_iterate = in_coords.shape[1]/8
    cdef int j_start, j_end

    cdef float[:, :] real_in_coords = np.real(in_coords)
    cdef float[:, :] imag_in_coords = np.imag(in_coords)

    cdef float *real_c
    cdef float *imag_c

    cdef AVX.float8 real_c_float8, imag_c_float8

    cdef AVX.float8 real_z_float8, imag_z_float8
    cdef AVX.float8 mag_squared
    cdef AVX.float8 not_go_mask

    cdef int go
    cdef AVX.float8 iter
    cdef AVX.float8 temp_z_real, temp_z_imag
    cdef AVX.float8 to_subtract
    cdef AVX.float8 decrementer = AVX.float_to_float8(-1)

    cdef AVX.float8 ridiculous_value = AVX.float_to_float8(RIDICULOUS)
    cdef AVX.float8 max_iterations_f8 = AVX.float_to_float8(max_iterations)
    cdef AVX.float8 over_max_iterations

    print 'before', np.asarray(out_counts).sum()
    with nogil:
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=1):
            for j in range(num_cols_to_iterate): # Parallelize via AVX here...do 8 at a time
                j_start = 8*j
                j_end = 8*(j+1)
                if j_end >= in_coords.shape[1]:
                    j_end = in_coords.shape[1] - 1

                real_c = &real_in_coords[i][0] # get pointers to the arrays
                imag_c = &imag_in_coords[i][0]

                real_c_float8 = array_to_float8(real_c, j_start, j_end)
                imag_c_float8 = array_to_float8(imag_c, j_start, j_end)

                real_z_float8 = AVX.float_to_float8(0)
                imag_z_float8 = AVX.float_to_float8(0)

                # Need to iterate over 8 values at once...blahhhhh
                iter = AVX.float_to_float8(0)

                while True:

                    mag_squared = magnitude_squared_float8(real_z_float8, imag_z_float8)
                    not_go_mask = AVX.greater_than(mag_squared, AVX.float_to_float8(4))

                    if AVX.signs(not_go_mask) == 255:
                        break

                    # Increment iter...we will adjust for improper goers in a second
                    iter = AVX.add(iter, AVX.float_to_float8(1)) #Increment at beginning, like for loop
                    # Subtract 1 from the ones that weren't supposed to go
                    to_subtract = AVX.bitwise_and(not_go_mask, decrementer)
                    iter = AVX.add(iter, to_subtract)

                    temp_z_real = do_mandelbrot_update_real(real_z_float8, imag_z_float8, real_c_float8)
                    temp_z_imag = do_mandelbrot_update_imag(real_z_float8, imag_z_float8, imag_c_float8)

                    # printf('\n')
                    # printf('Original:\n')
                    # print_float8(real_z_float8)
                    # printf('Updated:\n')
                    # print_float8(temp_z_real)
                    # printf('\n')

                    real_z_float8 = temp_z_real
                    imag_z_float8 = temp_z_imag

                    # We need to make sure those that shouldn't go don't update anymore...we set their value to
                    # something ridiculous such that the magnitude will always be greater than 4 next iteration

                    real_z_float8 = AVX.add(AVX.bitwise_and(not_go_mask, ridiculous_value),
                                            AVX.bitwise_andnot(not_go_mask, real_z_float8))
                    imag_z_float8 = AVX.add(AVX.bitwise_and(not_go_mask, ridiculous_value),
                                            AVX.bitwise_andnot(not_go_mask, imag_z_float8))

                    # Check that you are not equal to the maximum number of iterations
                    over_max_iterations = AVX.greater_than(iter, max_iterations_f8)
                    # One is over the max iteration...and we increment each time, so if one is over we are done!
                    if AVX.signs(over_max_iterations) > 0:
                        break

                # Assign the iterations
                assign_values_to_matrix(iter, &out_counts[i][0], j_start, j_end)
    print 'after', np.asarray(out_counts).sum()

cdef void print_float8(AVX.float8 f8) nogil:
    cdef np.float32_t* iter_view = <float *> malloc(8*sizeof(np.float32_t))

    AVX.to_mem(f8, iter_view)
    cdef int i
    for i in range(8):
        printf('%f \n', iter_view[i])
    printf('Done with float8')
    printf('\n')
    free(iter_view)

cdef void assign_values_to_matrix(AVX.float8 iter, np.uint32_t *to_big_matrix, int j_start, int j_end) nogil:
    cdef float* iter_view = <float *> malloc(8*sizeof(np.float32_t))

    AVX.to_mem(iter, iter_view)

    # Now assign appropriately
    cdef int j
    cdef int count = 0
    for j in range(j_start, j_end):
        to_big_matrix[j] = <np.uint32_t> iter_view[count]
        count += 1

    free(iter_view)

cdef AVX.float8 do_mandelbrot_update_real(AVX.float8 z_real, AVX.float8 z_imag, AVX.float8 c_real) nogil:
    '''Real part is a^2 - b^2 + c_real'''

    cdef AVX.float8 a_squared = AVX.mul(z_real, z_real)
    cdef AVX.float8 b_squared = AVX.mul(z_imag, z_imag)
    cdef AVX.float8 a2_minus_b2 = AVX.sub(a_squared, b_squared)
    return AVX.add(a2_minus_b2, c_real)

cdef AVX.float8 do_mandelbrot_update_imag(AVX.float8 z_real, AVX.float8 z_imag, AVX.float8 c_imag) nogil:
    '''Imaginary part is 2ab + c_imag'''

    cdef AVX.float8 a_b = AVX.mul(z_real, z_imag)
    cdef AVX.float8 two_a_b = AVX.mul(AVX.float_to_float8(2), a_b)
    return AVX.add(two_a_b, c_imag)

cdef AVX.float8 array_to_float8(float *c, int j_start, int j_end) nogil:
    cdef float *filled_array = <float *> malloc(8*sizeof(c))

    cdef int count = 0
    cdef int j
    for j in range(j_start, j_end):
        filled_array[count] = c[j]
        count += 1
    # Fill in the rest with ridiculous values so you know they are garbage
    while count < 8:
        filled_array[count] = RIDICULOUS
        count += 1

    cdef AVX.float8 f8 = AVX.make_float8(filled_array[7],
                           filled_array[6],
                           filled_array[5],
                           filled_array[4],
                           filled_array[3],
                           filled_array[2],
                           filled_array[1],
                           filled_array[0])
    free(filled_array)
    return f8

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
    avxval = AVX.bitwise_and(mask, avxval)
    #avxval = AVX.signs(mask)

    AVX.to_mem(avxval, &(out_vals[0]))

    return np.array(out_view)
