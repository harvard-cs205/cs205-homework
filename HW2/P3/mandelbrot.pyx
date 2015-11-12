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

cdef AVX.float8 testfunction(AVX.float8 input) nogil: 
  cdef: 
    AVX.float8 output 
    AVX.float8 ones 
  ones = AVX.float_to_float8(1.0)
  output = AVX.add(ones, input)
  return output


cdef AVX.float8 square_number_real(AVX.float8 real, AVX.float8 imag) nogil:
    ## (x + yj)^2 has real part equal to x^2 - y^2 
    return AVX.sub(AVX.mul(real, real), AVX.mul(imag, imag))

cdef AVX.float8 square_number_imag(AVX.float8 real, AVX.float8 imag) nogil:
    ## (x + yj)^2 has real part equal to x^2 - y^2 
    return AVX.add(AVX.mul(real, imag), AVX.mul(real, imag))

cdef void write_output(np.uint32_t [:, :] out_counts, AVX.float8 answer, int i, int j) nogil:
  cdef: 
    float temp_array[8] 
    int offset   
  AVX.to_mem(answer, &(temp_array[0]))
  for offset in range(8): 
    out_counts[i, j + offset] = <np.uint32_t> temp_array[offset]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, dummy
       np.complex64_t c,
       AVX.float8 temp, magnitude, C_imag, C_real, iteration, ones, flter, z_real, z_imag, mask,

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
        ones = AVX.float_to_float8(1.0)
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=4):
            for j in xrange(in_coords.shape[1] / 8):

              flter = AVX.float_to_float8(1.0)
              j = 8 * j 
              C_imag = AVX.make_float8(
                in_coords[i, j+7].imag, in_coords[i, j+6].imag,
                in_coords[i, j+5].imag, in_coords[i, j+4].imag,
                in_coords[i, j+3].imag, in_coords[i, j+2].imag,
                in_coords[i, j+1].imag, in_coords[i, j+0].imag,
                )

              C_real = AVX.make_float8(
                in_coords[i, j+7].real, in_coords[i, j+6].real,
                in_coords[i, j+5].real, in_coords[i, j+4].real,
                in_coords[i, j+3].real, in_coords[i, j+2].real,
                in_coords[i, j+1].real, in_coords[i, j+0].real,
                )

              z_real = AVX.float_to_float8(0.0)
              z_imag = AVX.float_to_float8(0.0)
              iteration = AVX.float_to_float8(1.0)

              for dummy in range(max_iterations): 
                # square and add c  

                temp = AVX.add(square_number_real(z_real, z_imag), C_real)
                z_imag = AVX.add(square_number_imag(z_real, z_imag), C_imag)
                z_real = temp 
                # check magnitude
                magnitude = AVX.fmadd(z_real, z_real, AVX.mul(z_imag, z_imag))
                mask = AVX.less_than(magnitude, AVX.float_to_float8(4.0))

                if AVX.signs(mask) == 0: 
                  break 

                flter = AVX.bitwise_and(flter, mask)
                iteration = AVX.add(iteration,AVX.bitwise_and(ones, flter))
                
              write_output(out_counts, iteration, i, j)
              
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
