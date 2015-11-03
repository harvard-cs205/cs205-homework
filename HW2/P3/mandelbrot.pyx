import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

# A helper function I used to debug the code by printing AVX
cdef void print_AVX(AVX.float8 input_AVX):
  cdef:
    float copy_values[8]
  AVX.to_mem(input_AVX, &copy_values[0])
  for i in range(8):
    print copy_values[i]

cdef void counts_to_output(AVX.float8 counts, np.uint32_t [:, :] out_counts, int i, int j) nogil:
  cdef:
    float tmp_counts[8]
    int index
  AVX.to_mem(counts, &tmp_counts[0])
  for index in range(8):
    out_counts[i][j*8 + index] = <int> tmp_counts[index]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       np.complex64_t c, z
       np.float32_t [:,:] in_coords_real, in_coords_imag

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

    # Put the real and imaginary components in a non python data type
    in_coords_real = np.real(in_coords)
    in_coords_imag = np.imag(in_coords)

    # Define a bunch of 
    cdef:
      AVX.float8 c_real, c_imag, z_real, z_imag, iter_count, ones, over_four, temp_real, temp_imag, mag, four, zero
    with nogil:
        for i in prange(in_coords.shape[0], num_threads=4, schedule='static', chunksize=1):
            for j in range(in_coords.shape[1]/8):
                # Load the real and imaginary parts into AVX
                c_real = AVX.make_float8(
                  in_coords_real[i, j*8],
                  in_coords_real[i, j*8 + 1],
                  in_coords_real[i, j*8 + 2],
                  in_coords_real[i, j*8 + 3],
                  in_coords_real[i, j*8 + 4],
                  in_coords_real[i, j*8 + 5],
                  in_coords_real[i, j*8 + 6],
                  in_coords_real[i, j*8 + 7]
                )

                c_imag = AVX.make_float8(
                  in_coords_imag[i, j*8],
                  in_coords_imag[i, j*8 + 1],
                  in_coords_imag[i, j*8 + 2],
                  in_coords_imag[i, j*8 + 3],
                  in_coords_imag[i, j*8 + 4],
                  in_coords_imag[i, j*8 + 5],
                  in_coords_imag[i, j*8 + 6],
                  in_coords_imag[i, j*8 + 7]                
                )
                # with gil:
                #   print "C Real!"
                #   print_AVX(c_real)
                #   print "C IMAG!"
                #   print_AVX(c_imag)
                #   raw_input() 
                #c = in_coords[i, j]
                #
                # Initialize other things
                z_imag = AVX.float_to_float8(0.)
                z_real = AVX.float_to_float8(0.)
                four = AVX.float_to_float8(4.)
                zero = AVX.float_to_float8(0.)

                # An unchanging AVX with all ones, used several places
                ones = AVX.float_to_float8(1.)

                # Stores the iteration count for each of the 8 values
                iter_count = AVX.float_to_float8(0.)

                # 8 bits to store which data is still under 4 magnitude
                over_four = AVX.float_to_float8(0.)

                for iter in range(max_iterations):

                    # If all our values are greater than four then break
                    if AVX.signs(over_four) == 255:
                      break

                    # Increment the iteration count only for those that are under four
                    iter_count = AVX.add(iter_count, AVX.bitwise_andnot(over_four, ones))

                    # Store values for z^2
                    temp_real = AVX.sub(
                      AVX.mul(z_real, z_real),
                      AVX.mul(z_imag, z_imag)
                    )
                    temp_imag = AVX.add(
                      AVX.mul(z_real, z_imag),
                      AVX.mul(z_real, z_imag)
                    )

                    # Update z with z^2 + c
                    z_real = AVX.add(temp_real, c_real)
                    z_imag = AVX.add(temp_imag, c_imag)

                    # with gil:
                    #   print_AVX(z_real)
                    #   print_AVX(z_imag)

                    # Calculate the magnitude
                    mag = AVX.add(
                      AVX.mul(z_real, z_real),
                      AVX.mul(z_imag, z_imag)
                    )
                    # with gil:
                    #   print "Here"
                    #   print_AVX(over_four)
                    over_four = AVX.less_than(four, mag)
                    # with gil:
                    #   print "Magnitude!"
                    #   print_AVX(mag)
                    #   print "Over four"
                    #   print_AVX(over_four)
                    #   print AVX.signs(over_four)
                    #   raw_input()
                    #if magnitude_squared(z) > 4:
                    #    break
                    #z = z * z + c
                #out_counts[i, j] = iter
                    counts_to_output(iter_count, out_counts, i, j)



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
    print_AVX(avxval)

    # mask will be true where 2.0 < avxval
    mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)

    # invert mask and select off values, so should be 2.0 >= avxval
    avxval = AVX.bitwise_andnot(mask, avxval)

    AVX.to_mem(avxval, &(out_vals[0]))

    return np.array(out_view)
