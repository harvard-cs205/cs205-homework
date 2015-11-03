import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

# helper debug function
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

# helper function to transfer AVX data to the array
cdef void avx_to_output(AVX.float8 counts, np.uint32_t[:, :] out_counts, int i, int j) nogil:
    cdef:
        float tmp_counts[8]
        int jj

    AVX.to_mem(counts, &(tmp_counts[0]))
    
    for jj in range(8):
        out_counts[i, j+jj]=<int>tmp_counts[jj]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t[:, :] in_coords,
                 np.uint32_t[:, :] out_counts,
                 int max_iterations=511):
    cdef:
        int i, j, iter
        np.complex64_t c, z
        np.float32_t realc
        np.float32_t imagc
        np.float32_t valzero
        np.float32_t valfour
        np.float32_t[:, :] real_coords
        np.float32_t[:, :] imag_coords
        AVX.float8 avxrealc, avximagc, avxrealz, avximagz, avxrealz_tmp, avximagz_tmp, avxmagnitude, avxiters, avxtmp, avxfours, avxones

        # To declare AVX.float8 variables, use:
        # cdef:
        #     AVX.float8 v1, v2, v3
        #
        # And then, for example, to multiply them
        #     v3 = AVX.mul(v1, v2)
        #
        # You may find the numpy.real() and numpy.imag() fuctions helpful.

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[
        0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[
        1],  "Input and output arrays must be the same size"

    # get real / imag parts
    real_coords = np.real(in_coords)
    imag_coords = np.imag(in_coords)

    # version multithreading and AVX instructions...
    for i in prange(in_coords.shape[0], nogil=True, num_threads=4, schedule='static', chunksize=1):
      for j in range(in_coords.shape[1] / 8):

        # parallelize 8 complex numbers
        # first separate real and imaginary parts
        realc = real_coords[i, j*8]
        imagc = imag_coords[i, j*8]

        avxrealc = AVX.make_float8(real_coords[i, j * 8 + 7],
                                   real_coords[i, j * 8 + 6],
                                   real_coords[i, j * 8 + 5],
                                   real_coords[i, j * 8 + 4],
                                   real_coords[i, j * 8 + 3],
                                   real_coords[i, j * 8 + 2],
                                   real_coords[i, j * 8 + 1],
                                   real_coords[i, j * 8 + 0])
        avximagc = AVX.make_float8(imag_coords[i, j * 8 + 7],
                                   imag_coords[i, j * 8 + 6],
                                   imag_coords[i, j * 8 + 5],
                                   imag_coords[i, j * 8 + 4],
                                   imag_coords[i, j * 8 + 3],
                                   imag_coords[i, j * 8 + 2],
                                   imag_coords[i, j * 8 + 1],
                                   imag_coords[i, j * 8 + 0])

        avxiters = AVX.make_float8(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        avxones  = AVX.make_float8(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        avxfours = AVX.make_float8(4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0) # to check for magnitude

        # now do the loop
        # init the series with c (we skip over the first step
        # which is unnecessary in the original implemntation)
        avxrealz = avxrealc
        avximagz = avximagc

        for iter in range(max_iterations):
          # for a given complex number z = x + iy
          # it holds z^2 = (x^2 - y^2) + i(2xy)
          # we can use that to compute z^2 + c for 8 complex numbers in parallel!

          # now one step
          avxrealz_tmp = AVX.sub(AVX.mul(avxrealz, avxrealz), AVX.mul(avximagz, avximagz))
          avximagz_tmp = AVX.mul(avxrealz, avximagz)
          avximagz_tmp = AVX.add(avximagz_tmp, avximagz_tmp) # adding is faster and equal to * 2

          # get new avxrealz by adding c
          avxrealz = AVX.add(avxrealz_tmp, avxrealc)
          avximagz = AVX.add(avximagz_tmp, avximagc)

          # compute magnitude
          avxmagnitude = AVX.add(AVX.mul(avxrealz, avxrealz), AVX.mul(avximagz, avximagz))

          # now check whether for each component the magnitude is already larger than 4
          # i.e. add to iterations
          avxtmp = AVX.less_than(avxmagnitude, avxfours)

          # break if no change occurs
          if AVX.signs(avxtmp) == 0:
            break

          avxtmp = AVX.bitwise_and(avxtmp, avxones)
          avxiters = AVX.add(avxiters, avxtmp)

        # write to output
        avx_to_output(avxiters, out_counts, i, j * 8)


    # #serial, original version
    # with nogil:
    #     for i in range(in_coords.shape[0]):
    #         for j in range(in_coords.shape[1]):
    #             c = in_coords[i, j]
    #             z = 0
    #             for iter in range(max_iterations):
    #                 if magnitude_squared(z) > 4:
    #                     break
    #                 z = z * z + c
    #             out_counts[i, j] = iter


    # # version using multithreading
    # for i in prange(in_coords.shape[0], nogil=True, num_threads=4, schedule='static', chunksize=1):
    #   for j in range(in_coords.shape[1]):
    #     c = in_coords[i, j]
    #     z = 0
    #     for iter in range(max_iterations):
    #       if magnitude_squared(z) > 4:
    #         break
    #       z = z * z + c
    #     out_counts[i, j] = iter


# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval, tmp, mask
        float out_vals[8]
        float[:] out_view = out_vals

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

    AVX.to_mem(avxval, & (out_vals[0]))

    return np.array(out_view)
