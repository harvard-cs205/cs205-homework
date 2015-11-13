import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag


cdef void counts_to_output(AVX.float8 iterations, np.float32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float temp[8]
        int p

    AVX.to_mem(iterations, &(temp[0]))
    for p in range(8):
        out_counts[i, j*8+p] = <int> temp[p]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.float32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter, p
       #np.complex64_t c, z
       AVX.float8 real_part,imag_part,old_real,old_imag,z_real,z_imag,iterations,mul_real,mul_imag,magnitude,mask,ones,iter_new
       float temp[8]

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
        #for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=2):
        #    for j in range(in_coords.shape[1]):
        #        c = in_coords[i, j]
        #        z = 0
        #        for iter in range(max_iterations):
        #            if magnitude_squared(z) > 4:
        #                break
        #            z = z * z + c
        #        out_counts[i, j] = iter
  

        # my code for AVX part
        for i in range(in_coords.shape[0]):
            for j in prange(in_coords.shape[1]/8, num_threads=2):
                # make 8 floating-point values for real and imaginary parts
                real_part = AVX.make_float8(in_coords[i, j*8+7].real,
                                            in_coords[i, j*8+6].real,
                                            in_coords[i, j*8+5].real,
                                            in_coords[i, j*8+4].real,
                                            in_coords[i, j*8+3].real,
                                            in_coords[i, j*8+2].real,
                                            in_coords[i, j*8+1].real,
                                            in_coords[i, j*8].real)

                imag_part = AVX.make_float8(in_coords[i, j*8+7].imag,
                                            in_coords[i, j*8+6].imag,
                                            in_coords[i, j*8+5].imag,
                                            in_coords[i, j*8+4].imag,
                                            in_coords[i, j*8+3].imag,
                                            in_coords[i, j*8+2].imag,
                                            in_coords[i, j*8+1].imag,
                                            in_coords[i, j*8].imag)

                # declare z to be zero               
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)
                iterations = AVX.float_to_float8(0.0)

                for iter in range(max_iterations):
                    # calculate magnitude
                    mul_real = AVX.mul(z_real, z_real)
                    mul_imag = AVX.mul(z_imag, z_imag)
                    magnitude = AVX.add(mul_real, mul_imag)
                    
                    ones = AVX.float_to_float8(1.0)
                    # true where magnitude < 4
                    mask = AVX.less_than(magnitude, AVX.float_to_float8(4.0))
                    # invert mask
                    iter_new = AVX.bitwise_and(mask, ones)
                    iterations = AVX.add(iterations, iter_new)
                    
                    # calculate z*z+c for real and imag parts
                    old_real = z_real
                    old_imag = z_imag
                    z_real = AVX.add(AVX.sub(AVX.mul(old_real,old_real),AVX.mul(old_imag,old_imag)), real_part)
                    z_imag = AVX.add(AVX.mul(AVX.add(ones,ones), AVX.mul(old_real,old_imag)), imag_part)

                counts_to_output(iterations, out_counts, i, j)
                    


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
