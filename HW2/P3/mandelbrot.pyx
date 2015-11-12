import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef void print_complex_AVX(AVX.float8 real,
                             AVX.float8 imag, int row, int col) nogil:
    cdef:
        float real_parts[8]
        float imag_parts[8]
        int i

    AVX.to_mem(real, &(real_parts[0]))
    AVX.to_mem(imag, &(imag_parts[0]))
    with gil:
        for i in range(8):
            print("    {} ({},{}): {}, {}".format(i, row, col, real_parts[i], imag_parts[i]))

cdef void counts_to_output(AVX.float8 counts,
                      np.uint32_t [:, :] out_counts,
                      int i, int j) nogil:
    cdef:
        int k
        float tmp_counts[8]
    
    AVX.to_mem(counts, &(tmp_counts[0]))

    for k in range(8):
        out_counts[i,j+k] = <np.uint32_t> tmp_counts[k]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511,
                 int num_threads=1):
    cdef:
        int i, j, iter
        int NT = num_threads 
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

    with nogil:
        for i in prange(in_coords.shape[0], schedule="static", chunksize=1, num_threads = NT):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot_avx(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511,
                 int num_threads=1):
    cdef:
        int i, j, iter
        int NT = num_threads
        float [:,:] in_coords_real, in_coords_imag
        AVX.float8 c_real, c_imag
        AVX.float8 z_real_old, z_imag_old, z_real, z_imag
        AVX.float8 magnitude_squared
        AVX.float8 mask
        AVX.float8 iter_max_counts, iter_counts, counts
        AVX.float8 avx_iter, avx_0, avx_2, avx_4

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
    
    in_coords_real = np.real(in_coords) 
    in_coords_imag = np.imag(in_coords)

    with nogil:
        avx_0 = AVX.float_to_float8(0.0)
        avx_1 = AVX.float_to_float8(1.0)
        avx_2 = AVX.float_to_float8(2.0)
        avx_4 = AVX.float_to_float8(4.0)
        for i in prange(in_coords.shape[0], schedule="static", chunksize=1, num_threads = NT):
            for j in range(0,in_coords.shape[1],8):
                # Initialize before starting iterations
                c_real = AVX.make_float8(in_coords_real[i,j+7],
                                         in_coords_real[i,j+6],
                                         in_coords_real[i,j+5],
                                         in_coords_real[i,j+4],
                                         in_coords_real[i,j+3],
                                         in_coords_real[i,j+2],
                                         in_coords_real[i,j+1],
                                         in_coords_real[i,j])

                c_imag = AVX.make_float8(in_coords_imag[i,j+7],
                                         in_coords_imag[i,j+6],
                                         in_coords_imag[i,j+5],
                                         in_coords_imag[i,j+4],
                                         in_coords_imag[i,j+3],
                                         in_coords_imag[i,j+2],
                                         in_coords_imag[i,j+1],
                                         in_coords_imag[i,j])

                z_real = avx_0
                z_imag = avx_0
                counts = avx_0
                for iter in range(max_iterations):
                    # Compute the magnitude_squared
                    magnitude_squared = AVX.add(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))
                    # Generate mask only when magnitude_squared is less than 4
                    mask = AVX.less_than(magnitude_squared,avx_4)
                    # Increment by one only for masked counts
                    counts = AVX.add(counts,AVX.bitwise_and(mask,avx_1)) 
                    # if all magnitude_squareds are greater than 4, then break
                    if AVX.signs(mask)==0:
                        break
                    # Update new z values
                    z_real_old = z_real
                    z_imag_old = z_imag
                    z_real = AVX.add(AVX.sub(AVX.mul(z_real_old,z_real_old),\
                                             AVX.mul(z_imag_old,z_imag_old)),c_real)
                    z_imag = AVX.add(AVX.mul(avx_2,AVX.mul(z_real_old,z_imag_old)),c_imag)

                # Write back to out_counts
                counts_to_output(counts, out_counts, i, j)


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
