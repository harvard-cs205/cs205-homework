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
        for i in prange(in_coords.shape[0], num_threads=1, schedule='static', chunksize=1):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter



cdef void counts_to_output(AVX.float8 itercounts,
                           np.uint32_t[:, :] out_counts,
                           int i, int j) nogil:
    cdef:
        float mem_counts[8]
        int k

    AVX.to_mem(itercounts, &(mem_counts[0]))
    for k in range(8):
        out_counts[i, j+k] = <np.uint32_t> mem_counts[k]

cpdef mandelbrot2(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):

    cdef:
       int i, j, iter
       AVX.float8 c_real, c_imag, z_real, z_imag, mag, mask, iter_count, z_real_temp, z_imag_temp
       float[:, :] in_coords_real, in_coords_imag


    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    # Separate the real and imag part of in_coords into two arrays
    in_coords_real = np.real(in_coords)
    in_coords_imag = np.imag(in_coords)

    with nogil:
        # Parallelize i
        for i in prange(in_coords.shape[0], num_threads=1, schedule='static', chunksize=1):

            # Now we iterate through every 8 elements in each row
            for j in xrange(0, in_coords.shape[1], 8):

                # Grab 8 real/imag number at the same time
                c_real = AVX.make_float8(in_coords_real[i, j+7],
                                        in_coords_real[i, j+6],
                                        in_coords_real[i, j+5],
                                        in_coords_real[i, j+4],
                                        in_coords_real[i, j+3],
                                        in_coords_real[i, j+2],
                                        in_coords_real[i, j+1],
                                        in_coords_real[i, j])
                c_imag = AVX.make_float8(in_coords_imag[i, j+7],
                                         in_coords_imag[i, j+6],
                                         in_coords_imag[i, j+5],
                                         in_coords_imag[i, j+4],
                                         in_coords_imag[i, j+3],
                                         in_coords_imag[i, j+2],
                                         in_coords_imag[i, j+1],
                                         in_coords_imag[i, j])

                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)

                # Initiate the iter_count to be 0
                iter_count = AVX.float_to_float8(0.0)

                for iter in range(max_iterations):
                    # Calculate the magnitude squared
                    mag = AVX.add(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))
                    # Calculate the mask, which indicated digit wise whether the magnitude squared exceeds 4
                    mask = AVX.less_than(mag, AVX.float_to_float8(4.0))

                    # If all magnitudes are above 4, then break; else keep adding the mask to iter_count

                    if (AVX.signs(mask) == 0):
                        break

                    mask = AVX.bitwise_and(mask, AVX.float_to_float8(1.0))
                    iter_count = AVX.add(iter_count, mask)

                    # Calculate z for the next iteration
                    z_real_temp = AVX.sub(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))
                    z_imag_temp = AVX.add(AVX.mul(z_real, z_imag), AVX.mul(z_real, z_imag))
                    z_real = AVX.add(z_real_temp, c_real)
                    z_imag = AVX.add(z_imag_temp, c_imag)

                # Write iter_count to out_counts
                counts_to_output(iter_count, out_counts, i, j)