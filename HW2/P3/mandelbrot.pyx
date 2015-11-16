import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef void counts_to_output(AVX.float8 counts,
                      np.uint32_t[:, :] out_counts,
                      int i, int j) nogil:
    # Calling this function per thread allows us
    # to allocated a new pointer for tmp_counts on teh stack
    # for each thread, preventing collisions
    cdef:
        float tmp_counts[8] 
        int jj

    # Write out to memory.  Note that this in reverse order.
    AVX.to_mem(counts, &(tmp_counts[0]))
    
    # Cast all values as int
    for jj in range(8):
        out_counts[i, j+jj]= <int>tmp_counts[jj]
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       np.float32_t [:, :] in_coords_r, in_coords_i
       AVX.float8 counts, cr, ci, zr, zi, zr_temp, zi_temp, magnitude_squared, mask


    in_coords_r = np.real(in_coords)
    in_coords_i = np.imag(in_coords)

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
        # Multithreading over rows
        for i in prange(in_coords.shape[0], num_threads=1, schedule='static', chunksize=1) :
            for j in range(0, in_coords.shape[1], 8) :
                counts = AVX.float_to_float8(0.0)

                # The order of the arguments here is opposite the direction
                # that we will write counts into memory
                
                # Real components of c
                cr = AVX.make_float8(in_coords_r[i, j+7], 
                                in_coords_r[i, j+6], 
                                in_coords_r[i, j+5], 
                                in_coords_r[i, j+4], 
                                in_coords_r[i, j+3], 
                                in_coords_r[i, j+2], 
                                in_coords_r[i, j+1], 
                                in_coords_r[i, j])

                # Imaginary components of c
                ci = AVX.make_float8(in_coords_i[i, j+7], 
                                in_coords_i[i, j+6], 
                                in_coords_i[i, j+5], 
                                in_coords_i[i, j+4], 
                                in_coords_i[i, j+3], 
                                in_coords_i[i, j+2], 
                                in_coords_i[i, j+1], 
                                in_coords_i[i, j])

                zr = AVX.float_to_float8(0.0)
                zi = AVX.float_to_float8(0.0)

                for iter in range(max_iterations):
                    # Use fmadd to save an add instruction
                    magnitude_squared = AVX.fmadd(zr, zr, AVX.mul(zi,zi))
                    mask = AVX.less_than(magnitude_squared, AVX.float_to_float8(4.0))
                    
                    # If all the magnitudes are too high, stop incrementing counts.
                    if 0 == AVX.signs(mask):
                      break

                    #  Increment only the counts that have the correct magnitudes.
                    counts = AVX.add(counts, AVX.bitwise_and(AVX.float_to_float8(1.0), mask))

                    # (zr + zi*i)(zr + zi*i) + (cr + ci*i) = (zr * zr + cr - zi * zi) + (2 * zr * zi + ci)*i 
                    # We need temp variables since zr and zi are used for each calculation.
                    zr_temp = AVX.sub(AVX.fmadd(zr, zr, cr), AVX.mul(zi,zi))
                    zi_temp = AVX.fmadd(AVX.mul(AVX.float_to_float8(2.0), zr), zi, ci)
                    zr = zr_temp
                    zi = zi_temp

                # As stated on piazza, we use a function call to allocate a temporary stack
                # to write out the counts
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
