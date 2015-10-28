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
                 int max_iterations=511,
                 int num_threads=1):
    cdef:
       int i, j, l, iteration
       float k, sum
       float[8] mask_mem, counts_mem
       np.complex64_t c, z
       AVX.float8 c_reals, c_imags, z_real, z_imag, counts, magnitudes_squared, greater4, mask, current_count

       int nt = num_threads
       AVX.float8 compare4 = AVX.float_to_float8(4.0)
       AVX.float8 compare1 = AVX.float_to_float8(1.0)


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
        for i in prange(in_coords.shape[0], num_threads=nt, schedule="static", chunksize=1):
            for j in range(0, in_coords.shape[1], 8):
                # c = in_coords[i, j]
                c_reals = AVX.make_float8(in_coords[i, j].real, in_coords[i, j+1].real, in_coords[i, j+2].real,
                                        in_coords[i, j+3].real, in_coords[i, j+4].real, in_coords[i, j+5].real,
                                        in_coords[i, j+6].real, in_coords[i, j+7].real)

                c_imags = AVX.make_float8(in_coords[i, j].imag, in_coords[i, j+1].imag, in_coords[i, j+2].imag,
                                        in_coords[i, j+3].imag, in_coords[i, j+4].imag, in_coords[i, j+5].imag,
                                        in_coords[i, j+6].imag, in_coords[i, j+7].imag)

                # z = 0
                z_real = AVX.float_to_float8(0.0)
                z_imag = AVX.float_to_float8(0.0)
                counts = AVX.float_to_float8(0.0)

                for iteration in range(max_iterations):
                    # if magnitude_squared(z) > 4
                    magnitudes_squared = AVX.mul(z_real, z_real)
                    magnitudes_squared = AVX.fmadd(z_imag, z_imag, magnitudes_squared)
                    greater4 = AVX.less_than(compare4, magnitudes_squared)

                    # Mask to add counts to those bigger than magnitude 4 and a 0 count
                    mask = AVX.bitwise_and(AVX.less_than(counts, compare1), greater4)

                    # Update count
                    current_count = AVX.float_to_float8(<float> iteration)
                    counts = AVX.add(AVX.bitwise_and(mask, current_count), counts)

                    # Check if done
                    sum = 0.0
                    AVX.to_mem(greater4, mask_mem)
                    for k in mask_mem:
                        sum = sum + k

                    # Break if done
                    if sum == -8.0:
                        break

                    # z = z * z + c
                    # Real z
                    z_real = AVX.fmadd(z_real, z_real, z_real)
                    z_real = AVX.sub(z_real, AVX.mul(z_imag, z_imag))

                    # Imaginary z
                    z_imag = AVX.fmadd(z_imag, z_real, z_imag)
                    z_imag = AVX.fmadd(z_real, z_imag, z_imag)

                    # Add c
                    z_real = AVX.add(z_real, c_reals)
                    z_imag = AVX.add(z_imag, c_imags)

                # Store result
                AVX.to_mem(counts, counts_mem)
                out_counts[i, j] = <np.uint32_t> counts_mem[0]
                out_counts[i, j+1] = <np.uint32_t> counts_mem[1]
                out_counts[i, j+2] = <np.uint32_t> counts_mem[2]
                out_counts[i, j+3] = <np.uint32_t> counts_mem[3]
                out_counts[i, j+4] = <np.uint32_t> counts_mem[4]
                out_counts[i, j+5] = <np.uint32_t> counts_mem[5]
                out_counts[i, j+6] = <np.uint32_t> counts_mem[6]
                out_counts[i, j+7] = <np.uint32_t> counts_mem[7]


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
