import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef AVX.float8 magnitude_squared_avx(AVX.float8 r, AVX.float8 i) nogil:
    return AVX.add(AVX.mul(i, i), AVX.mul(r,r))

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
        for i in prange(in_coords.shape[0],schedule='static', chunksize=1,num_threads=4):
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
cpdef mandelbrot2(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter, mask_signs, ind
       AVX.float8 cnew_r, cnew_i, znew_r, znew_i, mags, iters, ones, fours, mask, twos, znew_i_tmp, znew_r_tmp
       np.complex64_t c, z
       float out_vals[8]
       float [:] out_view = out_vals

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
        # Define constants which we will use repeatedly.
        fours = AVX.float_to_float8(4.0)
        ones = AVX.float_to_float8(1.0)
        twos = AVX.float_to_float8(2.0)
        for i in prange(in_coords.shape[0],schedule='static', chunksize=1, num_threads=4):

            # Iterate over the input coordinates in steps of 8.
            for j in range(0, in_coords.shape[1], 8):

                # Real and complex parts of cs: read these in.
                cnew_r = AVX.make_float8(in_coords[i, j + 7].real,
                        in_coords[i, j + 6].real,
                        in_coords[i, j + 5].real,
                        in_coords[i, j + 4].real,
                        in_coords[i, j + 3].real,
                        in_coords[i, j + 2].real,
                        in_coords[i, j + 1].real,
                        in_coords[i, j].real)
                cnew_i = AVX.make_float8(in_coords[i, j + 7].imag,
                        in_coords[i, j + 6].imag,
                        in_coords[i, j + 5].imag,
                        in_coords[i, j + 4].imag,
                        in_coords[i, j + 3].imag,
                        in_coords[i, j + 2].imag,
                        in_coords[i, j + 1].imag,
                        in_coords[i, j].imag)

                # Initialize the zs to all 0s.
                znew_r = AVX.float_to_float8(0.0)
                znew_i = AVX.float_to_float8(0.0)
                mags = AVX.float_to_float8(0.0)
                iters = AVX.float_to_float8(0.0)
                for iter in range(max_iterations):
                  mags = magnitude_squared_avx(znew_r, znew_i)
                  mask = AVX.less_than(mags, fours)

                  # This condition means everything had magn. > 4, so we're done.
                  if AVX.signs(mask) == 0:
                    break

                  # Compute z^2 = (a^2 - b^2) + (2ab)i
                  znew_r_tmp = AVX.sub(AVX.mul(znew_r, znew_r), AVX.mul(znew_i, znew_i)) # a^2 - b^2
                  znew_i_tmp = AVX.mul(twos, AVX.mul(znew_r, znew_i)) # 2ab

                  # Add c to conclude
                  znew_r = znew_r_tmp
                  znew_i = znew_i_tmp
                  znew_r = AVX.add(znew_r, cnew_r)
                  znew_i = AVX.add(znew_i, cnew_i)

                  # This line adds to iterations only for the ones where magnitude < 4.
                  iters = AVX.add(iters, AVX.bitwise_and(mask, ones))
                AVX.to_mem(iters, &(out_vals[0]))

                # Copy out the values we need.
                for ind in range(8):
                  out_counts[i, j + ind] = <int> out_vals[ind]

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

# used as: print example_sqrt_8(np.arange(8, dtype=np.float32))
