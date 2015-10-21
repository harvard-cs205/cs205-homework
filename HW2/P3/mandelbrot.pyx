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
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=4):
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
        int max_iterations=511):
    cdef:
        int i, j, iter
       # To declare AVX.float8 variables, use:
       # cdef:
       #     AVX.float8 v1, v2, v3
       #
       # And then, for example, to multiply them
       #     v3 = AVX.mul(v1, v2)
       #
       # You may find the numpy.real() and numpy.imag() fuctions helpful.

       # Store the real and imaginary poarts of both z and c separately
       # Store the magnitude of z
       # Store masks for where |z|^2 > 4, |z|^2 < 4
       # Store the values of Re(z), Im(z), Re(c), Im(c) where |z|^2 < 4
       # And store the updated components of z
        AVX.float8 z_r_vals, z_z_vals, c_r_vals, c_z_vals, z_mag_cutoff, z_mag, z_mag_small, z_mag_big 
        AVX.float8 z_r_rel_floats, z_z_rel_floats, c_r_rel_floats, c_z_rel_floats
        AVX.float8 z_new_r, z_new_z, iter_counter
        AVX.float8 tmp1, tmp2, tmp3
        np.float64_t [:, :] in_coords_r, in_coords_z
        float new_out_counts[8]



    # Separate the in_coords into real and imaginary parts
    in_coords_r = np.real(in_coords)
    in_coords_z = np.imag(in_coords)

    # Get ourselves an AVX.float8 value which just contains '4' for the magnitude of z
    z_mag_cutoff = AVX.float_to_float8(4)

    # And also get ourselves something that just stores all 1's for incrementing
    all_ones = AVX.float_to_float8(1)

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    with nogil:
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=4):
            # We do a step size of 8 because we are going to handle 8 at a time
            # The assert above guarantees that we will not miss any items in doing so
            for j in range(in_coords.shape[1], 8):

                # Construct our AVX values - gotta be a better way to do this
                c_r_vals = AVX.make_float8(in_coords_r[i, j + 7],
                        in_coords_r[i, j + 6],
                        in_coords_r[i, j + 5],
                        in_coords_r[i, j + 4], 
                        in_coords_r[i, j + 3],
                        in_coords_r[i, j + 2],
                        in_coords_r[i, j + 1],
                        in_coords_r[i, j])

                c_z_vals = AVX.make_float8(in_coords_z[i, j + 7],
                        in_coords_z[i, j + 6],
                        in_coords_z[i, j + 5],
                        in_coords_z[i, j + 4], 
                        in_coords_z[i, j + 3],
                        in_coords_z[i, j + 2],
                        in_coords_z[i, j + 1],
                        in_coords_z[i, j])

                # Both the real and imaginary parts of z start as 0
                z_r_vals = AVX.float_to_float8(0)
                z_z_vals = AVX.float_to_float8(0)

                # And also count the number of iterations for each AVX value
                iter_counter = AVX.float_to_float8(0)

                for iter in range(max_iterations):
                    # For an imaginary number z = x + iy, |z|^2 = x^2 + y^2
                    # Calculate the magnitudes of each complex number in our AVX representation
                    z_mag = AVX.fmadd(z_r_vals, z_r_vals, AVX.mul(z_z_vals, z_z_vals))

                    # Now get a mask that tells us which of these are less than 4
                    # And another one that tells us which are greater than 4
                    # This lets us access the ones we want to modify, as well as later pick off the values
                    # That we did not modify
                    z_mag_small = AVX.less_than(z_mag, z_mag_cutoff)
                    z_mag_big = AVX.greater_than(z_mag, z_mag_cutoff)

                    # If nothing needs to be updated, we are done
                    if (AVX.signs(z_mag_small) == 0):
                        break

                    # Every z with |z|^2 less than 4 is getting updated - so increment accordingly
                    iter_counter = AVX.add(iter_counter, AVX.bitwise_and(all_ones, z_mag_small))

                    # Now we only want to iterate those floats which are less than 4 
                    # so separate their real and imaginary parts
                    z_r_rel_floats = AVX.bitwise_and(z_mag_small, z_r_vals)
                    z_z_rel_floats = AVX.bitwise_and(z_mag_small, z_z_vals)

                    # And we also want those components of C, because we only want to modify those
                    c_r_rel_floats = AVX.bitwise_and(z_mag_small, c_r_vals)
                    c_z_rel_floats = AVX.bitwise_and(z_mag_small, c_z_vals)

                    # Now we want to iterate each of these values according to
                    # z' = z * z + c
                    # For z = x + iy, this gives us:
                    # Re(z') = x^2 + Re(c) - y^2
                    # Im(z') = 2xy + Im(y)
                    
                    tmp1 = AVX.mul(z_r_rel_floats, z_r_rel_floats)
                    tmp1 = AVX.add(tmp1, c_r_rel_floats)
                    tmp2 = AVX.mul(z_z_rel_floats, z_z_rel_floats)
                    z_new_r = AVX.sub(tmp1, tmp2)


#                    z_new_r = AVX.sub(AVX.fmadd(z_r_rel_floats, z_r_rel_floats, c_r_rel_floats),
#                            AVX.mul(z_z_rel_floats, z_z_rel_floats))
                    z_new_z = AVX.fmadd(AVX.mul(AVX.float_to_float8(2), z_r_rel_floats), z_z_rel_floats, c_z_rel_floats)

                    # Now that we have the updated z values (for those that should be updated), we keep the 
                    # old, un-updated values and overwrite the new ones
                    z_r_vals = AVX.add(AVX.bitwise_and(z_r_vals, z_mag_big), z_new_r)
                    z_z_vals = AVX.add(AVX.bitwise_and(z_r_vals, z_mag_big), z_new_z)

                AVX.to_mem(iter_counter, &(new_out_counts[0]))
                for iter in range(8):
                    out_counts[i, j+iter] = <int> new_out_counts[iter]



# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval
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

    AVX.to_mem(avxval, &(out_vals[0]))

    return np.array(out_view)
