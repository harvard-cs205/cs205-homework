import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(True) #change back to False
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       np.float32_t [:,:] realin_coords, imagin_coords, fout_counts
       np.complex64_t c, z
       AVX.float8 realzAVX, imagzAVX, realcAVX, imagcAVX, realztemp, imagztemp, modulez, iterAVX, mask
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

    # with nogil:
    #     for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=1):
    #         for j in range(in_coords.shape[1]):
    #             c = in_coords[i, j]
    #             z = 0
    #             for iter in range(max_iterations):
    #                 if magnitude_squared(z) > 4:
    #                     break
    #                 z = z * z + c
    #             out_counts[i, j] = iter

    realin_coords = np.real(in_coords)
    imagin_coords = np.imag(in_coords)
    fout_counts = np.zeros(np.shape(out_counts), dtype="float32")


    with nogil:
        for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=1):
            for j in range(0,in_coords.shape[1],8):
                # minus sign for calculation
                realcAVX = AVX.make_float8(realin_coords[i, j+7], realin_coords[i, j+6], realin_coords[i, j+5],
                realin_coords[i, j+4], realin_coords[i, j+3], realin_coords[i, j+2], realin_coords[i, j+1],
                realin_coords[i, j+0])

                imagcAVX = AVX.make_float8(imagin_coords[i, j+7], imagin_coords[i, j+6], imagin_coords[i, j+5],
                imagin_coords[i, j+4], imagin_coords[i, j+3], imagin_coords[i, j+2], imagin_coords[i, j+1],
                imagin_coords[i, j+0])

                realzAVX = AVX.float_to_float8(0.)
                imagzAVX = AVX.float_to_float8(0.)
                iterAVX = AVX.float_to_float8(0.)
                mask = AVX.float_to_float8(1)
                for iter in range(max_iterations):
                    if AVX.signs(mask) == 0:
                        break
                    # if magnitude_squared(z) > 4:
                    #     break

                    # real z = a+ib, a^2-(b^2-cr)
                    realztemp = AVX.fmsub(imagzAVX, imagzAVX, realcAVX)
                    realztemp = AVX.fmsub(realzAVX, realzAVX, realztemp)
                    # imag 2ab + ci
                    imagztemp = AVX.mul(imagzAVX, realzAVX)
                    imagztemp = AVX.fmsub(imagztemp, AVX.float_to_float8(2.), imagcAVX)
                    # update
                    realzAVX = realztemp
                    imagzAVX = imagztemp
                    # compute module a^2+b^2
                    modulez = AVX.mul(realzAVX, realzAVX)
                    modulez = AVX.fmadd(imagzAVX, imagzAVX, modulez)
                    # mask
                    mask = AVX.less_than(modulez, AVX.float_to_float8(2.))
                    # increment the ones to be incremented
                    iterAVX = AVX.add(iterAVX, AVX.bitwise_and(mask, AVX.float_to_float8(1.)) )

                AVX.to_mem(iterAVX, &(fout_counts[i, j]))
    out_counts = fout_counts.astype(int)


# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval, sumval, fromFloat, mask
        float out_vals[8]
        int temp
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

    fromFloat = AVX.float_to_float8(1.)
    avxval = AVX.sqrt(avxval)
    sumval = AVX.add(avxval, avxval)

    # Error
    # Illegal instruction (core dumped) !!!
    mask = AVX.less_than(sumval, fromFloat)
    temp = AVX.signs(mask)


    # AVX.to_mem(boolval, &(out_vals[0]))
    # print out_vals

    # return np.array(out_view)
    return temp
