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


cdef void count_to_output(AVX.float8 count, np.uint32_t [:, :] out_counts, int i, int j) nogil:
    cdef:
        float tempCount[8]
        int k

    AVX.to_mem(count, &(tempCount[0]))

    for k in range(8):
        out_counts[i,j+k] = <np.uint32_t> tempCount[k]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j,k
       # np.complex64_t c, z
       AVX.float8 cReal, cImag, zReal, zImag, zRealxImag,zReal_old, magn, mask, ones, finalCounts

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

    
    for i in prange(in_coords.shape[0], nogil=True, num_threads=4, schedule='static', chunksize=1):
        for j in range(0,in_coords.shape[1],8):
            cReal = AVX.make_float8((in_coords[i, j+7]).real,
                         (in_coords[i, j+6]).real,
                         (in_coords[i, j+5]).real,
                         (in_coords[i, j+4]).real,
                         (in_coords[i, j+3]).real,
                         (in_coords[i, j+2]).real,
                         (in_coords[i, j+1]).real,
                         (in_coords[i, j+0]).real)

            cImag = AVX.make_float8((in_coords[i, j+7]).imag,
                         (in_coords[i, j+6]).imag,
                         (in_coords[i, j+5]).imag,
                         (in_coords[i, j+4]).imag,
                         (in_coords[i, j+3]).imag,
                         (in_coords[i, j+2]).imag,
                         (in_coords[i, j+1]).imag,
                         (in_coords[i, j+0]).imag)

            zReal = AVX.float_to_float8(0.0)
            zImag = AVX.float_to_float8(0.0)
            
            ones = AVX.float_to_float8(1.0)
            finalCounts = AVX.float_to_float8(0.0)

            for k in range(max_iterations):
                # computation 
                # real^2 - imag^2 + C
                zReal_old = zReal
                zReal = AVX.add(AVX.sub(AVX.mul(zReal,zReal),AVX.mul(zImag,zImag)),cReal)
                # real * imag
                zRealxImag = AVX.mul(zReal_old,zImag)
                # 2(real * imag) + Ci
                zImag = AVX.add(AVX.add(zRealxImag,zRealxImag), cImag)

                # magnitude ^ 2
                magn = AVX.add(AVX.mul(zReal, zReal), AVX.mul(zImag,zImag))
                mask = AVX.less_than(magn,AVX.float_to_float8(4.0))


                # if no more iterations to do
                if AVX.signs(mask) == 0:
                    break

                # add to counts
                finalCounts = AVX.add(finalCounts, AVX.bitwise_and(mask, ones))
                
            count_to_output(finalCounts, out_counts, i, j)


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
