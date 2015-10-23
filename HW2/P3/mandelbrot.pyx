import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef void counts_to_output(AVX.float8 iterCount, np.uint32_t [:, :]out_counts, int i, int j) nogil:
    cdef:
        float tmpCount[8]
        int k

    AVX.to_mem(iterCount, tmpCount)
    for k in range(8):
        out_counts[i, j + k] = int(tmpCount[7 - k])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, k, iter
       np.complex64_t c, z

       AVX.float8 zreal
       AVX.float8 zimag
       AVX.float8 creal, cimag
       AVX.float8 zrealSqr, zimagSqr
       AVX.float8 newZreal, newZimag
       
       AVX.float8 iterCount, iterMask
       AVX.float8 magnitudes
       int maskSign
       # float outIterCount[8]
       #AVX.float8 allTrues = AVX.float_to_float8(
       # To declare AVX.float8 variables, use:
       # cdef:
       #     AVX.float8 v1, v2, v3
       #
       # And then, for example, to multiply them
       #     v3 = AVX.mul(v1, v2)
       #
       # You may find the numpy.real() and numpy.imag() fuctions helpful.
       np.float32_t [:, :] in_coords_real = np.real(in_coords)
       np.float32_t [:, :] in_coords_imag = np.imag(in_coords)

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    #with nogil:
    for i in prange(in_coords.shape[0], nogil=True, schedule='static', chunksize=1, num_threads=4):
        #for j in range(in_coords.shape[1]):
        for j in range(0, in_coords.shape[1], 8):
            iterCount = AVX.float_to_float8(0.0)
            creal = AVX.make_float8(in_coords_real[i, j],
                                        in_coords_real[i, j+1],
                                        in_coords_real[i, j+2],
                                        in_coords_real[i, j+3],
                                        in_coords_real[i, j+4],
                                        in_coords_real[i, j+5],
                                        in_coords_real[i, j+6],
                                        in_coords_real[i, j+7])
            cimag = AVX.make_float8(in_coords_imag[i, j],
                                        in_coords_imag[i, j+1],
                                        in_coords_imag[i, j+2],
                                        in_coords_imag[i, j+3],
                                        in_coords_imag[i, j+4],
                                        in_coords_imag[i, j+5],
                                        in_coords_imag[i, j+6],
                                        in_coords_imag[i, j+7])

            zreal, zimag = AVX.float_to_float8(0.0), AVX.float_to_float8(0.0)
            for iter in range(max_iterations):
                zrealSqr = AVX.mul(zreal, zreal)
                zimagSqr = AVX.mul(zimag, zimag)
                magnitudes = AVX.add(zrealSqr, zimagSqr)
                # Check if all is greater than 4.0
                iterMask = AVX.less_than(magnitudes, AVX.float_to_float8(4.0))
                maskSign = AVX.signs(iterMask)
                if maskSign == 0:
                    break
                iterCount = AVX.add(iterCount, AVX.bitwise_and(iterMask, AVX.float_to_float8(1.0)))
                # iterCount = AVX.add(iterCount, AVX.bitwise_and(iterMask, AVX.float_to_float8(1.0)))
                # newZ = z * z + c
                # z = zr + zi * i
                # c = cr + ci * i
                # z * z = zr^2 - zi^2 + 2*zr*zi * i
                # newZr = zr^2 - zi^2 + cr
                # newZi = (2*zr*zi + ci) * i
                newZreal = AVX.add(AVX.sub(zrealSqr, zimagSqr), creal)
                newZimag = AVX.add(AVX.mul(AVX.float_to_float8(2.0), AVX.mul(zreal, zimag)), cimag)

                zreal = newZreal # AVX.bitwise_and(iterMask, newZreal)
                zimag = newZimag # AVX.bitwise_and(iterMask, newZimag)
            
            counts_to_output(iterCount, out_counts, i, j)
            '''
            c = in_coords[i, j]
            z = 0
            for iter in range(max_iterations):
                if magnitude_squared(z) > 4:
                    break
                z = z * z + c
            out_counts[i, j] = iter
            if i == 2500 and j == 2000:
                with gil:
                    print out_counts[i, j]
           ''' 
# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval, tmp, mask
        float out_vals[8]
        float [:] out_view = out_vals
        int sign
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

    #AVX.to_mem(avxval, &(out_vals[0]))
    sign = AVX.signs(mask)
    AVX.to_mem(mask, &(out_vals[0]))
    print 'Sign: {0}'.format(sign)
    return np.array(out_view)
