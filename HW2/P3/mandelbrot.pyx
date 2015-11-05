import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag


cdef void counts_to_output(AVX.float8 tmp_counts,
                            np.uint32_t [:,:] out_counts,
                            int i, int j) nogil:
    cdef:
        float counts[8]
        int x

    # put into counts float array
    AVX.to_mem(tmp_counts, &(counts[0]))


    # update out_counts
    for x in range(8):
        out_counts[i,8*j + x] = <int>counts[x]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       np.complex64_t c, z
       float[:,:] c_reals,c_imags


       # To declare AVX.float8 variables, use:
       # cdef:
       #     AVX.float8 v1, v2, v3
       #
       # And then, for example, to multiply them
       #     v3 = AVX.mul(v1, v2)
       #
       # You may find the numpy.real() and numpy.imag() fuctions helpful.
       AVX.float8 c_real, c_imag, z_real, z_imag, mask, tmp_counts, tmp

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    # get real and imaginary coords
    c_reals = np.real(in_coords)
    c_imags = np.imag(in_coords)


    with nogil:
        # multithread over rows i
        for i in prange(in_coords.shape[0],
                        schedule='static',
                        chunksize=1,
                        num_threads=4):

            # assuming columns is multiple of 8
            # take j as groups of 8. so j=0 is first group of 8,
            # j=1 is second group of 8, etc
            for j in range(in_coords.shape[1]/8):
                # get group of 8 real and imaginary
                c_real = AVX.make_float8(c_reals[i,8*j],
                                         c_reals[i,8*j+1],
                                         c_reals[i,8*j+2],
                                         c_reals[i,8*j+3],
                                         c_reals[i,8*j+4],
                                         c_reals[i,8*j+5],
                                         c_reals[i,8*j+6],
                                         c_reals[i,8*j+7])

                c_imag = AVX.make_float8(c_imags[i,8*j],
                                         c_imags[i,8*j+1],
                                         c_imags[i,8*j+2],
                                         c_imags[i,8*j+3],
                                         c_imags[i,8*j+4],
                                         c_imags[i,8*j+5],
                                         c_imags[i,8*j+6],
                                         c_imags[i,8*j+7])

                # z starts as 0's
                z_real = AVX.make_float8(0,0,0,0,0,0,0,0)
                z_imag = AVX.make_float8(0,0,0,0,0,0,0,0)
                # tmp_counts starts as 0's
                tmp_counts = AVX.make_float8(0,0,0,0,0,0,0,0)



                # update z and counts
                for iter in range(max_iterations):
                    # update z = z*z + c
                    # save copy of z_real as it is overwritten
                    tmp = z_real

                    z_real = AVX.add(c_real,
                             AVX.sub(AVX.mul(z_real,z_real),
                                     AVX.mul(z_imag,z_imag)))
                    z_imag = AVX.add(c_imag,
                             AVX.mul(AVX.float_to_float8(2.0),
                                     AVX.mul(tmp,z_imag)))

                    # make mask z < 4.0
                    mask = AVX.less_than(AVX.add(AVX.mul(z_real,z_real),
                                                 AVX.mul(z_imag,z_imag)),
                                         AVX.float_to_float8(4.0))

                    # if all z >= 4.0 stop the loop
                    if AVX.signs(mask) == 0.0:
                        break

                    # increment tmp_counts for z < 4
                    tmp_counts = AVX.add(
                                 AVX.add(AVX.bitwise_and(mask,tmp_counts),
                                         AVX.float_to_float8(1.0)),
                                 AVX.bitwise_andnot(mask,tmp_counts))

                    # update out_counts without multiple threads
                    # overwriting
                    counts_to_output(tmp_counts, out_counts,i,j)


                """
                # for without AVX
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter
                """

# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values,
                     np.complex64_t [:, :] in_coords,
                     np.uint32_t [:,:] out_counts):
    cdef:
        AVX.float8 avxval, tmp, mask, habba
        np.complex64_t[:,:] c1
        float [:,:] c2
        float out_vals[8]
        float [:] out_view = out_vals
        int i=0,j=0

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

    msk = AVX.less_than(avxval,AVX.float_to_float8(1.0))
    if AVX.signs(msk) == 0.0:
        avxval = AVX.add(avxval, AVX.float_to_float8(5.0))

    AVX.to_mem(avxval, &(out_vals[0]))


    z = AVX.make_float8(1,2,3,4,5,6,7,8)
    AVX.to_mem(z,&(out_vals[0]))
    print out_vals[0]
    print out_vals[1]
    print out_vals[7]

    c1 = in_coords
    c2 = np.real(in_coords)
    habba = AVX.make_float8(c2[i,8*j],
                            c2[i,8*j+1],
                            c2[i,8*j+2],
                            c2[i,8*j+3],
                            c2[i,8*j+4],
                            c2[i,8*j+5],
                            c2[i,8*j+6],
                            c2[i,8*j+7])

    out_counts[0,0] = 12
    print out_counts[0,0]



    AVX.to_mem(habba,&(out_vals[0]))

    print AVX.signs(AVX.less_than(z,AVX.float_to_float8(4.0)))

    print "hi"


    print np.array(out_view)



    return np.array(out_view)
