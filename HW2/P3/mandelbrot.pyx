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
                 int n_threads,
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
        for i in prange(in_coords.shape[0], 
                        schedule='static', 
                        chunksize=1,
                        num_threads=n_threads):
            for j in range(in_coords.shape[1]):
                c = in_coords[i, j]
                z = 0
                for iter in range(max_iterations):
                    if magnitude_squared(z) > 4:
                        break
                    z = z * z + c
                out_counts[i, j] = iter
    return out_counts
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot_avx (np.complex64_t [:, :] in_coords,
             np.uint32_t [:, :] out_counts,
             int n_threads,
             int max_iterations=511):
    cdef:
       int i, j, iter
       np.complex64_t c, z
       AVX.float8 c_real, c_imag, z_real, z_imag, z_mag, 
       AVX.float8 z_rprod, z_iprod, z_irprod, iter8, mask
       
       np.float32_t [:, :] in_coords_real, in_coords_imag
       float out_vals[8]

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

    for j in range(in_coords.shape[1] / 8):
        with nogil:
            for i in prange(in_coords.shape[0], 
                            schedule='static', 
                            chunksize=1,
                            num_threads=n_threads):

                z_real = AVX.float_to_float8(0)
                z_imag = AVX.float_to_float8(0)
                z_mag = AVX.float_to_float8(0)
                c_real = AVX.make_float8(in_coords_real[i, j * 8 + 7],
                         in_coords_real[i, j * 8 + 6],
                         in_coords_real[i, j * 8 + 5],
                         in_coords_real[i, j * 8 + 4],
                         in_coords_real[i, j * 8 + 3],
                         in_coords_real[i, j * 8 + 2],
                         in_coords_real[i, j * 8 + 1],
                         in_coords_real[i, j * 8])
                c_imag = AVX.make_float8(in_coords_imag[i, j * 8 + 7],
                         in_coords_imag[i, j * 8 + 6],
                         in_coords_imag[i, j * 8 + 5],
                         in_coords_imag[i, j * 8 + 4],
                         in_coords_imag[i, j * 8 + 3],
                         in_coords_imag[i, j * 8 + 2],
                         in_coords_imag[i, j * 8 + 1],
                         in_coords_imag[i, j * 8])

                # Start with all iteration counts at 0
                iter8 = AVX.float_to_float8(0)
                mask = AVX.less_than(z_mag, AVX.float_to_float8(4.0))
                for iter in range(max_iterations):
                    z_rprod = AVX.mul(z_real, z_real)
                    z_iprod = AVX.mul(z_imag, z_imag)
                    z_irprod = AVX.mul(z_real, z_imag)
                    z_mag = AVX.add(z_rprod, z_iprod)

                    # Determine which magnitudes are less than 4
                    # mask tells me which elements are under-bound
                    # and thus whose iteration counts need to be updated
                    iter8 = AVX.add(AVX.bitwise_and(mask, 
                            AVX.float_to_float8(iter)),
                            AVX.bitwise_andnot(mask, iter8))
                    mask = AVX.less_than(z_mag, AVX.float_to_float8(4.0))
    
                    if AVX.signs(mask) == 0:
                        break                                    

                    z_real = AVX.add(AVX.sub(z_rprod, z_iprod), 
                                     c_real)
                    z_imag = AVX.fmadd(z_irprod, 
                                     AVX.float_to_float8(2.0), 
                                     c_imag)
                    
                AVX.to_mem(iter8, &(out_vals[0]))

                out_counts[i, j * 8 + 7] = <np.uint32_t>out_vals[7]
                out_counts[i, j * 8 + 6] = <np.uint32_t>out_vals[6]
                out_counts[i, j * 8 + 5] = <np.uint32_t>out_vals[5]
                out_counts[i, j * 8 + 4] = <np.uint32_t>out_vals[4]
                out_counts[i, j * 8 + 3] = <np.uint32_t>out_vals[3]
                out_counts[i, j * 8 + 2] = <np.uint32_t>out_vals[2]
                out_counts[i, j * 8 + 1] = <np.uint32_t>out_vals[1]
                out_counts[i, j * 8] = <np.uint32_t>out_vals[0]
                
    return out_counts

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

    print_complex_AVX(avxval, mask)
    # invert mask and select off values, so should be 2.0 >= avxval
    avxval = AVX.bitwise_andnot(mask, avxval)

    print_complex_AVX(avxval, mask)    
    print(AVX.signs(mask))
    print(AVX.signs(AVX.float_to_float8(-1.0)))
    AVX.to_mem(avxval, &(out_vals[0]))

    return np.array(out_vals)[0]
