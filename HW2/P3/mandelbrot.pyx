import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

# cdef np.float32_t avx_mag_square(np.ndarray[np.float32_t, ndim=8] avx_z):
#     return AVX.sqrt(avx_z)

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

    
    # out_counts = mandel_multi(in_coords, out_counts, max_iterations)
    out_counts = mandel_cal(in_coords, out_counts, max_iterations)

cpdef serial_code():
    # Serial Code
    # with nogil:
    #     for i in range(in_coords.shape[0]):
    #         for j in range(in_coords.shape[1]):
    #             c = in_coords[i, j]
    #             z = 0
    #             for iter in range(max_iterations):
    #                 if magnitude_squared(z) > 4:
    #                     break
    #                 z = z * z + c
    #             out_counts[i, j] = iter
    return 0

cpdef mandel_multi(np.complex64_t [:, :] in_coords, 
                   np.uint32_t [:, :] out_counts,
                   int max_iterations=511):
  cdef:
       int i, j, iter
       np.complex64_t c, z

  # Completed Multithreading Code
  for i in prange(in_coords.shape[0], nogil=True, schedule='static', chunksize=1, num_threads=1):
      for j in xrange(in_coords.shape[1]):
          c = in_coords[i, j]
          z = 0
          for iter in xrange(max_iterations):
              if magnitude_squared(z) > 4:
                  break
              z = z * z + c
          out_counts[i, j] = iter

  return out_counts

cpdef mandel_cal(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    
    cdef:
        AVX.float8 real_avx, imag_avx, complex_avx
        AVX.float8 z_real, z_imag
        AVX.float8 mag_real_avx, mag_imag_avx, mag_tot_avx
        AVX.float8 cutoff_avx, two_avx, one_avx, mask_avx, compare_avx, count_avx, c_plus_one
        AVX.float8 z_x2_cr, z_r, z_i, z_xy

        np.float32_t [:,:] real_np, imag_np

        int i, j, k, iter

    # Convert Complex to real and image float32 arrays
    real_np = np.real(in_coords)
    imag_np = np.imag(in_coords)
  
    # Initialize the real and imaginary AVX floats
    real_avx = AVX.float_to_float8(0)
    imag_avx = AVX.float_to_float8(0)
    
    # Initialize the magnitude vectors for real, imag, and output
    mag_real_avx = AVX.float_to_float8(0)
    mag_imag_avx = AVX.float_to_float8(0)
    mag_tot_avx = AVX.float_to_float8(0)
    
    # Initialize cutoff AVX vector 
    cutoff_avx = AVX.float_to_float8(4)
    two_avx = AVX.float_to_float8(2)
    one_avx = AVX.float_to_float8(1)
    compare_avx = AVX.float_to_float8(0)
    mask_avx = AVX.float_to_float8(0)
    
    for i in prange(in_coords.shape[0], nogil=True, schedule='static', chunksize=1, num_threads=4):
    #for i in xrange(in_coords.shape[0]):
        for j in range(0, in_coords.shape[1], 8):
            real_avx = AVX.make_float8(real_np[i, j], real_np[i, j+1], real_np[i, j+2], real_np[i, j+3],
              real_np[i, j+4], real_np[i, j+5], real_np[i, j+6], real_np[i, j+7])
            imag_avx = AVX.make_float8(imag_np[i, j], imag_np[i, j+1], imag_np[i, j+2], imag_np[i, j+3],
              imag_np[i, j+4], imag_np[i, j+5], imag_np[i, j+6], imag_np[i, j+7])

            # z = 0
            z_real = AVX.float_to_float8(0)
            z_imag = AVX.float_to_float8(0)
            z_x2_cr = AVX.float_to_float8(0)
            z_xy = AVX.float_to_float8(0)
            z_r = AVX.float_to_float8(0)
            z_i = AVX.float_to_float8(0)

            count_avx = AVX.float_to_float8(0)
            # c_plus_one = AVX.float_to_float8(0)
            # compare_avx = AVX.float_to_float8(0)

            for iter in range(max_iterations):
                # Compute Magnitude 
                mag_real_avx = AVX.mul(z_real, z_real)
                mag_tot_avx = AVX.fmadd(z_imag, z_imag, mag_real_avx)
               
                # Only modify if less than 4
                compare_avx = AVX.less_than(mag_tot_avx, cutoff_avx)
                mask_avx = AVX.bitwise_and(one_avx, compare_avx)
                count_avx = AVX.add(count_avx, mask_avx)

                if not AVX.signs(compare_avx):
                  break

                z_r = AVX.fmsub(z_real, z_real, AVX.fmsub(z_imag, z_imag, real_avx))
                z_imag = AVX.fmadd(two_avx, AVX.mul(z_real, z_imag), imag_avx)
                z_real = z_r
              
                #AVX.to_mem(count_avx, &())
            for k in range(8):
                out_counts[i,j+k] = <np.uint32_t> ((<np.float32_t*> &count_avx)[k])

    return out_counts

# An example using AVX instructions
cpdef example_sqrt_8(np.float32_t[:] values):
    cdef:
        AVX.float8 avxval, tmp, mask, new
        float out_vals[8], out_new[8]
        float [:] out_view = out_vals
        float [:] new_view = out_new

    assert values.shape[0] == 8
    new = AVX.float_to_float8(5)

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
    AVX.to_mem(new, &(out_new[0]))
    
    print("Out View:", np.array(out_view))
    print("New:", np.array(new_view))

    return np.array(out_view)
