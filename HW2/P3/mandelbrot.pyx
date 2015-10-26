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

    mandel_cal(in_coords)

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
  for i in prange(in_coords.shape[0], nogil=True, schedule='static', chunksize=1):
      for j in xrange(in_coords.shape[1]):
          c = in_coords[i, j]
          z = 0
          for iter in xrange(max_iterations):
              if magnitude_squared(z) > 4:
                  break
              z = z * z + c
          out_counts[i, j] = iter

  return out_counts

cpdef mandel_cal(np.complex64_t [:, :] in_coords):
    cdef:
        AVX.float8 real_avx, imag_avx, complex_avx, z_avx
        AVX.float8 real_chunk_avx, imag_chunk_avx

        np.float32_t [:,:] real_np, imag_np
        np.float32_t [:,:] real_tmp, imag_tmp
        np.float32_t real_vals[8]
        np.float32_t imag_vals[8]
        np.float32_t out_vals[8]
        np.float32_t [:] out_real = real_vals
        np.float32_t [:] out_imag = imag_vals
        np.float32_t [:] out_view = out_vals

    # Convert Complex to real and image float32 arrays
    real_np = np.real(in_coords)
    imag_np = np.imag(in_coords)
  
    # Initialize the real and imaginary AVX floats
    real_avx = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
    imag_avx = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
    
    # Initialize the magnitude vectors for real, imag, and output
    mag_out_avx = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
    mag_real_avx = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
    mag_imag_avx = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
    
    # Initialize cutoff AVX vector 
    cutoff = AVX.make_float8(4, 4, 4, 4, 4, 4, 4, 4)
    mult_two = AVX.make_float8(2, 2, 2, 2, 2, 2, 2, 2)
    
    # Initialize the z calcuation parts z = x^2 + 2xyi + yi^2
    z_int = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
    z_xsqur = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
    z_ysqur = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
    z_xysub = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)
    z_2xy = AVX.make_float8(0, 0, 0, 0, 0, 0, 0, 0)

    for i in xrange(in_coords.shape[0]):
       for j in xrange(0, in_coords.shape[1], 8):
           real_avx = AVX.make_float8(real_np[i, j], real_np[i, j+1], real_np[i, j+2], real_np[i, j+3],
            real_np[i, j+4], real_np[i, j+5], real_np[i, j+6], real_np[i, j+7])
           imag_avx = AVX.make_float8(imag_np[i, j], imag_np[i, j+1], imag_np[i, j+2], imag_np[i, j+3],
            imag_np[i, j+4], imag_np[i, j+5], imag_np[i, j+6], imag_np[i, j+7])

           z_xsqur = AVX.sqrt(real_avx)
           z_ysqur = AVX.sqrt(imag_avx)

           
    # for i in prange(in_coords.shape[0], nogil=True, schedule='static', chunksize=1):
    #     for j in xrange(in_coords.shape[1]):
    #         c = in_coords[i, j]
    #         z = 0
    #         for iter in xrange(max_iterations):
    #             if magnitude_squared(z) > 4:
    #                 break
    #             z = z * z + c
    #         out_counts[i, j] = iter

    return 0

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
    print(np.array(out_view))
    return np.array(out_view)
