import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange, parallel


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

cdef AVX.float8 mag_squared(AVX.float8 reals, AVX.float8 imags) nogil:
    cdef:
        AVX.float8 mag
    mag = AVX.add(AVX.mul(imags, imags), AVX.mul(reals, reals))
    return mag
cdef void print_complex_AVX(AVX.float8 real,
                             AVX.float8 imag, int other) nogil:
    cdef:
        float real_parts[8]
        float imag_parts[8]
        int i

    AVX.to_mem(real, &(real_parts[0]))
    AVX.to_mem(imag, &(imag_parts[0]))
    with gil:
        for i in range(8):
            print("    {}: {}, {}, {}".format(i, real_parts[i], imag_parts[i], other))

cdef void counts_to_output(AVX.float8 counts, np.uint32_t [:, :] out_counts, int out_counts_i, int out_counts_j) nogil:
  cdef: 
    float out_vals[8]
    int elem
  AVX.to_mem(counts, &(out_vals[0]))
  for elem in range(8):
    out_counts[out_counts_i, out_counts_j+elem] = <int> out_vals[elem]

cdef AVX.float8 get_reals(np.complex64_t [:] items):
  cdef:
    np.ndarray[float, ndim=1] reals
    AVX.float8 real_avx

  reals = np.real(items)
  real_avx = AVX.make_float8(reals[7], reals[6], reals[5], reals[4], reals[3], reals[2], reals[1], reals[0])
  return real_avx
cdef AVX.float8 get_imags(np.complex64_t [:] items):
  cdef:
    np.ndarray[float, ndim=1] imags
    AVX.float8 real_avx
  imags = np.imag(items)
  imag_avx = AVX.make_float8(imags[7], imags[6], imags[5], imags[4], imags[3], imags[2], imags[1], imags[0])
  return imag_avx

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, elem, rep
       AVX.float8  c_real, c_imag, z_real, z_imag, counts, mags, mask, z_real_tmp, z_imag_tmp, ones_less

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"
    print in_coords.shape[0], in_coords.shape[1]

    with nogil:
      for i in prange(in_coords.shape[0], schedule='static', chunksize=1, num_threads=4):
        for j in range(0, in_coords.shape[1], 8):
          # Initialize real and imaginary parts of z
          z_real = AVX.float_to_float8(0)
          z_imag = AVX.float_to_float8(0)
          
          # Get the real and imaginary parts of the in_coords
          with gil:
            c_real = get_reals(in_coords[i, j:j+8])
            c_imag = get_imags(in_coords[i, j:j+8])
          
          # Initialize the counts
          counts = AVX.float_to_float8(0)

          for rep in range(max_iterations):
            # Get the magnitude squared
            mags = mag_squared(z_real, z_imag)
            # Get mask for the ones less than 4
            mask = AVX.less_than(mags, AVX.float_to_float8(4.0))

            # add a 1 to counts whenever it is less than 4.
            ones_less = AVX.bitwise_and(AVX.float_to_float8(1.), mask)
            counts = AVX.add(counts, ones_less)
            #print_complex_AVX(ones_less, ones_less, rep)
            
            # If None are less than 4, break
            if AVX.signs(mask) == 0:
              break

            # This does Z = z*z + c
            z_real_tmp = AVX.sub(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))
            z_imag_tmp = AVX.mul(AVX.mul(z_real, z_imag), AVX.float_to_float8(2.))
            z_real = AVX.add(z_real_tmp, c_real)
            z_imag = AVX.add(z_imag_tmp, c_imag)
          # Write the counts to the out_counts
          counts_to_output(counts, out_counts, i, j)

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
