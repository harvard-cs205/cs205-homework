cimport numpy as np
import numpy
cimport cython
cimport AVX
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)

cdef void counts_to_output(AVX.float8 counts, np.uint32_t [:, :] out_counts,
                  int i, int j, int max_iterations) nogil:
  # Writes counts to out_counts in a threadsafe way.
  cdef:
    float tmp_counts[8]
    int k

  AVX.to_mem(counts, &(tmp_counts[0]))

  for k in range(8):
    out_counts[i, j+k] = <np.uint32_t> tmp_counts[k]

cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):

    cdef:
       int i, j, k, iter
       np.float64_t [8] nextr, nexti
       float [:, :] reals, imags
       AVX.float8 re, im, mag_sq, mask, z_re, z_im, z_re_new, counts

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

    reals = numpy.real(in_coords)
    imags = numpy.imag(in_coords)

    with nogil:
      for i in prange(in_coords.shape[0], num_threads=4, schedule='static', chunksize=1):
        for j in xrange(0, in_coords.shape[1], 8):
          re = AVX.make_float8(reals[i, j+7],
            reals[i, j+6],
            reals[i, j+5], 
            reals[i, j+4],
            reals[i, j+3],
            reals[i, j+2],
            reals[i, j+1],
            reals[i, j])
          im = AVX.make_float8(imags[i, j+7],
            imags[i, j+6],
            imags[i, j+5], 
            imags[i, j+4],
            imags[i, j+3],
            imags[i, j+2],
            imags[i, j+1],
            imags[i, j])

          z_re = AVX.float_to_float8(0.)
          z_im = AVX.float_to_float8(0.)
          counts = AVX.float_to_float8(0.)

          # The computation takes place in this loop. The counts are only updated
          # for threads whose magnitude is < 4. This is accomplished by masking with
          # an AVX.float8 of all ones and adding the result to count.
          # The rest of the lines are just the regular Mandelbrot computations
          # rephrased in AVX. 
          for iter in range(max_iterations):
            z_re_new = AVX.add(AVX.sub(AVX.mul(z_re, z_re), AVX.mul(z_im, z_im)), re)
            z_im = AVX.fmadd(AVX.mul(z_re, z_im), AVX.float_to_float8(2.), im)
            z_re = z_re_new
            mag_sq = AVX.add(AVX.mul(z_re, z_re), AVX.mul(z_im, z_im))
            mask = AVX.less_than(mag_sq, AVX.float_to_float8(4.))
            counts = AVX.add(counts, AVX.bitwise_and(mask, AVX.float_to_float8(1.)))

          counts_to_output(counts, out_counts, i, j, max_iterations)