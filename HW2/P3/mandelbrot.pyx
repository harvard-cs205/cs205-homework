import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

cdef void counts_to_output(AVX.float8 counts,
               np.uint32_t[:, :] out_counts,
               int i, int j) nogil:
  cdef:
    float out_vals[8]
    int k

  AVX.to_mem(counts, &(out_vals[0]))

  for k in range(8):
    out_counts[i, j + k] = <np.uint32_t> out_vals[k]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
         np.uint32_t [:, :] out_counts,
         int max_iterations=511):
  cdef:
    int i, j, iter
    int num_threads
    AVX.float8 counts, creal, cimag, zreal, zimag, temp_zreal, temp_zimag, mask, mag

  assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
  assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
  assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

  num_threads = 4

  with nogil:
    for i in prange(in_coords.shape[0], num_threads=num_threads, chunksize=1, schedule='static'):
      for j in range(0, in_coords.shape[1], 8):
        counts = AVX.float_to_float8(0.0)

        creal = AVX.make_float8((in_coords[i, j+7]).real,
                    (in_coords[i, j+6]).real,
                    (in_coords[i, j+5]).real,
                    (in_coords[i, j+4]).real,
                    (in_coords[i, j+3]).real,
                    (in_coords[i, j+2]).real,
                    (in_coords[i, j+1]).real,
                    (in_coords[i, j]).real)

        cimag = AVX.make_float8((in_coords[i, j+7]).imag,
                    (in_coords[i, j+6]).imag,
                    (in_coords[i, j+5]).imag,
                    (in_coords[i, j+4]).imag,
                    (in_coords[i, j+3]).imag,
                    (in_coords[i, j+2]).imag,
                    (in_coords[i, j+1]).imag,
                    (in_coords[i, j]).imag)

        zreal = AVX.float_to_float8(0.0)
        zimag = AVX.float_to_float8(0.0)

        for iter in range(max_iterations):
          counts = AVX.add(counts, AVX.bitwise_and(mask, AVX.float_to_float8(1.0)))
          
          mag = AVX.fmadd(zreal, zreal, AVX.mul(zimag, zimag))
          mask = AVX.less_than(mag, AVX.float_to_float8(4.0))
          
          if AVX.signs(mask) == 0:
            break
          
          temp_zreal = AVX.fmsub(zreal, zreal, AVX.mul(zimag, zimag))
          temp_zimag = AVX.fmadd(zreal, zimag, AVX.mul(zreal, zimag))
          zreal = AVX.add(temp_zreal, creal)
          zimag = AVX.add(temp_zimag, cimag)

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
