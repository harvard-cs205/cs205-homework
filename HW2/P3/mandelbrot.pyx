import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free


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


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
				 np.uint32_t [:, :] out_counts,
				 int max_iterations=511):
	cdef:
		int i, j, iter
		np.complex64_t c, z


	assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
	assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
	assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

	with nogil:
		for i in range(in_coords.shape[0]):
			for j in range(in_coords.shape[1]):
				c = in_coords[i, j]
				z = 0
				for iter in range(max_iterations):
					if magnitude_squared(z) > 4:
						break
					z = z * z + c
				out_counts[i, j] = iter


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef p_mandelbrot(np.complex64_t [:, :] in_coords,
				   np.uint32_t [:, :] out_counts,
				   int nthread,
				   int max_iterations=511):
	cdef:
		int i, j, iter
		np.complex64_t c, z

	assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
	assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
	assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

	with nogil:
		for i in prange(in_coords.shape[0], num_threads=nthread, schedule='static', chunksize=1):
			for j in range(in_coords.shape[1]):
				c = in_coords[i, j]
				z = 0
				for iter in range(max_iterations):
					if magnitude_squared(z) > 4:
						break
					z = z * z + c
				out_counts[i, j] = max_iterations



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef avx_mandelbrot(np.complex64_t [:, :] in_coords,
					 np.uint32_t [:, :] out_counts,
					 int nthread,
					 int max_iterations=511):
	cdef:
		int i, j, iter, max_range
		np.complex64_t c
		np.float32_t [:, :] ic_real, ic_imag
		float out_vals[8]
		float [:] out_view = out_vals
		AVX.float8 z1, z2, z1_new, count, limit, c1, c2, mag_sq, gt_eval, increment

	assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
	assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
	assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

	ic_real = np.real(in_coords)
	ic_imag = np.imag(in_coords)
	max_range = in_coords.shape[1]+8
	
	with nogil, parallel(num_threads=nthread):
		buf = <float *> malloc(sizeof(float) * 8)
		

		for i in prange(in_coords.shape[0], schedule='static', chunksize=1):
			for j in range(0,in_coords.shape[1],8):

				c1 = AVX.make_float8(ic_real[i,j+7],
								     ic_real[i,j+6],
								     ic_real[i,j+5],
								     ic_real[i,j+4],
								     ic_real[i,j+3],
								     ic_real[i,j+2],
								     ic_real[i,j+1],
								     ic_real[i,j  ])

				c2 = AVX.make_float8(ic_imag[i,j+7],
								     ic_imag[i,j+6],
								     ic_imag[i,j+5],
								     ic_imag[i,j+4],
								     ic_imag[i,j+3],
								     ic_imag[i,j+2],
								     ic_imag[i,j+1],
								     ic_imag[i,j  ])

				count = AVX.float_to_float8(0.) # start at 1 so final increment is counted
				limit = AVX.float_to_float8(4.)
				z1 = AVX.float_to_float8(0.)
				z2 = AVX.float_to_float8(0.)

				for iter in range(max_iterations):

					mag_sq  = AVX.add( AVX.mul(z1,z1), AVX.mul(z2,z2) )
					gt_eval = AVX.greater_than ( mag_sq, limit )

					if AVX.signs( gt_eval ) == 255:
						print_complex_AVX(z1, z2)
						with gil:
							print 
							print
							print "Hit threshold with all of this AVX set"
							print
							print
						break
					else:
						increment = AVX.bitwise_andnot ( gt_eval, AVX.float_to_float8(1.) )
						count     = AVX.add         (   count, increment	)
						print_complex_AVX(z1, z2)
					# (a+bi)(c+di) == (ac - bd) + (ad + bc)i
					# when both terms are the same:
					# (a+bi)(a+bi) == (aa - bb) + (ab + ab)i == (aa + bb) + 2(ab)i
					# (z1+z2i)(z1+z2i) == (z1z1 - z2z2) + (z1z2 + z1z2)i == (z1z1 - z2z2) + 2(z1z2)i == newz1 + newz2i
					
					z1_new = AVX.sub( AVX.mul(z1,z1), AVX.mul(z2,z2) ) # squaring step z1
					z1_new = AVX.add( z1_new, c1 ) # add C1 step
					z2 = AVX.mul( AVX.mul(z1,z2), AVX.float_to_float8(2.) ) # squaring step z2
					z2 = AVX.add( z2, c2 ) # add C2 step
					z1 = z1_new


				print_complex_AVX(z1, z2)
				AVX.to_mem(count, &(buf[0]))
				with gil:
					print 
					print
					print "AVX set after full iter loop for indices {} through {}:".format(j,j+8)
					cts = []
					for i in range(8):
						cts.append(buf[i])
					print cts
					print
					print
				
				out_counts[i,j+0] = <np.uint32_t> buf[0]
				out_counts[i,j+1] = <np.uint32_t> buf[1]
				out_counts[i,j+2] = <np.uint32_t> buf[2]
				out_counts[i,j+3] = <np.uint32_t> buf[3]
				out_counts[i,j+4] = <np.uint32_t> buf[4]
				out_counts[i,j+5] = <np.uint32_t> buf[5]
				out_counts[i,j+6] = <np.uint32_t> buf[6]
				out_counts[i,j+7] = <np.uint32_t> buf[7]
	
					
		free(buf)

