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


cpdef mandelbrot_AVX(np.complex64_t [:, :] in_coords,
				np.uint32_t [:, :] out_counts,
				int n_threads,
				int max_iterations=511):
	cdef:
		int i, j, iter, k
		np.float32_t [:, :] in_coords_real, in_coords_imag
		AVX.float8 c_real, c_imag, iter_AVX, z_mod
		AVX.float8 mask, z_real, z_imag, z_real_temp


	in_coords_real = np.real(in_coords)
	in_coords_imag = np.imag(in_coords)


	# parallize the rows with prange
	for i in prange(in_coords.shape[0], nogil = True, schedule = 'static', chunksize =1,
	 				num_threads = n_threads):


		for j in range(0,in_coords.shape[1],8):
			# define c_real (in the AVX 7 to 0 format)
			c_real = AVX.make_float8(in_coords_real[i, j+7], in_coords_real[i, j+6], in_coords_real[i, j+5],
			in_coords_real[i, j+4], in_coords_real[i, j+3], in_coords_real[i, j+2], in_coords_real[i, j+1],
			in_coords_real[i, j+0])
			# define c_imag
			c_imag = AVX.make_float8(in_coords_imag[i, j+7], in_coords_imag[i, j+6], in_coords_imag[i, j+5],
			in_coords_imag[i, j+4], in_coords_imag[i, j+3], in_coords_imag[i, j+2], in_coords_imag[i, j+1],
			in_coords_imag[i, j+0])
			# initialize z_real, z_imag and iter_AVX which is the output of the image
			z_real = AVX.float_to_float8(0.)
			z_imag = AVX.float_to_float8(0.)
			iter_AVX = AVX.float_to_float8(0.)
			mask = AVX.float_to_float8(-1.)


			# now iterate and count teh number of iterations
			for iter in range(max_iterations):

				# z = a+ib, real z is a^2-(b^2-cr)
				# temporary because still needed to compute z_imag
				z_real_temp = AVX.sub(AVX.fmadd(z_real, z_real, c_real),
													 AVX.mul(z_imag, z_imag))
				# imag 2ab + ci
				z_imag = AVX.fmadd(AVX.mul(AVX.float_to_float8(2.),z_real), z_imag, c_imag)
				z_real = z_real_temp

				# find the squared magnitude of z
				# |z|^2 = x^2 + y^2 where z = x + iy
				z_mod = AVX.add(AVX.mul(z_real, z_real), AVX.mul(z_imag, z_imag))

				# mask repeat while z_mod < 4
				mask = AVX.less_than(z_mod, AVX.float_to_float8(4.))

				#avoid unecessary iteration
				#when all the mask is 0 it means that there is no need to update
				if AVX.signs(mask) == 0:
					break

				# update iter_AVX:
				# AVX.bitwise_and(mask,  AVX.float_to_float8(1.)) gives a one only where mask is not zero
				iter_AVX = AVX.add(iter_AVX, AVX.bitwise_and(mask,  AVX.float_to_float8(1.)))


			# Assign the iterations to out_counts
			for k in range(8):
				out_counts[i,j+k]= <np.uint32_t> (<np.float32_t *> &iter_AVX)[k]


cpdef mandelbrot(np.complex64_t [:, :] in_coords,
				np.uint32_t [:, :] out_counts,
				int n_threads,
				int max_iterations=511):
	cdef:
		int i, j, iter
		np.complex64_t c, z

	assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
	assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
	assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

	# parallize the rows with prange
	for i in prange(in_coords.shape[0], nogil = True, schedule = 'static', chunksize =1,
					num_threads = n_threads):
		for j in range(in_coords.shape[1]):
			c = in_coords[i, j]
			z = 0
			for iter in range(max_iterations):
				if magnitude_squared(z) > 4:
					break
				z = z * z + c
			out_counts[i, j] = iter
	print type(np.real(z))



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

	#avxval = AVX.sqrt(avxval)

	#-----------------------------------------
	# FORMER VERSION
	#-----------------------------------------

	# mask will be true where 2.0 < avxval
	#mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)

	# invert mask and select off values, so should be 2.0 >= avxval
	#avxval = AVX.bitwise_andnot(mask, avxval)

	#AVX.to_mem(avxval, &(out_vals[0]))
	#-----------------------------------------


	# mask will be true where 2.0 < avxval
	mask = AVX.less_than(avxval,AVX.float_to_float8(10.0))
	print 'sign:',AVX.signs(mask)
	mask = AVX.bitwise_and(mask, AVX.float_to_float8(1.0))

	# invert mask and select off values, so should be 2.0 >= avxval
	avxval = AVX.add(avxval,mask)

	AVX.to_mem(mask, &(out_vals[0]))


	return np.array(out_view)

# A function to print complex AVX numbers
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
