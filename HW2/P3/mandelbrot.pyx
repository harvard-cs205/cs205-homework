import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange
from libc.stdlib cimport malloc, free

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
	return z.real * z.real + z.imag * z.imag


cdef AVX.float8 update_z_real(AVX.float8 z_real, 
			AVX.float8 z_imag, AVX.float8 c_real) nogil:

	# update z: z = z^2 + Re(c)
	# with z^2 = x^2 - y^2 + i 2xy where z = x+iy

	# x_s = x^2 + Re(c)
	cdef AVX.float8 x_s = AVX.fmadd(z_real, z_real, c_real)
	# y_s = y^2
	cdef AVX.float8 y_s = AVX.mul(z_imag, z_imag)

	return AVX.sub(x_s,y_s)

cdef void iter_to_mem(AVX.float8 iter, 
						np.uint32_t *to_big_matrix,
						int slice_start,
						int slice_end) nogil:
	cdef float *iter_view = <float *> malloc(8*sizeof(float))

	AVX.to_mem(iter, iter_view)

	# Now assign appropriately
	cdef int j
	cdef int count = 0
	for j in range(slice_start, slice_end):
		to_big_matrix[j] = <np.uint32_t> iter_view[count]
		count += 1

	free(iter_view)

@cython.boundscheck(False)
@cython.wraparound(False)


cpdef mandelbrot_multrithreads_ILP(np.complex64_t [:, :] in_coords,
				np.uint32_t [:, :] out_counts,
				int n_threads,
				int n_elem,
				int max_iterations=511):
	cdef:
		int i, j,m_8,j_m, iter
		np.complex64_t c, z
		AVX.float8 c_real, c_imag, iter_float8, z_mag
		AVX.float8 mask, mag_check, z_float8_real, z_float8_imag
		float out_vals[8]
		float [:] out_view = out_vals


	assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
	assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
	assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

	# parallize the rows with prange
	for i in prange(in_coords.shape[0], nogil = True, schedule = 'static', chunksize =1,
					num_threads = n_threads):
	# keep unparallelized for testing
	#for i in range(in_coords.shape[0]):

		for m_8 in range(n_elem):

			z_float8_real = AVX.float_to_float8(0.0)
			z_float8_imag = AVX.float_to_float8(0.0)
			iter_float8 = AVX.float_to_float8(0.0)

			for iter in range(max_iterations):

				# find magnitude of z
				z_mag = AVX.add(AVX.mul(z_float8_real,z_float8_real),AVX.mul(z_float8_imag,z_float8_imag))

				# if the magnitude of all the 8 float >4 stop iterating	
				if AVX.signs(AVX.less_than(AVX.float_to_float8(4.0), z_mag)) == 255:
					break				


				# generate avxval distinguishing real and imaginary parts

				c_real = AVX.make_float8(in_coords[i, m_8*8 + 7].real,
										in_coords[i, m_8*8 + 6].real,
										in_coords[i, m_8*8 + 5].real,
										in_coords[i, m_8*8 + 4].real,
										in_coords[i, m_8*8 + 3].real,
										in_coords[i, m_8*8 + 2].real,
										in_coords[i, m_8*8 + 1].real,
										in_coords[i, m_8*8 + 0].real)
				c_imag = AVX.make_float8(in_coords[i, m_8*8 + 7].imag,
										in_coords[i, m_8*8 + 6].imag,
										in_coords[i, m_8*8 + 5].imag,
										in_coords[i, m_8*8 + 4].imag,
										in_coords[i, m_8*8 + 3].imag,
										in_coords[i, m_8*8 + 2].imag,
										in_coords[i, m_8*8 + 1].imag,
										in_coords[i, m_8*8 + 0].imag)


				# update z: Re(z) = x^2 -y^2 + Re(c)
				z_float8_real = update_z_real(z_float8_real,z_float8_imag, c_real)
				# Im(z) = 2xy + Im(c)
				z_float8_imag = AVX.fmadd(AVX.mul(z_float8_real,AVX.float_to_float8(2)) ,z_float8_imag,c_imag)


				# mask will be true where 4.0 < avxval
				mask = AVX.less_than(z_mag, AVX.float_to_float8(4.0))

				# create mask with 1 if true and 0 if wrong
				mask = AVX.bitwise_and(mask, AVX.float_to_float8(1.0))


				# update iter_float8
				iter_float8 = AVX.add(iter_float8,mask)

				
				# Assign the iterations
				iter_to_mem(iter_float8,&out_counts[i,0], m_8*8, m_8*8+7)
			


			#for j_m in range(8):
			#	j = m_8*8 + j_m
			#	c = in_coords[i, j]
			#	z = 0
			#	for iter in range(max_iterations):
				#	if magnitude_squared(z) > 4:
					#	break
				#	z = z * z + c
			#	out_counts[i, j] = iter



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





