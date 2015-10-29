import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange
from libc.stdlib cimport malloc, free

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
	return z.real * z.real + z.imag * z.imag


@cython.boundscheck(False)
@cython.wraparound(False)


cpdef mandelbrot_multrithreads_ILP(np.complex64_t [:, :] in_coords,
				np.uint32_t [:, :] out_counts,
				int n_threads,
				int n_elem,
				int max_iterations=511):
	cdef:
		int i,m_8,k, iterations
		AVX.float8 c_real, c_imag, iter_float8, z_mag
		AVX.float8 mask, z_float8_real, z_float8_imag, z_float8_real_new
		AVX.float8 four_float8, two_float8, one_float8, zero_float8

	four_float8 = AVX.float_to_float8(4.0)
	two_float8 = AVX.float_to_float8(2.0)
	one_float8 = AVX.float_to_float8(1.0)
	zero_float8 = AVX.float_to_float8(0.0)


	# parallize the rows with prange
	for i in prange(in_coords.shape[0], nogil = True, schedule = 'static', chunksize =1,
	 				num_threads = n_threads):


		# there are n_elem = 4000/8 = 500 vectors of 8 values per rows
		for m_8 in range(n_elem):

			z_float8_real = zero_float8
			z_float8_imag = zero_float8
			iter_float8 = zero_float8

			# store 8 adjacent values of the matrix in_coords into AVX
    		# Note that the order of the arguments here is opposite the direction when
    		# we retrieve them into memory.
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
			# now iterate and count teh number of iterations
			for iterations in range(max_iterations):

				# find the squared magnitude of z
				# |z|^2 = x^2 + y^2 where z = x + iy
				z_mag = AVX.add(AVX.mul(z_float8_real,z_float8_real),AVX.mul(z_float8_imag,z_float8_imag))

				# if the magnitude of all the 8 floats > 4 then stop iterating		
				if not AVX.signs(AVX.less_than(z_mag,four_float8)):
					break		

				# update z: z = z^2 + Re(c)

				# with z^2 = x^2 - y^2 + i 2xy where z = x+iy
				z_float8_real_new = AVX.sub(AVX.fmadd(z_float8_real, z_float8_real, c_real),
													 AVX.mul(z_float8_imag, z_float8_imag))


				# Im(z) = 2xy + Im(c)
				z_float8_imag = AVX.fmadd(AVX.mul(z_float8_real,two_float8) ,z_float8_imag,c_imag)
				z_float8_real = z_float8_real_new


				# mask will be true where |z|^2 < 4
				mask = AVX.less_than(z_mag, four_float8)

				# create mask with 1 if true and 0 if wrong
				mask = AVX.bitwise_and(mask, one_float8)


				# update iter_float8: add +1 when |z|^2 < 4 and +0 when |z|^2 > 4
				iter_float8 = AVX.add(iter_float8,mask)

				
				# Assign the iterations to out_counts
				for k in range(8):
					out_counts[i,m_8*8+k]= <np.uint32_t> (<np.float32_t *> &iter_float8)[k]			
			

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





