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

	''' Added prange() to outer loop to allow for multithreading '''
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

	''' Added instruction-level parallelism with AVX'''
	cdef:
		int i, j, iter, max_range
		np.complex64_t c
		np.float32_t [:, :] ic_real, ic_imag
		float out_vals[8]
		float [:] out_view = out_vals
		AVX.float8 z1, z2, z1_new, c1, c2, mag_sq, lt_eval, sign_check, increment, count, limit

	assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
	assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
	assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"

	'''AVX doesn't have support for a complex number type, but we can mimic multiply and add operations using
	   only the real-valued components of the real and imaginary parts of a complex number.  np.real and np.imag,
	   respectively, pull out these components from the in_coords matrix of complex numbers.
	'''
	ic_real = np.real(in_coords)
	ic_imag = np.imag(in_coords)
	max_range = in_coords.shape[1]+8
	
	with nogil, parallel(num_threads=nthread): #num_threads is for prange(), but needs to be defined here.

		'''buf servers as a thread-local buffer for storing iteration counts on the float8 in each thread.
		   we write buf to out_counts at the end of each iter loop
		'''
		buf = <float *> malloc(sizeof(float) * 8)
		

		for i in prange(in_coords.shape[0], schedule='static', chunksize=1):
			for j in range(0,in_coords.shape[1],8):

				''' See longer comment below for more on mimicking a complex number data type.
				    For now, note that we let complex number C = c1 + c2i.
				'''
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

				count = AVX.float_to_float8(0.) # increment counter
				limit = AVX.float_to_float8(4.) # threshold against which to compare magnitude-squared 
				z1 = AVX.float_to_float8(0.) # coefficient of real-valued component of complex number
				z2 = AVX.float_to_float8(0.) # coefficient of imag-valued component of complex number

				for iter in range(max_iterations):

					mag_sq  = AVX.add( AVX.mul(z1,z1), AVX.mul(z2,z2) ) # compute magnitude squared of z
					lt_eval = AVX.less_than ( mag_sq, limit ) # less_than comparison against limit (see Piazza @413)

					''' The way we use lt_eval to check for stopping, as well as to increment, is a little confusing.
					    There are lots of ands/andnot shennanigans, mainly due to the Inf->NaN issue noted in Piazza @413.
					    Here's an example of what happens:

					    We arrive at a point where an Inf and a NaN are going to be combined by z1 (real) and z2 (imag):
					    Z1:
							[             nan   1.07974923e+19   7.14108400e+07  -2.52881387e+04
							  -6.27973572e+02   6.01599789e+00  -3.91640902e+00  -1.02400398e+00]

						Z2:
							[            -inf  -8.63728671e+18   1.70362976e+08   1.69796938e+05
							   2.76288062e+03  -6.29116859e+01  -2.45457911e+00  -4.54195887e-01]

						magnitude_squared:
							[             nan   1.91188564e+38   3.41230513e+16   2.94704906e+10
							   8.02786050e+06   3.99407227e+03   2.13632183e+01   1.25487804e+00]

						Note that in our magnitude_squared float 8, 
						AVX slots 1 through 6 (zero-indexed) are already above the threshold of 4.  
						Actually, slot 0 is too, as it got to NaN by being really big.
						So we need to keep iterating only on slot 7, and we want to stop counting on the others.
						
						But NaN will always return false, no matter what it's compared to. So we don't want to 
						compare mag_sq > 4, because the NaN should come out as True, but it will return False.
						
						Since it's going to be NaN anyway, we can just flip the comparison to mag_sq < 4. NaN still
						doesn't care what we're comparing to, but at least it gives us the right answer considering it's 
						actualy a big number gone wrong.  With mag_sq < 4, we get this for our float8:

							[  0.   0.   0.   0.   0.   0.   0.  nan] 
							(0 = false, nan = True)

						Now we have the correct comparison results.  NaN here is (weirdly) a synonym for -1, which
						is (equally weirdly) a synonym for True. Fine. When the entire float8 above is 0, we can 
						stop iterating. To see if we need to keep iterating, we can check the signed bits of each 
						slot with andnot(-1). In this case, bitwise_andnot gives 127, which makes sense as the last
						slot is worth 128 (255-128=127). And we can update our count tracker by using and(1), which
						in this case gives the float8 to add to our running count as:

							[ 0.  0.  0.  0.  0.  0.  0.  1.]

						Which is what we want.  This is a roundabout way of getting what we want, but it works.
					'''
					sign_check = AVX.bitwise_andnot( lt_eval, AVX.float_to_float8(-1.) ) 

					if AVX.signs( sign_check ) == 255:
						break
					else:
						increment = AVX.bitwise_and( lt_eval, AVX.float_to_float8(1.) )
						count     = AVX.add( count, increment)

					''' Operations on Complex Numbers:

						On each iteration, we want to perform the update:

							z = z * z + c

						But we don't have an explicitly-typed complex number to work with.
						Addition and multiplication of complex numbers can be performed using the real-valued
						coefficients: 

							a, b, x, y 
							for c = a + bi
								z = x + yi

							Multiplication:
							c * c = (a + bi)(a + bi)
								  = (a * a - b * b) + 2(a * b)i

							Addition:
							z + c = (x + yi) + (a + bi)
								  = (x + a) + (y + b)i
					'''

					z1_new = AVX.sub( AVX.mul(z1,z1), AVX.mul(z2,z2) ) # squaring step z1
					z1_new = AVX.add( z1_new, c1 ) # add C1 step
					z2 = AVX.mul( AVX.mul(z1,z2), AVX.float_to_float8(2.) ) # squaring step z2
					z2 = AVX.add( z2, c2 ) # add C2 step
					z1 = z1_new


				AVX.to_mem(count, &(buf[0])) # write count to buffer
				
				out_counts[i,j+0] = <np.uint32_t> buf[0] # write buffer to out_counts
				out_counts[i,j+1] = <np.uint32_t> buf[1]
				out_counts[i,j+2] = <np.uint32_t> buf[2]
				out_counts[i,j+3] = <np.uint32_t> buf[3]
				out_counts[i,j+4] = <np.uint32_t> buf[4]
				out_counts[i,j+5] = <np.uint32_t> buf[5]
				out_counts[i,j+6] = <np.uint32_t> buf[6]
				out_counts[i,j+7] = <np.uint32_t> buf[7]
	
					
		free(buf) # free buf!

