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
cpdef mandelbrot(np.complex64_t[:, :] in_coords,
				 np.uint32_t[:, :] out_counts,
				 int max_iterations=511):
	cdef:
		int i, j, iter, nt = 8
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
	print 'num_threads=', nt
	with nogil:
		for i in range(in_coords.shape[0]):
				# Increment j by 8!!!!!! OR divide total iterations by 8
			for j in prange(in_coords.shape[1], schedule='static', chunksize=1, num_threads=nt):
				# Obtain lock here...Maybe don't need lock...
				# Convert c,z to float8
				c = in_coords[i, j]
				z = 0
				for iter in range(max_iterations):
					# Use greater_than() and not to get <= 4
					# z=bitwise_and(mask,z) ex: z=[0,0,1,3,2,2,0,4]
					# c=bitwise_and(mask,c) ex:
					# c=[0,0,3i-2j,2i+j,7i-j,3i+j,0,5i+j]
					if magnitude_squared(z) > 4:
						break
					#z = fmadd(z,z,c)
					z = z * z + c
				out_counts[i, j] = iter


cpdef test2():
	cdef:
		AVX.float8 b

	# returns 31 which is 00011111
	b = AVX.make_float8(1, 1, 0, -1, -1, -1, -1, -1)
	return AVX.signs(b)

cdef AVX.float8 float8FromArray(a):
	cdef:
		float o[8]
		AVX.float8 b
	assert len(a) == 8
	b = AVX.make_float8(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])
	# AVX.to_mem(b, &(o[0]))
	return b


cdef AVX.float8 intSignToPosNeg8(int sign):
	print 'sign=', sign
	print 'binary sign=', "{:b}".format(sign)
	tmp = [-1 if int(a) == 1 else 1 for a in "{:b}".format(sign)]
	while len(tmp) < 8:
		tmp = [1] + tmp
	assert len(tmp) == 8
	print 'tmp=', tmp
	return float8FromArray(tmp)


# An example using AVX instructions
cpdef test(np.complex64_t[:] values):
	cdef:

		AVX.float8 avxval, tmp, mask, a, b, ab, abminus, aaminusbb
		float out_vals1[8],out_vals2[8]
		#float[:] out_view = out_vals
		#np.float64_t[:] out3 = 
		#np.float64_t[:] out4 = 

	assert values.shape[0] == 8

	# mask will be true where 2.0 < avxval
	#mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)

	# inverts left FIRST before AND with right, so should be 2.0 >= avxval
	#avxval = AVX.bitwise_andnot(mask, avxval)
	# Note that the order of the arguments here is opposite the direction when
	# we retrieve them into memory.

	a = AVX.make_float8(numpy.real(values[0]),
						numpy.real(values[1]),
						numpy.real(values[2]),
						numpy.real(values[3]),
						numpy.real(values[4]),
						numpy.real(values[5]),
						numpy.real(values[6]),
						numpy.real(values[7]))
	b = AVX.make_float8(numpy.imag(values[0]),
						numpy.imag(values[1]),
						numpy.imag(values[2]),
						numpy.imag(values[3]),
						numpy.imag(values[4]),
						numpy.imag(values[5]),
						numpy.imag(values[6]),
						numpy.imag(values[7]))
	ab = AVX.mul(a, b)
	aaminusbb = AVX.sub(AVX.mul(a, a), AVX.mul(b, b))
	#tmp = intSignToPosNeg8(AVX.signs(ab))
	
	complex=AVX.mul(AVX.float_to_float8(2),ab)

	#avxval = AVX.add(AVX.bitwise_and(avxval2,avxval),avxval)
	#avxval = AVX.fmadd(avxval,avxval,avxval)
	AVX.to_mem(aaminusbb, &(out_vals1[0]))
	AVX.to_mem(complex, &(out_vals2[0]))
	
	return [y+x*1j for y,x in zip(out_vals1,out_vals2)]
	
