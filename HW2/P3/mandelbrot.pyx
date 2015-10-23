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
		int i, j, iter, si, nt = 8
		np.complex64_t c, z
		AVX.float8 toWrite, mask, creal, cimag, zreal, zimag
		float toWriteVal[8]

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
			#divide total iterations by 8
			for j in prange(in_coords.shape[1]/8, schedule='static', chunksize=1, num_threads=nt):
				with gil:
					si = j*8 #startIndex	

					toWrite = AVX.float_to_float8(1)
					zreal = AVX.float_to_float8(0)
					zimag = AVX.float_to_float8(0)
					for iter in range(max_iterations):
						mask = AVX.greater_than(AVX.add(AVX.mul(zreal,zreal),AVX.mul(zimag,zimag)),AVX.float_to_float8(4))

						if AVX.signs(mask)==0:break

						#Save iter for those > 4

						toWrite = AVX.bitwise_and(AVX.bitwise_and(mask,AVX.float_to_float8(iter)),toWrite)
						
						#Separate real and imag

						creal = AVX.make_float8(numpy.real(in_coords[i,si]),numpy.real(in_coords[i,si+1]),
											numpy.real(in_coords[i,si+2]),numpy.real(in_coords[i,si+3]),
											numpy.real(in_coords[i,si+4]),numpy.real(in_coords[i,si+5]),
											numpy.real(in_coords[i,si+6]),numpy.real(in_coords[i,si+7]))
						
						cimag = AVX.make_float8(numpy.imag(in_coords[i,si]),numpy.imag(in_coords[i,si+1]),
											numpy.imag(in_coords[i,si+2]),numpy.imag(in_coords[i,si+3]),
											numpy.imag(in_coords[i,si+4]),numpy.imag(in_coords[i,si+5]),
											numpy.imag(in_coords[i,si+6]),numpy.imag(in_coords[i,si+7]))
						
						zreal = AVX.sub(AVX.fmadd(zreal,zreal,creal),AVX.mul(zimag,zimag))
						zimag = AVX.add(AVX.mul(AVX.mul(AVX.float_to_float8(2),zreal),zimag),cimag)
					AVX.to_mem(toWrite, &(toWriteVal[0]))
					print 'toWriteVal=',np.array(toWriteVal)
					for idx,k in enumerate(reversed(toWriteVal)):
						out_counts[i,si+idx] = k
				#out_counts[i, j] = iter

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrotOld(np.complex64_t [:, :] in_coords,
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

	with nogil:
		for i in range(in_coords.shape[0]):
			for j in range(in_coords.shape[1]):
				c = in_coords[i, j]
				z = 0
				#with gil:
				#	print 'element ',i,' ',j
				for iter in range(max_iterations):
					if magnitude_squared(z) > 4:
						break
					z = z * z + c
				with gil:
					if j%8==0:
						print 'Row ',i,'col ',j
					print iter,
				out_counts[i, j] = iter

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot2(np.complex64_t[:, :] in_coords,
				 np.uint32_t[:, :] out_counts,
				 int max_iterations=511):
	cdef:
		int i, j, iter, si, nt = 8
		np.complex64_t c, z
		AVX.float8 toWrite, mask, creal, cimag, zreal,zrealtmp, zimag,zimagtmp,mag,
		float toWriteVal[8], iszero[8]

	assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
	assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
	assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"
	print 'num_threads=', nt
	with nogil:
		for i in range(in_coords.shape[0]):
			#divide total iterations by 8
			#for j in prange(in_coords.shape[1]/8, schedule='static', chunksize=1, num_threads=nt):
			for j in range(in_coords.shape[1]/8):
				with gil:
					si = j*8 #startIndex
					creal = AVX.make_float8(numpy.real(in_coords[i,si]),numpy.real(in_coords[i,si+1]),
											numpy.real(in_coords[i,si+2]),numpy.real(in_coords[i,si+3]),
											numpy.real(in_coords[i,si+4]),numpy.real(in_coords[i,si+5]),
											numpy.real(in_coords[i,si+6]),numpy.real(in_coords[i,si+7]))
						
					cimag = AVX.make_float8(numpy.imag(in_coords[i,si]),numpy.imag(in_coords[i,si+1]),
											numpy.imag(in_coords[i,si+2]),numpy.imag(in_coords[i,si+3]),
											numpy.imag(in_coords[i,si+4]),numpy.imag(in_coords[i,si+5]),
											numpy.imag(in_coords[i,si+6]),numpy.imag(in_coords[i,si+7]))
					#toWrite = AVX.float_to_float8(0)
					toWrite = AVX.float_to_float8(max_iterations)
					zreal = AVX.float_to_float8(0.0)
					zimag = AVX.float_to_float8(0.0)

					for iter in range(max_iterations):
						mag =AVX.add(AVX.mul(zreal,zreal),AVX.mul(zimag,zimag))
						mask = AVX.greater_than(mag,AVX.float_to_float8(4.0))
						
						#Save iter for those > 4
						toWrite = AVX.sub(toWrite,AVX.bitwise_and(mask,AVX.float_to_float8(max_iterations)))
						toWrite = AVX.add(AVX.bitwise_and(mask,AVX.float_to_float8(iter)),toWrite)
						AVX.to_mem(toWrite, &(toWriteVal[0]))

						
						zreal = AVX.bitwise_andnot(mask,zreal)
						zimag = AVX.bitwise_andnot(mask,zimag)
						creal = AVX.bitwise_andnot(mask,creal)
						cimag = AVX.bitwise_andnot(mask,cimag)
						
						#IF zreal==[0..0] and zimag==[0..0] and iter != 0 then break
						
						#check if all elements are 0 meaning all were >4 at some point
					
						
						if iter!= 0:
							AVX.to_mem(zreal, &(iszero[0]))#Write to array in order to check
							if np.count_nonzero(iszero)==0:
								AVX.to_mem(zimag, &(iszero[0]))
								if np.count_nonzero(iszero)==0:
									break
									#AVX.to_mem(creal, &(iszero[0]))
									#if np.count_nonzero(iszero)==0:
									#	AVX.to_mem(cimag, &(iszero[0]))
									#	if np.count_nonzero(iszero)==0:
									#		break
						
						zrealtmp = AVX.add(AVX.sub(AVX.mul(zreal,zreal),AVX.mul(zimag,zimag)),creal)
						zimagtmp=AVX.add(AVX.mul(AVX.float_to_float8(2.0),AVX.mul(zreal,zimag)),cimag)
						zreal=zrealtmp
						zimag=zimagtmp
					print 'Row ',i,'col ',j*8,' toWriteval=',toWriteVal
					for idx,k in enumerate(reversed(toWriteVal)):
						out_counts[i,si+idx] = k
				



cdef AVX.float8 float8FromArray(np.float32_t [:]a):
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
cpdef test(np.complex64_t[:, :] values):
	cdef:
		AVX.float8  mask, a,b, aaminusbb
		float out_vals1[8]
		float out_vals2[8]
		#float[:] out_view = out_vals
	#assert values.shape[0] == 8

	# mask will be true where 2.0 < avxval
	#mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)
	# inverts left FIRST before AND with right, so should be 2.0 >= avxval
	#avxval = AVX.bitwise_andnot(mask, avxval)
	#avxval = AVX.add(AVX.bitwise_and(avxval2,avxval),avxval)
	#avxval = AVX.fmadd(avxval,avxval,avxval)
	# Note that the order of the arguments here is opposite the direction when
	# we retrieve them into memory.

	print type(numpy.real(values[0])[0])
	#a = float8FromArray(numpy.real(values[0]))
	a = AVX.make_float8(numpy.real(values[0][0]),
						numpy.real(values[0][1]),
						numpy.real(values[0][2]),
						numpy.real(values[0][3]),
						numpy.real(values[0][4]),
						numpy.real(values[0][5]),
						numpy.real(values[0][6]),
						numpy.real(values[0][7]))
	b = AVX.make_float8(numpy.imag(values[0][0]),
						numpy.imag(values[0][1]),
						numpy.imag(values[0][2]),
						numpy.imag(values[0][3]),
						numpy.imag(values[0][4]),
						numpy.imag(values[0][5]),
						numpy.imag(values[0][6]),
						numpy.imag(values[0][7]))
	print 'values[0]',numpy.real(values[0])
	aaminusbb = AVX.sub(AVX.mul(a, a), AVX.mul(b, b))
	complex=AVX.mul(AVX.float_to_float8(2),AVX.mul(a, b))

	AVX.to_mem(aaminusbb, &(out_vals1[0]))
	AVX.to_mem(complex, &(out_vals2[0]))
	#return np.array(out_vals1)
	return [y+x*1j for y,x in zip(out_vals1,out_vals2)]
	
