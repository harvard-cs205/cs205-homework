# cython: profile=True
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
#cpdef mandelbrot2(np.complex64_t[:, :] in_coordsReal,
#				 np.float32_t[:, :] out_counts,
#				 int max_iterations=511):
cpdef mandelbrot2(np.float32_t[:, :] in_coordsReal,
				 np.float32_t[:,:] in_coordsImag,
				 np.float32_t[:, :] out_counts,
				 int max_iterations=511):
	cdef:
		int i, j, k, iter, si, toWriteValLength, nt = 8,interval=8
		np.complex64_t c, z
		AVX.float8 toWrite, mask, creal, cimag, zreal,zrealtmp, zimag,zimagtmp,mag,
		float iszero[8]
		float toWriteVal[8]
		float notAnum
		#np.float32_t toWriteVal[8]
	assert in_coordsReal.shape[1] % 8 == 0, "Input array must have 8N columns"
	assert in_coordsReal.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
	assert in_coordsReal.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"
	
	notAnum = float('nan')
	with nogil:
		for i in xrange(in_coordsReal.shape[0]):
		#divide total iterations by 8
			for j in prange(in_coordsReal.shape[1]/8, schedule='static', chunksize=1, num_threads=nt):#for j in xrange(in_coordsReal.shape[1]/8):
				si = j*8 #startIndex
				creal = AVX.make_float8(in_coordsReal[i,si],in_coordsReal[i,si+1],
										in_coordsReal[i,si+2],in_coordsReal[i,si+3],
										in_coordsReal[i,si+4],in_coordsReal[i,si+5],
										in_coordsReal[i,si+6],in_coordsReal[i,si+7])
					
				cimag = AVX.make_float8(in_coordsImag[i,si],in_coordsImag[i,si+1],
										in_coordsImag[i,si+2],in_coordsImag[i,si+3],
										in_coordsImag[i,si+4],in_coordsImag[i,si+5],
										in_coordsImag[i,si+6],in_coordsImag[i,si+7])
				toWrite = AVX.float_to_float8(max_iterations)
				zreal = AVX.float_to_float8(0.0)
				zimag = AVX.float_to_float8(0.0)	
				for iter in xrange(max_iterations):
				#Compute magnitude
					mag =AVX.add(AVX.mul(zreal,zreal),AVX.mul(zimag,zimag))
				#Mask those greater than 4
					mask = AVX.greater_than(mag,AVX.float_to_float8(4.0))
				#Save iter to toWrite array for those greater than 4
					toWrite = AVX.add(AVX.bitwise_and(mask,AVX.float_to_float8(iter)),AVX.sub(toWrite,AVX.bitwise_and(mask,AVX.float_to_float8(max_iterations))))
					AVX.to_mem(toWrite, &(toWriteVal[0]))	
				#Get data less than or equal to 4
					zreal = AVX.bitwise_andnot(mask,zreal)
					zimag = AVX.bitwise_andnot(mask,zimag)
					creal = AVX.bitwise_andnot(mask,creal)
					cimag = AVX.bitwise_andnot(mask,cimag)
					
				#IF zreal==[0..0] and zimag==[0..0] and iter != 0 then break	
					if iter!= 0:
						AVX.to_mem(AVX.less_than(AVX.float_to_float8(0),zreal), &(iszero[0]))
						if (iszero[0]!=notAnum and iszero[1]!=notAnum and iszero[2]!=notAnum and iszero[3]!=notAnum and iszero[4]!=notAnum and iszero[5]!=notAnum and iszero[6]!=notAnum and iszero[7]!=notAnum):
							AVX.to_mem(AVX.greater_than(AVX.float_to_float8(0),zreal), &(iszero[0]))
							if (iszero[0]!=notAnum and iszero[1]!=notAnum and iszero[2]!=notAnum and iszero[3]!=notAnum and iszero[4]!=notAnum and iszero[5]!=notAnum and iszero[6]!=notAnum and iszero[7]!=notAnum):
								break

				#Compute z=z*z+c for real/complex parts separately
					zrealtmp = AVX.add(AVX.sub(AVX.mul(zreal,zreal),AVX.mul(zimag,zimag)),creal)
					zimagtmp = AVX.add(AVX.mul(AVX.float_to_float8(2.0),AVX.mul(zreal,zimag)),cimag)
					zreal=zrealtmp
					zimag=zimagtmp
			#Write the toWrite array to out_counts
				for k in xrange(interval-1,-1,-1):
					out_counts[i,si+interval-1-k] = toWriteVal[k]

	
#cdef inline int checkZero(float [:]iszero):
#	return np.count_nonzero(iszero)

#cpdef test3(np.uint32_t [:,:]c):
#	cdef np.uint32_t [:] n
#	n = np.array([99,99])
#	c[1,0:2]=n#

#cpdef test2(np.uint32_t[:] out_counts):
#	cdef:
#		int idx,i,a
#		AVX.float8 b
#		np.float32_t toWriteVal[8]
#	with nogil:
#		#a = toWriteVal.shape[0]
#		for i in range(out_counts.shape[0]-1,0,-1):
#			if i==4:
#				break
#			with gil:
#				out_counts[i] = np.log(99)
#		b = AVX.float_to_float8(1)
			

# An example using AVX instructions
#cpdef test(np.complex64_t[:, :] values):
#	cdef:
#		AVX.float8  mask, a,b, aaminusbb
#		float out_vals1[8]
#		float out_vals2[8]
#		#float[:] out_view = out_vals
#	#assert values.shape[0] == 8#

#	# mask will be true where 2.0 < avxval
#	#mask = AVX.less_than(AVX.float_to_float8(2.0), avxval)
#	# inverts left FIRST before AND with right, so should be 2.0 >= avxval
#	#avxval = AVX.bitwise_andnot(mask, avxval)
#	#avxval = AVX.add(AVX.bitwise_and(avxval2,avxval),avxval)
#	#avxval = AVX.fmadd(avxval,avxval,avxval)
#	# Note that the order of the arguments here is opposite the direction when
#	# we retrieve them into memory.#

#	print type(numpy.real(values[0])[0])
#	#a = float8FromArray(numpy.real(values[0]))
#	a = AVX.make_float8(numpy.real(values[0][0]),
#						numpy.real(values[0][1]),
#						numpy.real(values[0][2]),
#						numpy.real(values[0][3]),
#						numpy.real(values[0][4]),
#						numpy.real(values[0][5]),
#						numpy.real(values[0][6]),
#						numpy.real(values[0][7]))
#	b = AVX.make_float8(numpy.imag(values[0][0]),
#						numpy.imag(values[0][1]),
#						numpy.imag(values[0][2]),
#						numpy.imag(values[0][3]),
#						numpy.imag(values[0][4]),
#						numpy.imag(values[0][5]),
#						numpy.imag(values[0][6]),
#						numpy.imag(values[0][7]))
#	print 'values[0]',numpy.real(values[0])
#	aaminusbb = AVX.sub(AVX.mul(a, a), AVX.mul(b, b))
#	complex=AVX.mul(AVX.float_to_float8(2),AVX.mul(a, b))#

#	AVX.to_mem(aaminusbb, &(out_vals1[0]))
#	AVX.to_mem(complex, &(out_vals2[0]))
#	#return np.array(out_vals1)
#	return [y+x*1j for y,x in zip(out_vals1,out_vals2)]
	


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrotOld(np.complex64_t [:, :] in_coords,
                np.uint32_t [:, :] out_counts,
                int max_iterations=511):
	cdef:
		int i, j, iter
		np.complex64_t c, z#
	assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
	assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
	assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"#
	with nogil:
		for i in range(in_coords.shape[0]):
			for j in range(in_coords.shape[1]):
				c = in_coords[i, j]
				z = 0
				for iter in range(max_iterations):
					if magnitude_squared(z) > 4:
						break
					z = z * z + c
				#with gil:
				#	if j%8==0:
				#		print 'Row ',i,'col ',j
				#	print iter,
				out_counts[i, j] = iter#

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef AVX.float8 getReal(np.complex64_t[:,:] in_coords,int i,int si):
#	return AVX.make_float8(np.real(in_coords[i,si]),np.real(in_coords[i,si+1]),
#										np.real(in_coords[i,si+2]),np.real(in_coords[i,si+3]),
#										np.real(in_coords[i,si+4]),np.real(in_coords[i,si+5]),
#										np.real(in_coords[i,si+6]),np.real(in_coords[i,si+7]))	#
#

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef inline AVX.float8 getImag(np.complex64_t[:,:] in_coords,int i,int si):
#	return AVX.make_float8(np.imag(in_coords[i,si]),np.imag(in_coords[i,si+1]),
#										np.imag(in_coords[i,si+2]),np.imag(in_coords[i,si+3]),
#										np.imag(in_coords[i,si+4]),np.imag(in_coords[i,si+5]),
#										np.imag(in_coords[i,si+6]),np.imag(in_coords[i,si+7]))
