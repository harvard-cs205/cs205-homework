# cython: profile=True
import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
	return z.real * z.real + z.imag * z.imag
   

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.float32_t[:, :] in_coordsReal,
				 np.float32_t[:,:] in_coordsImag,
				 np.float32_t[:, :] out_counts,
				 int max_iterations=511):
	cdef:
		int i, j, k, iter, nt = 8,sign,
		AVX.float8 toWrite, mask, creal, cimag, zreal,zrealtmp, zimag,zimagtmp,mag,four,two,zrealsqr,zimagsqr,one,addToIter
	#assert in_coordsReal.shape[1] % 8 == 0, "Input array must have 8N columns"
	#assert in_coordsReal.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
	#assert in_coordsReal.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"
	
	with nogil:
		four = AVX.float_to_float8(4.0)
		two = AVX.float_to_float8(2.0)
		one = AVX.float_to_float8(1.0)
		for i in xrange(in_coordsReal.shape[0]):
			for j in prange(0, in_coordsReal.shape[1], 8, schedule='static', chunksize=1, num_threads=nt):
				#Separate real and imaginary parts for computation
				creal = AVX.make_float8(in_coordsReal[i,j+7],in_coordsReal[i,j+6],
										in_coordsReal[i,j+5],in_coordsReal[i,j+4],
										in_coordsReal[i,j+3],in_coordsReal[i,j+2],
										in_coordsReal[i,j+1],in_coordsReal[i,j])
					
				cimag = AVX.make_float8(in_coordsImag[i,j+7],in_coordsImag[i,j+6],
										in_coordsImag[i,j+5],in_coordsImag[i,j+4],
										in_coordsImag[i,j+3],in_coordsImag[i,j+2],
										in_coordsImag[i,j+1],in_coordsImag[i,j])

				toWrite = AVX.float_to_float8(0.0)
				zreal = AVX.float_to_float8(0.0)
				zimag = AVX.float_to_float8(0.0)	
				for iter in xrange(max_iterations):
				#Calculate magnitude
					zrealsqr = AVX.mul(zreal,zreal)
					zimagsqr = AVX.mul(zimag,zimag)
					mag =AVX.add(zrealsqr,zimagsqr)
					mask = AVX.greater_than(four, mag)
				#Break if mag < 4 (in the case that it returns all 0s then all magnitudes are >= 4 thus their sign will return 0 and we should break)
					sign = AVX.signs(mask)
					if sign == 0:
						break
				#Add 1 to toWrite for those mag < 4
					addToIter = AVX.bitwise_and(one,mask)
					toWrite = AVX.add(toWrite,addToIter)
				#Calculate z= z*z + c
					zrealtmp = AVX.add(AVX.sub(zrealsqr,zimagsqr),creal)
					zimagtmp = AVX.fmadd(two,AVX.mul(zreal,zimag),cimag)
					zreal = zrealtmp
					zimag = zimagtmp
			#Write results back
				AVX.to_mem(toWrite, &(out_counts[i,j]))	
