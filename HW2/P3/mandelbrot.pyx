import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange


cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

# Collaborated with Sami Goche on this part
# to discuss ideas and approaches.

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int max_iterations=511):
    cdef:
       int i, j, iter
       #np.complex64_t c,
       AVX.float8 c_IMAGINARY, c_REAL, z_REAL, z_IMAGINARY, iterations, fours, ones, magnitude_squared, mask, z_REAL_prime, z_IMAGINARY_prime

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
        for i in prange(in_coords.shape[0], num_threads=4, schedule='static', chunksize=1):
            #for j in range(in_coords.shape[1]):
            for j in range(0, in_coords.shape[1], 8):
                #c = in_coords[i, j]
                #z=0
                
                #create now different values for c and z that have a real axis and imaginary axis
                #where both are float8s so we can do instruction-level parallelism. 
                c_REAL = AVX.make_float8(in_coords[i, j+7].real, in_coords[i, j+6].real, in_coords[i, j+5].real, in_coords[i, j+4].real, in_coords[i, j+3].real, in_coords[i, j+2].real, in_coords[i, j+1].real, in_coords[i, j].real)
                c_IMAGINARY = AVX.make_float8(in_coords[i, j+7].imag, in_coords[i, j+6].imag, in_coords[i, j+5].imag, in_coords[i, j+4].imag, in_coords[i, j+3].imag, in_coords[i, j+2].imag, in_coords[i, j+1].imag, in_coords[i, j].imag)
                z_REAL = AVX.float_to_float8(0.0)
                z_IMAGINARY = AVX.float_to_float8(0.0)
                
                #initalize a storage variable for the number of iterations each value we are looking at
                #took to evaluate out. Also create a fours (to compare to the magnitude later) and a ones
                #to do bitwise logic with later. 
                iterations = AVX.float_to_float8(0.0)
                fours = AVX.float_to_float8(4.0)
                ones = AVX.float_to_float8(1.0)

                for iter in range(max_iterations):
                    # if magnitude_squared(z) > 4:
                    #     break
                    # z = z * z + c

                    #calculate a magntiude_squared like we did in the serial code but using the fmadd
                    #function. Now create a mask so we have a bit vector where values are less than four.
                    magnitude_squared = AVX.fmadd(z_REAL, z_REAL, AVX.mul(z_IMAGINARY, z_IMAGINARY))
                    mask = AVX.less_than(magnitude_squared, fours)
                    
                    #Stopping condition is when all are greater than 4. Saw the piazza post about this,
                    #which was helpful. 
                    if AVX.signs(mask) == 0:
                      break
                    
                    #Increment to iterations in the case where we need to (what the AVX.bitwise_and does)
                    iterations = AVX.add(iterations, AVX.bitwise_and(mask, ones))

                    #recalculate the z_real_prime and the z_imaginary prime. 
                    #Had to review: 
                    #http://www.regentsprep.org/regents/math/algtrig/ato6/multlesson.htm
                    z_REAL_prime = AVX.fmsub(z_REAL, z_REAL, AVX.mul(z_IMAGINARY, z_IMAGINARY))
                    z_IMAGINARY_prime = AVX.fmadd(z_IMAGINARY, z_REAL, AVX.mul(z_IMAGINARY, z_REAL))

                    z_REAL = AVX.add(z_REAL_prime, c_REAL)
                    z_IMAGINARY = AVX.add(z_IMAGINARY_prime, c_IMAGINARY)

                #out_counts[i, j] = iter
                #AVX.to_mem(iterations, &(out_counts[i, j]))

                #The function below was referenced in piazza, which is where I saw the recommendation
                #to make something like this. 
                counts_to_output(iterations, out_counts, i, j)

#This signature was from the piazza post on this topic
cdef void counts_to_output(AVX.float8 counts, np.uint32_t [:, :] out_counts, int i, int j) nogil:
  cdef:
    float tempCounts[8]
    int index

  AVX.to_mem(counts, &(tempCounts[0]))

  for index in xrange(8):
    out_counts[i, j + index] = <np.uint32_t>tempCounts[index]




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

#AVX.to_mem(iters, &())