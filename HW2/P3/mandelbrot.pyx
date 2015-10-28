import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange

cimport openmp

cdef np.float64_t magnitude_squared(np.complex64_t z) nogil:
    return z.real * z.real + z.imag * z.imag

# FUNCTION COMPUTING THE MAGNITUDE OF A COMPLEX FLOAT 8 Z, THIS
# RETUNS ANOTHER FLOAT8 VARIABLE AND OPARATES IN PARALLEL WITHOUT GIL
cdef AVX.float8 z_mag(AVX.float8 z_r, AVX.float8 z_i) nogil:
    cdef AVX.float8 z_r_mag = AVX.mul(z_r,z_r)
    cdef AVX.float8 z_i_mag = AVX.mul(z_i,z_i)
    cdef AVX.float8 z_mag_squ = AVX.add(z_r_mag, z_i_mag)
    return AVX.sqrt(z_mag_squ)

@cython.boundscheck(False)
@cython.wraparound(False)

cpdef mandelbrot_avx(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int nthreads,
                 int max_iterations=511):
    cdef:
        int i, j, iter
        np.complex64_t c, z

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"



    cdef AVX.float8 c_real, c_imag

    cdef int j_s, ii

    cdef float[:,:] r_coords = np.real(in_coords)
    cdef float[:,:] i_coords = np.imag(in_coords)
    
    cdef AVX.float8 z_r
    cdef AVX.float8 z_i

    cdef AVX.float8 z_r_mag
  
    cdef AVX.float8 z_mag_squ 
    
    cdef AVX.float8 z_magnitude

    cdef AVX.float8 z_2
    z_2=AVX.float_to_float8(2.0)

    cdef AVX.float8 not_go_mask

    cdef AVX.float8 z_r_temp1, z_r_temp2, z_r_temp3, z_i_temp1

    cdef AVX.float8 mask

    cdef AVX.float8 iter_float8

    cdef AVX.float8 z_4,z_1
    z_4=AVX.float_to_float8(4.0)
    z_1=AVX.float_to_float8(1)

    cdef AVX.float8 iter_0=AVX.float_to_float8(0)

    print "size of r_coords",r_coords.size
    print "size of i_coords",i_coords.size

    cdef int avx_range=in_coords.shape[1]/8 #assuming that the shape is divisible by 8 AVX computations
    
    for i in prange(in_coords.shape[0],nogil=True ,num_threads=nthreads, schedule='static', chunksize=1):
        for j in range(avx_range):
            
            j_s=8*j
            

            c_real=AVX.make_float8(r_coords[i,j_s],r_coords[i,j_s+1],r_coords[i,j_s+2],r_coords[i,j_s+3],r_coords[i,j_s+4],r_coords[i,j_s+5],r_coords[i,j_s+6],r_coords[i,j_s+7])
            c_imag=AVX.make_float8(i_coords[i,j_s],i_coords[i,j_s+1],i_coords[i,j_s+2],i_coords[i,j_s+3],i_coords[i,j_s+4],i_coords[i,j_s+5],i_coords[i,j_s+6],i_coords[i,j_s+7])
            
            z_r=AVX.float_to_float8(0.0)
            z_i=AVX.float_to_float8(0.0)

            iter_float8=iter_0

            for ii in range(max_iterations):
                z_r_temp1=AVX.mul(z_r,z_r)
                z_r_temp2=AVX.mul(z_i,z_i)
                z_r=AVX.sub(z_r_temp1,z_r_temp2)
                z_r=AVX.add(z_r,c_real)

                z_i_temp1=AVX.mul(z_r,z_i)
                z_i=AVX.fmadd(z_i_temp1, z_2, c_imag)

                z_r_mag = AVX.mul(z_r,z_r)
                z_mag_squ = AVX.fmadd(z_i, z_i, z_r_mag)

                z_magnitude=AVX.sqrt(z_mag_squ)

                not_go_mask=AVX.less_than(z_4,z_magnitude)
                if AVX.signs(not_go_mask) ==255:
                    break

                mask = AVX.mul(AVX.less_than(z_magnitude, z_4),z_1)
                iter_float8 = AVX.add(iter_float8,mask)

            #AVX.to_mem(iter_float8, &(out_counts[i,j_s]))
                


cpdef mandelbrot_thread(np.complex64_t [:, :] in_coords,
                 np.uint32_t [:, :] out_counts,
                 int nthreads,
                 int max_iterations=511):
    cdef:
        int i, j, iter
        np.complex64_t c, z

    assert in_coords.shape[1] % 8 == 0, "Input array must have 8N columns"
    assert in_coords.shape[0] == out_counts.shape[0], "Input and output arrays must be the same size"
    assert in_coords.shape[1] == out_counts.shape[1],  "Input and output arrays must be the same size"


    for i in prange(in_coords.shape[0],nogil=True ,num_threads=nthreads, schedule='static', chunksize=1):
        for j in range(in_coords.shape[1]):
            c = in_coords[i, j]
            z = 0
            for iter in range(max_iterations):
                if magnitude_squared(z) > 4:
                    break
                z = z * z + c
            out_counts[i, j] = iter


cpdef mandelbrot_nothread(np.complex64_t [:, :] in_coords,
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
