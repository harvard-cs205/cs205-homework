import numpy as np
cimport numpy as np
cimport cython
import numpy
cimport AVX
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf, stdout, fprintf

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

# FUNCTION DEFINED TO SERVE AS HELPER FUNCTION IN PRINTING AVX FLOATS.
# THIS FUNCTION WAS MAINLY USED FOR DEBUGGING

# cdef void print_float8(AVX.float8 f8) nogil:
#     cdef float *iter_view = <float *> malloc(8*sizeof(float))

#     AVX.to_mem(f8, iter_view)
#     cdef int i
#     for i in range(8):
#         printf('%f \n', iter_view[i])
#     printf('Done with float8')
#     printf('\n')
#     free(iter_view)

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


    # DEFINITIONS OF VARIABLES USED, SOME OF THEM MAY BE RETIRED AND NOT USED IN THE CODE 
    # BELOW, ALSO DEFINITIONS ARE MOSTLY DONE OUTSIDE OF THE LOOP IN ORDER TO AVOID OVERHEAD
    cdef AVX.float8 c_real, c_imag

    cdef int j_s, ii,k

    cdef float[:,:] r_coords = np.real(in_coords)
    cdef float[:,:] i_coords = np.imag(in_coords)
    
    cdef AVX.float8 z_r
    cdef AVX.float8 z_i

    cdef AVX.float8 z_r_mag
  
    cdef AVX.float8 z_mag_squ 
    
    cdef AVX.float8 z_magnitude

    cdef AVX.float8 not_go_mask

    cdef AVX.float8 z_r_temp1, z_r_temp2, z_r_temp3, z_i_temp1,z_r_temp

    cdef AVX.float8 mask

    cdef AVX.float8 iter_float8

    cdef AVX.float8 mask_t

    # DEFINING FUNCTIONS IN AVX FLOATS TO BE SET INTO THE LATER EQUATIONS AS INITIALIZATIONS
    cdef AVX.float8 z_4,z_1,z_0,z_2
    z_4=AVX.float_to_float8(4.0)
    z_1=AVX.float_to_float8(1)
    z_0=AVX.float_to_float8(0.0)
    z_2=AVX.float_to_float8(2.0)
    
    
    for i in prange(in_coords.shape[0],nogil=True ,num_threads=nthreads, schedule='static', chunksize=1):
        for j in range(0,in_coords.shape[1],8):
            # ADAPTATION FOR PREVIOUS IMPLEMENTATION (PLEASE IGNORE THIS LINE)
            j_s=j

            # CREATING THE REAL COMPONENT OF C BASED ON THE REAL PART COORDINATES IN
            # AS A FLOAT 8 WITH THE DIFFERENT VALUES IN PARALLEL
            c_real=AVX.make_float8(r_coords[i,j_s+7],
                                        r_coords[i,j_s+6],
                                        r_coords[i,j_s+5],
                                        r_coords[i,j_s+4],
                                        r_coords[i,j_s+3],
                                        r_coords[i,j_s+2],
                                        r_coords[i,j_s+1],
                                        r_coords[i,j_s])

            # CREATING THE IMAGINARY COMPONENT OF C BASED ON THE REAL PART COORDINATES IN
            # AS A FLOAT 8 WITH THE DIFFERENT VALUES IN PARALLEL
            c_imag=AVX.make_float8(i_coords[i,j_s+7],
                                        i_coords[i,j_s+6],
                                        i_coords[i,j_s+5],
                                        i_coords[i,j_s+4],
                                        i_coords[i,j_s+3],
                                        i_coords[i,j_s+2],
                                        i_coords[i,j_s+1],
                                        i_coords[i,j_s])

            # DEFINING THE INTIAL VALUES OF Z REAL COMPLEX AND ITERATIONS AS 0
            z_r=z_0
            z_i=z_0
            iter_float8=z_0

            for ii in range(max_iterations):
                # CREATING THE MASK COMPARING THE MAGNITUDE OF Z TO 2^2 WHICH IS 4. 
                mask_t=AVX.less_than(AVX.fmadd(z_i, z_i, AVX.mul(z_r,z_r)), z_4)

                # THIS COMPARE USES THE ABOVE TO KNOW IF WE SHOULE EXIT THE CODE 
                # IF THIS IS TRUE WE WILL EXIT THE CODE, AND IT WILL ONLY BE TRUE IF 
                # ALL OF THE VALUES IN THE NOT_GO_MASK IS FULL
                if not AVX.signs(mask_t):
                    break

                # CREATING FLOAT 8 FOR THE REAL AND UPDATIND THE VALUES
                # WITH THE ADDED C THROUGH THE EQUATION
                # Z_R= RE(Z*Z+C)=Z_R^2-Z_I^2+C_R
                z_r_temp=z_r
                z_r=AVX.add(AVX.sub(AVX.mul(z_r,z_r),AVX.mul(z_i,z_i)),c_real)

                # CREATING FLOAT 8 FOR THE IMAGINARY PART OF C FOR THE 
                # EQUATION DESCRIBED ABOVE BUT APPLIED TO IMAGINARY SUCH THAT
                # Z_I=Z_R*Z_I*2
                z_i=AVX.fmadd(AVX.mul(z_r_temp,z_i), z_2, c_imag)

                # ONCE WE OBTAIN THE VALUE ABOVE WE CAN ADD THE MASK TO THE ITERATION
                # AS THE MASK WILL EITHER BE 1 OR 0 DEPENDING ON THE INDEQUALITY 
                # BOOLEAN RESULT FROM THE PART ABOVE
                iter_float8 = AVX.add(iter_float8,AVX.bitwise_and(mask_t, z_1))
            
            # THIS PORTION TRANSFERS ALL OF THE DATA STORED IN AVX INTO THE OUT_COUNTS ARRAYS
            # TO BE EXPORTED TO THE OUTSIDE FUNCTION PERFORMING THE COMPUTATION
            for k in range(8):
                out_counts[i,j+k]= <np.uint32_t> ((<np.float32_t*> &iter_float8)[k])
            


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
