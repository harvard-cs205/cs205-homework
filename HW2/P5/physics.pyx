#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
cimport omp_defs
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
from cython.parallel import prange

# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT
ctypedef np.int32_t INT

cdef inline int overlapping(FLOAT *x1,
                            FLOAT *x2,
                            float R) nogil:
    cdef:
        float dx = x1[0] - x2[0]
        float dy = x1[1] - x2[1]
    return (dx * dx + dy * dy) < (4 * R * R)



cdef inline int moving_apart(FLOAT *x1, FLOAT *v1,
                             FLOAT *x2, FLOAT *v2) nogil:
    cdef:
        float deltax = x2[0] - x1[0]
        float deltay = x2[1] - x1[1]
        float vrelx = v2[0] - v1[0]
        float vrely = v2[1] - v1[1]
    # check if delta and velocity in same direction
    return (deltax * vrelx + deltay * vrely) > 0.0


cdef inline void collide(FLOAT *x1, FLOAT *v1,
                         FLOAT *x2, FLOAT *v2) nogil:
    cdef:
        float x1_minus_x2[2]
        float v1_minus_v2[2]
        float change_v1[2]
        float len_x1_m_x2, dot_v_x
        int dim
    # https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
    for dim in range(2):
        x1_minus_x2[dim] = x1[dim] - x2[dim]
        v1_minus_v2[dim] = v1[dim] - v2[dim]
    len_x1_m_x2 = x1_minus_x2[0] * x1_minus_x2[0] + x1_minus_x2[1] * x1_minus_x2[1]
    dot_v_x = v1_minus_v2[0] * x1_minus_x2[0] + v1_minus_v2[1] * x1_minus_x2[1]
    for dim in range(2):
        change_v1[dim] = (dot_v_x / len_x1_m_x2) * x1_minus_x2[dim]
    for dim in range(2):
        v1[dim] -= change_v1[dim]
        v2[dim] += change_v1[dim]  # conservation of momentum

cdef void sub_update(FLOAT[:, ::1] XY,
                     FLOAT[:, ::1] V,
                     float R,
                     int counter,
                     UINT[:, ::1] Grid,
                     int grid_size, INT[:,:] overlapping,omp_lock_t *locks) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int i, j, dim,c,l,f,ii,jj
        float eps = 1e-5
        INT currentBallInGrid,tmpBallIndex
    # SUBPROBLEM 4: Add locking

    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.

    for l in prange(grid_size,num_threads=4):
        for j in xrange(grid_size):
            currentBallInGrid = Grid[l,j]
            if currentBallInGrid != -1:
                acquire(&locks[currentBallInGrid]) 
                XY1 = &(XY[currentBallInGrid, 0])
                V1 = &(V[currentBallInGrid, 0])
            else: 
                continue

#            Check for overlap among balls 2 grid spaces away. Make sure not to check a pair twice
#            Method for checking overlap, see simple figure below. 
#            Current ball is the plus sign
#            Potential overlap at positions marked x
#            Do not check positions marked with percent sign
#               |% % + x x   
#               |x x x x x   
#               |x x x x x   

            for ii in range(l,l+3):
                for jj in range(j-2,j+3):
                    if (ii == l and jj <= j) or \
                        ii < 0 or ii >= grid_size or jj < 0 or jj >= grid_size: continue
                    tmpBallIndex = Grid[ii,jj]
                    if tmpBallIndex != -1:
                        acquire(&locks[tmpBallIndex])        
                        XY2 = &(XY[tmpBallIndex,0])
                        V2 = &(V[tmpBallIndex,0])

                        if not moving_apart(XY1, V1, XY2, V2):
                            collide(XY1, V1, XY2, V2)   

                        #give a slight impulse to help separate them
                        for dim in range(2):
                            V2[dim] += eps * (XY2[dim] - XY1[dim])  
                        release(&locks[tmpBallIndex])
            if currentBallInGrid != -1:
                release(&locks[currentBallInGrid])



cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             int grid_size, 
             uintptr_t locks_ptr,
             float t, float grid_spacing, INT[:,:] toCheck,int nt):
    cdef:
        int count = XY.shape[0]
        int i, j, dim, roundedPositionX, roundedPositionY, oX,oY
        FLOAT *XY1, *XY2, *V1, *V2
        unsigned int grid_x,grid_y
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    #assert XY.shape[0] == V.shape[0]
    #assert XY.shape[1] == V.shape[1] == 2

    with nogil:
        # bounce off of walls
        for i in prange(count, schedule='static', chunksize=count/4,num_threads=nt):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        # I stuffed both loops into sub_update so only call it once
        sub_update(XY, V, R, count, Grid, grid_size,toCheck,locks)

        # SUBPROBLEM 2: update the grid values.
        for i in prange(count,schedule='static',chunksize=count/4,num_threads=nt):
            roundedPositionX = <unsigned int>(XY[i,0]/grid_spacing)
            roundedPositionY = <unsigned int>(XY[i,1]/grid_spacing)
            if roundedPositionX >= 0 and roundedPositionY >= 0 and roundedPositionX < grid_size and roundedPositionY < grid_size:
                Grid[roundedPositionX,roundedPositionY] = -1

            for dim in range(2):
                XY[i, dim] += V[i, dim] * t

            roundedPositionX = <int>(XY[i,0]/grid_spacing)
            roundedPositionY = <int>(XY[i,1]/grid_spacing)
            if roundedPositionX >= 0 and roundedPositionY >= 0 and roundedPositionX < grid_size and roundedPositionY < grid_size:
                Grid[roundedPositionX,roundedPositionY] = i
           

def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
