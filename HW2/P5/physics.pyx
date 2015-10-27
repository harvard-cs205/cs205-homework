#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
#from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
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
                     INT[:, ::1] Grid,
                     int grid_size, INT[:,:] overlapping) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int i, j, dim, gridx, gridy
        float eps = 1e-5
        INT currentBallInGrid,tmpBallIndex
        int overlapping2[12][2]
        #np.int32_t [:,:] overlapping #At most 12 balls to check for overlap for each ball

    
        
    # SUBPROBLEM 4: Add locking

    #XY1 = &(XY[i, 0])
    #V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    
    for i in range(grid_size):
        for j in range(grid_size):
            currentBallInGrid = Grid[i,j]
            
            if currentBallInGrid != -1:#True if contains index of ball
                XY1 = &(XY[currentBallInGrid, 0])
                V1 = &(V[currentBallInGrid, 0])
            else: 
                continue
            
#            Suppose plus sign(+) is current index into grid
#            Then we look at positions marked x, rest are marked %
#               |% % + x x   
#               |x x x x x   
#               |x x x x x   
            overlapping2[0][0] = i
            overlapping2[0][1] = j+1
            overlapping2[1][0] = i
            overlapping2[1][1] = j+2
            overlapping2[2][0] = i+1
            overlapping2[2][1] = j-2
            overlapping2[3][0] = i+1
            overlapping2[3][1] = j-1
            overlapping2[4][0] = i+1
            overlapping2[4][1] = j
            overlapping2[5][0] = i+1
            overlapping2[5][1] = j+1
            overlapping2[6][0] = i+1
            overlapping2[6][1] = j+2
            overlapping2[7][0] = i+2
            overlapping2[7][1] = j-2
            overlapping2[8][0] = i+2
            overlapping2[8][1] = j-1
            overlapping2[9][0] = i+2
            overlapping2[9][1] = j
            overlapping2[10][0] = i+2
            overlapping2[10][1] = j+1
            overlapping2[11][0] = i+2
            overlapping2[11][1] = j+2
            
#            with gil:
#                try:
#                    overlapping[0:2] = np.array([[i,j+2]]) 
#                    overlapping[2:7] = np.array([[i+1,j-2],[i+1,j-1],[i+1,j],[i+1,j+1],[i+1,j+2]])
#                    overlapping[7:12] = np.array([[i+2,j-2],[i+2,j-1],[i+2,j],[i+2,j+1],[i+2,j+2]])
#                except:
#                    pass



            #for i in range(overlapping.shape[0]):
            for i in range(12):
                gridx = overlapping[i,0]
                gridy = overlapping[i,1]

                #Bounds checking
                if (gridx >= 0 and gridx < grid_size and gridy >= 0 and gridy < grid_size):
                    tmpBallIndex = Grid[gridx,gridy]

                    if tmpBallIndex != -1:
                        XY2 = &(XY[tmpBallIndex,0])
                        V2 = &(V[tmpBallIndex,0])

                        if not moving_apart(XY1, V1, XY2, V2):
                            collide(XY1, V1, XY2, V2)
                        #give a slight impulse to help separate them
                        for dim in range(2):
                            V2[dim] += eps * (XY2[dim] - XY1[dim])    
                       



cdef void sub_updateOld(FLOAT[:, ::1] XY,
                     FLOAT[:, ::1] V,
                     float R,
                     int i, int count,
                     INT[:, ::1] Grid,
                     int grid_spacing) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, dim
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    for j in range(i + 1, count):
        XY2 = &(XY[j, 0])
        V2 = &(V[j, 0])
        if overlapping(XY1, XY2, R):
            # SUBPROBLEM 4: Add locking
            if not moving_apart(XY1, V1, XY2, V2):
                collide(XY1, V1, XY2, V2)

            # give a slight impulse to help separate them
            for dim in range(2):
                V2[dim] += eps * (XY2[dim] - XY1[dim])





cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             INT[:, ::1] Grid,
             float R,
             int grid_size, 
             uintptr_t locks_ptr,
             float t, float grid_spacing, INT[:,:] toCheck,
             INT[:] originalXPos,INT[:] originalYPos,int nt):
    cdef:
        int count = XY.shape[0]
        int i, j, dim, roundedPositionX, roundedPositionY
        FLOAT *XY1, *XY2, *V1, *V2
        # SUBPROBLEM 4: uncomment this code.
        # omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    #assert XY.shape[0] == V.shape[0]
    #assert XY.shape[1] == V.shape[1] == 2

    with nogil:
        # bounce off of walls
        for i in prange(count, schedule='static', chunksize=count/4,num_threads=nt):
        #for i in range(count):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1
        # bounce off of each other
        sub_update(XY, V, R, count, Grid, grid_size,toCheck)

        # update positions
        for i in prange(count,schedule='static',chunksize=count/4,num_threads=nt):
        #for i in range(count):
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t

        # SUBPROBLEM 2: update the grid values.
        #for i in range(count):
        for i in prange(count,schedule='static',chunksize=count/4,num_threads=nt):
            Grid[originalXPos[i],originalYPos[i]] = -1
        #for i in range(count):
        for i in prange(count,schedule='static',chunksize=count/4,num_threads=nt):
            roundedPositionX = <int>(XY[i,0]/grid_spacing)
            roundedPositionY = <int>(XY[i,1]/grid_spacing)
            if roundedPositionX >= 0 and roundedPositionY >= 0 and roundedPositionX < grid_size and roundedPositionY < grid_size:
                Grid[roundedPositionX,roundedPositionY] = i
           

#def preallocate_locks(num_locks):
#    cdef omp_lock_t *locks = get_N_locks(num_locks)
#    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
#    return <uintptr_t> <void *> locks