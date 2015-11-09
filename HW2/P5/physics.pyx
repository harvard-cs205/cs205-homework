#cython: boundscheck=False, wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from cython.parallel import prange
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release

# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT
cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b
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
                     int i, int count,
                     UINT[:, ::1] Grid,
                     float grid_spacing,
                     omp_lock_t *locks) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2,
        int dim, grid_size, grid_index1, grid_index2, k, l,\
                surr_box
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    '''
    #Initial Implementation
    #Slower because of excess variables
    grid_size = <unsigned int>((1.0 / grid_spacing) + 1)
    grid_index1 = <unsigned int>(XY1[0]*grid_size)
    grid_index2 = <unsigned int>(XY1[1]*grid_size)
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################+
    ################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    surr_box = 2
    Grid[grid_index1,grid_index2] = <UINT>i
    top = grid_index1 - surr_box - 1
    if top < 0:
        top = 0
    bottom = grid_index1 + surr_box
    if bottom > grid_size-1:
        bottom = grid_size-1
    left = grid_index2 - surr_box - 1
    if left < 0:
        left = 0
    right = grid_index2 + surr_box
    if right > grid_size-1:
        right = grid_size-1
    #release(&(locks[i]))
    for k in range(top,bottom):
        for l in range(left,right):
            if k < grid_size and l < grid_size and Grid[k,l] < count:
                if Grid[k,l] !=  -1 and Grid[k,l] != i:
                    #acquire(&(locks[Grid[k,l]]))
                    XY2 = &(XY[Grid[k,l],0])
                    V2 = &(V[Grid[k,l],0])
                    if overlapping(XY1, XY2, R):
                        if not moving_apart(XY1, V1, XY2, V2):
                            collide(XY1, V1, XY2, V2)
                        for dim in range(2):
                            V2[dim] += eps * (XY2[dim] - XY1[dim])
                    #release(&(locks[Grid[k,l]]))
    
    
    '''
    grid_size = <unsigned int>((1.0 / grid_spacing) + 1)
    grid_index1 = <unsigned int>(XY1[0]*grid_size)
    grid_index2 = <unsigned int>(XY1[1]*grid_size)
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################+
    ################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    #Width of the box to check for other balls in
    surr_box = 2
    for k in range(grid_index1-surr_box-1,grid_index1+surr_box):
        for l in range(grid_index2-surr_box-1,grid_index2+surr_box):
            if k < grid_size and l < grid_size and Grid[k,l] < count\
                    and k>=0 and l>=0:
                #Checking if the grid element is empty and 
                #its not the same grid element
                if Grid[k,l] !=  -1 and Grid[k,l] != i:
                    
                    #Acquiring lock i
                    
                    XY2 = &(XY[Grid[k,l],0])
                    V2 = &(V[Grid[k,l],0])
                    if overlapping(XY1, XY2, R):
                        if not moving_apart(XY1, V1, XY2, V2):
                            if i < Grid[k,l]:
                                acquire(&(locks[i]))
                                acquire(&(locks[Grid[k,l]]))
                            else:
                                acquire(&(locks[Grid[k,l]]))
                                acquire(&(locks[i]))
                            collide(XY1, V1, XY2, V2)
                            if i > Grid[k,l]:
                                release(&(locks[Grid[k,l]]))
                                release(&(locks[i]))
                            else:
                                release(&(locks[i]))
                                release(&(locks[Grid[k,l]]))
                        for dim in range(2):
                            V2[dim] += eps * (XY2[dim] - XY1[dim])
                    




    '''
    #Original Code
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
    '''
    
    
cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t):
    cdef:
        int count = XY.shape[0]
        int i, j, dim, grid_index1, grid_index2
        FLOAT *XY1, *XY2, *V1, *V2
        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    with nogil:
        # bounce off of walls
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, schedule='static',chunksize=count/4,num_threads=4):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):

                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, schedule='static',chunksize=count/4,num_threads=4):
            sub_update(XY, V, R, i, count, Grid, grid_spacing,locks)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        for i in prange(count, schedule='static',chunksize=count/4,num_threads=4):
            '''
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
            '''
            #Checking bounds
            if XY[i,0]>=0 and XY[i,0]<=1 and XY[i,1]>=0 and XY[i,1]<=1:
                grid_index1 = <unsigned int>(XY[i,0]/grid_spacing)
                grid_index2 = <unsigned int>(XY[i,1]/grid_spacing)
                #Resetting the grid value
                Grid[grid_index1, grid_index2] = -1
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
            #Checking bounds
            if XY[i,0]>=0 and XY[i,0]<=1 and XY[i,1]>=0 and XY[i,1]<=1:
                grid_index1 = <unsigned int>(XY[i,0]/grid_spacing)
                grid_index2 = <unsigned int>(XY[i,1]/grid_spacing)
                #Assigning the grid value
                Grid[grid_index1, grid_index2] = i
            

#for i in prange(count, schedule='static', chunksize=count/4,num_threads=4):

def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
