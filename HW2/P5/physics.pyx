#cython: boundscheck=False, wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
from cython.parallel import prange
# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT

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
                     uintptr_t locks_ptr) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, dim, current_grid_x, current_grid_y, x, y, grid_x, grid_y, idx, j_idx, grid_size
        int objects_within_2sqs[25] #This will contain at most 25 objects
        float eps = 1e-5
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    # SUBPROBLEM 4: Add locking
    acquire(&(locks[i]))
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    
    #idea: change this for loop to iterate over all grid squares within 2 squares of XY1
    # then check all j from those squares
    grid_size = Grid.shape[0]
    current_grid_x = <int>(XY1[0]/grid_spacing)
    current_grid_y = <int>(XY1[1]/grid_spacing)
    #make sure this object is in bounds
    if ((current_grid_x >= 0) and (current_grid_x < grid_size) and (current_grid_y >= 0) and (current_grid_y < grid_size)): 
        #get indexes of objects within 2 squares of XY1's grid square
        for x in range(-2,3):
            for y in range(-2,3):
                idx = (x+2) * 5 + (y+2) # index in objects_within_2sqs
                objects_within_2sqs[idx] = -1
                grid_x = current_grid_x + x
                grid_y = current_grid_y + y
                #make sure this square is in the grid
                if ((grid_x >= 0) and (grid_x < grid_size) and (grid_y >= 0) and (grid_y < grid_size)):
                    #j is the index of the object in Grid[grid_x,grid_y]
                    j = Grid[grid_x,grid_y]
                    #make sure the object is not the same as object as i, and only store objects that have index greater than j, that way we don't account for collision twice
                    if (j > i):
                        objects_within_2sqs[idx] = j
        for j_idx in range(25):# we set object_within_2sqs to have length 25 
            j = objects_within_2sqs[j_idx]
            if (j != -1):
            #for j in range(i + 1, count):
                XY2 = &(XY[j, 0])
                V2 = &(V[j, 0])
                if overlapping(XY1, XY2, R):
                    # SUBPROBLEM 4: Add locking
                    acquire(&(locks[j]))
                    if not moving_apart(XY1, V1, XY2, V2):
                        collide(XY1, V1, XY2, V2)
                    # give a slight impulse to help separate them
                    for dim in range(2):
                        V2[dim] += eps * (XY2[dim] - XY1[dim])
                    release(&(locks[j]))
    release(&(locks[i]))
cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t):
    cdef:
        int count = XY.shape[0]
        int i, j, dim, num_threads, new_grid_x, new_grid_y, idx_x, idx_y
        FLOAT *XY1, *XY2, *V1, *V2
        int grid_size = Grid.shape[0]
        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr
    num_threads = 4
    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    with nogil:
        # bounce off of walls
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count,num_threads=num_threads,schedule='static', chunksize=count/4):
        #for i in range(count):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count,num_threads=num_threads,schedule='static', chunksize=count/4):
        #for i in range(count):
            sub_update(XY, V, R, i, count, Grid, grid_spacing, locks_ptr)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        
        #reset all values in grid
        for idx_x in prange(grid_size,num_threads=num_threads,schedule='static'):
            for idx_y in range(grid_size):
                Grid[idx_x,idx_y] = -1
        for i in prange(count,num_threads=num_threads,schedule='static', chunksize=count/4):
        #for i in range(count):
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
            #update grid to reflect object i's new location
            new_grid_x = <int>(XY[i,0]/grid_spacing)
            new_grid_y = <int>(XY[i,1]/grid_spacing)
            if ((new_grid_x >= 0) and (new_grid_x < grid_size) and (new_grid_y >= 0) and (new_grid_y < grid_size)):
                Grid[new_grid_x, new_grid_y] = i #THIS COULD CAUSE ISSUES WITHOUT LOCKS WHEN RUN WITH MULTIPLE THREADS
            


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
