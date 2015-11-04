#cython: boundscheck=False, wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
from cython.parallel import prange
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release



# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT

#define un signed int to act as -1
cdef UINT global_negative = 4294967295

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
                     int width, omp_lock_t *locks) nogil: #added the locking array 
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, dim, grid_x, grid_y, g, h
        float eps = 1e-5



    # SUBPROBLEM 4: Add locking

    #initiate lock for analyised point
    
    acquire(&(locks[i]))

    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])

    #type cast to translate from float position to grid position
    #for both the x and y coordinates
    # this is the current anlaysed point, i, positions
    grid_x = <int> (XY1[0] / grid_spacing)
    grid_y = <int> (XY1[1] / grid_spacing)


    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.

    #Loop through the other positions, using a window and a
    #clamped argument, keeping you in bounds throughout the analysis
    for g in range(max(grid_x - 3, 0), min(width, grid_x + 4)):
        for h in range(max(grid_y - 3, 0), min(width, grid_y + 4)):

            #type cast the analyzed point
            j = <int> Grid[g, h]

            #if the grid point is not empty and associated with z-ordering
            if j != global_negative and j > i:

                XY2 = &(XY[j, 0])
                V2 = &(V[j, 0])

                if overlapping(XY1, XY2, R):

                    # SUBPROBLEM 4: Add locking

                    #Acquire loc for element in XY2 with ball
                    acquire(&(locks[j]))


                    if not moving_apart(XY1, V1, XY2, V2):
                        collide(XY1, V1, XY2, V2)

                    # give a slight impulse to help separate them
                    for dim in range(2):
                        V2[dim] += eps * (XY2[dim] - XY1[dim])

                    #Release loc for element in XY2 with ball
                    release(&(locks[j]))

    #release lock for analyzed point
    release(&(locks[i]))



cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t, 
             int width): # add the grid width

    cdef:
        int count = XY.shape[0]
        int i, j, dim
        FLOAT *XY1, *XY2, *V1, *V2
        int grid_x, grid_y #initlalize these values for update points



        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr


    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    with nogil:
        # bounce off of walls
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.

        for i in prange(count, schedule='static', chunksize = count / 4, num_threads=4):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.

        for i in prange(count, schedule='static', chunksize = count / 4, num_threads=4):
            sub_update(XY, V, R, i, count, Grid, grid_spacing, width, locks)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        
        Grid[:, :] = -1

        for i in prange(count, schedule='static', chunksize = count / 4, num_threads=4):
            
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
                if XY[i, dim] < 0:
                    XY[i, dim] = 0
                if XY[i, 0] > 1:
                    XY[i, dim] = 1

            grid_x = <int> (XY[i, 0] / grid_spacing)
            grid_y = <int> (XY[i, 1] / grid_spacing)


            Grid[grid_x, grid_y] = i


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks



