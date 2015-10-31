#cython: boundscheck=False
#cython: wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
from cython.parallel import prange
from libc.stdio cimport printf

# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT
cdef int int_max = 4294967295



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
        FLOAT *XY1
        FLOAT *XY2
        FLOAT *V1
        FLOAT *V2
        int j, dim, X_cent, Y_cent, X_max, X_min, Y_max, Y_min
        int Y_check, X_check
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.

    #find the grid coordinates for the center of the ball
    # considered 
    X_cent = <int>(XY[i, 0] / grid_spacing)
    Y_cent = <int>(XY[i, 1] / grid_spacing)
    # find the max and min grid coordinates to track
    # we check the objects 2 grid squares away
    # therefore 3 grid points away
    X_max =  min(X_cent+3, Grid.shape[0] - 1)
    X_min =  max(X_cent-3, 0)
    # find the max and min grid coordinates to track
    # we check the objects 2 grid squares away
    # therefore 3 grid points away
    Y_max =  min(Y_cent+3, Grid.shape[1] - 1)
    Y_min =  max(Y_cent-3, 0)

    # loop over the neighbors to check
    # if there is any ball
    for X_check in range(X_min, X_max+1):
        for Y_check in range(Y_min, Y_max):
            j = Grid[X_check, Y_check]
            # check only one side (avoid double counting)
            # don't check when there is no ball (j = int_max not -1 like said in the driver)
            if j>i and j != int_max :
                XY2 = &(XY[j, 0])
                V2 = &(V[j, 0])
                if overlapping(XY1, XY2, R):
                    # SUBPROBLEM 4: Add locking
                    # to avoir deadlock, we access them in the same order
                    # here it is smaller first
                    if not moving_apart(XY1, V1, XY2, V2):
                        acquire(&(locks[i]))
                        acquire(&(locks[j]))
                        collide(XY1, V1, XY2, V2)
                        release(&(locks[i]))
                        release(&(locks[j]))
                    # give a slight impulse to help separate them
                    for dim in range(2):
                        V2[dim] += eps * (XY2[dim] - XY1[dim])

cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t,
             int n_threads):
    cdef:
        int count = XY.shape[0]
        int i, j, dim
        FLOAT *XY1, *XY2, *V1, *V2
        int X_cent, Y_cent, update_X_cent, update_Y_cent

        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    cdef int before_xgrid, before_ygrid, after_xgrid, after_ygrid
    # --------------------
    # bounce off of walls
    # --------------------
    #
    # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
    # scheduling.
    for i in prange(count ,nogil = True, schedule = 'static', chunksize =count/100 , num_threads = n_threads):
    # Serial loop kept for testing
    #for i in range(count):
        for dim in range(2):
            if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                V[i, dim] *= -1
    # --------------------
    # bounce off of each other
    # --------------------
    #
    # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
    # scheduling.
    for i in prange(count ,nogil = True, schedule = 'static', chunksize =count/100 , num_threads = n_threads):
    # Serial loop kept for testing
    #for i in range(count):
        sub_update(XY, V, R, i, count, Grid, grid_spacing, locks)
    # --------------------
    # update positions
    # --------------------
    #
    # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
    #    scheduling).
    # SUBPROBLEM 2: update the grid values.
    for i in prange(count ,nogil = True, schedule = 'static', chunksize =count/100 , num_threads = n_threads):
    # Serial loop kept for testing
    #for i in range(count):


        X_cent = <int> (XY[i, 0] / grid_spacing)
        Y_cent = <int> (XY[i, 1] / grid_spacing)

        # center has to stay inside the box
        X_cent =  min(X_cent, Grid.shape[0] - 1)
        Y_cent =  min(Y_cent, Grid.shape[1] - 1)


        # update the location of the objects 
        for dim in range(2):
            XY[i, dim] += V[i, dim] * t

        # update the grid coordinate 
        update_X_cent = <int>(XY[i, 0] / grid_spacing)
        update_Y_cent = <int>(XY[i, 1] / grid_spacing)

        # again the center has to stay inside the box
        update_X_cent =  min(update_X_cent, Grid.shape[0] - 1)
        update_Y_cent =  min(update_Y_cent, Grid.shape[1] - 1)

        #if the particule changed location
        if (X_cent != update_X_cent) or (Y_cent != update_Y_cent):

            # former location becomes empty i.e., no object
            # so the value of the grid is int_max
            Grid[X_cent, Y_cent] = int_max

            # replace the former value of the grid with the new objects
            Grid[update_X_cent, update_Y_cent] = i


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
