#cython: boundscheck=False
#cython: wraparound=False

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
                     float dx) nogil:
    cdef:
        FLOAT *XY1
        FLOAT *XY2
        FLOAT *V1
        FLOAT *V2
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
    cdef:
        UINT[:, :] neighbor_ball_ids    
        UINT X_cent, Y_cent , X_left, X_right, Y_bot, Y_top


    # Check the balls adjacent to the considered ball i
    # Recall that one ball is in 4 grids (or grid nodes)
    # if the center of the ball is at (X_cent, Y_cent)
    # where X_cent = x_cent / dx with x_cent the x-coordiantes of the center
    # and the grid spacing is dx
    # then the ball is inside the square with bottom left corner
    # (X_cent - 1, Y_cent - 1) and top right corner (X_cent + 1, Y_cent + 1)
    # Therefore we need to check if there are balls from the left bottom corner
    # (X_cent - 2 , Y_cent - 2 ) and top right corner  (X_cent + 2,  Y_cent + 2)


    # First find the center of the ball on the grid
    X_cent = <int>  (XY[i, 0]/dx)
    Y_cent = <int>  (XY[i, 1]/ dx)

    # Second find X_right, X_left, Y_bot, Y_top
    # Write here the exeptions: like the ball we are checking the neigbors
    # from is on one of the 4 sides of the walls


    # if X_cent >= Grid.shape[1] - 1:
    #     X_right = 1
    # else:
    #     X_right = 2
    # if X_cent <= 1:
    #     X_left = 1
    # else:
    #     X_left = 2
    # if Y_cent >= Grid.shape[0] - 1:
    #     Y_bot = 1
    # else:
    #     Y_bot = 2
    # if Y_cent <= 1:
    #     Y_top = 1
    # else:
    #     X_left = 2 

    # for testing 
    X_left = 3
    X_right = 3 
    Y_bot = 3
    Y_top = 3  

    # list of grid indexes to check

    neighbor_ball_ids = Grid[X_cent - X_left : X_cent + X_right, Y_cent - Y_top : Y_cent + Y_bot]

    # from the grid indexes check if there is any ball at these locations
    # and get their number                                                    
    cdef int row_balls, col_balls
    for row_balls in range(neighbor_ball_ids.shape[0]):
        for col_balls in range(neighbor_ball_ids.shape[1]):
            j = neighbor_ball_ids[row_balls, col_balls] 
            # the goal of the previous 3 lines is to do
            # for j in neighbor_ball_ids
            # but cannot do this without gil !    
    #for j in range(i + 1, count):

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
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t,
             int n_threads):
    cdef:
        int count = XY.shape[0]
        int i, j, dim
        #FLOAT *XY1, *XY2, *V1, *V2
        # SUBPROBLEM 4: uncomment this code.
        # omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

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
        sub_update(XY, V, R, i, count, Grid, grid_spacing)
    # --------------------
    # update positions
    # --------------------
    #
    # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
    #    scheduling).
    # SUBPROBLEM 2: update the grid values.
    for i in prange(count, num_threads=num_threads, schedule='static', chunksize=chunksize):
    before_xgrid = <int>(XY[i, 0]/grid_spacing)
    before_ygrid = <int>(XY[i, 1]/grid_spac
    # Make sure the before values are not out of bounds...
    if before_xgrid >= Grid.shape[0]:
        before_xgrid = Grid.shape[0] - 1
    if before_xgrid < 0:
        before_xgrid = 0
    if before_ygrid >= Grid.shape[1]:
        before_ygrid = Grid.shape[1] - 1
    if before_ygrid < 0:
        before_ygrid
    for dim in range(2):
        XY[i, dim] += V[i, dim] * t
    # Based on the new position, update the grid...
    after_xgrid = <int>(XY[i, 0]/grid_spacing)
    after_ygrid = <int>(XY[i, 1]/grid_spacing)
    # Need to make sure you are not out of bounds
    if after_xgrid >= Grid.shape[0]:
        after_xgrid = Grid.shape[0] - 1
    if after_xgrid < 0:
        after_xgrid = 0
    if after_ygrid >= Grid.shape[1]:
        after_ygrid = Grid.shape[1] - 1
    if after_ygrid < 0:
        after_ygrid
    if (before_xgrid != after_xgrid) or (before_ygrid != after_ygrid):
        Grid[before_xgrid, before_ygrid] = UINT32_MAX
        Grid[after_xgrid, after_ygrid] = i
    # for i in prange(count ,nogil = True, schedule = 'static', chunksize =count/100 , num_threads = n_threads):
    # # Serial loop kept for testing
    # #for i in range(count):
    #     for dim in range(2):
    #         XY[i, dim] += V[i, dim] * t


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
