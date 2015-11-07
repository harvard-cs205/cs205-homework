#cython: boundscheck=False, wraparound=False
from cython.parallel import parallel, prange
cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release

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
                     omp_lock_t *locks) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, dim
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    acquire(&(locks[i]))
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    release(&(locks[i]))
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.

    #get the grid coordinate of the point we consider
    cdef int x_grid = <int>(XY[i, 0]/grid_spacing)
    cdef int y_grid = <int>(XY[i, 1]/grid_spacing)
    # get the max and min x to check within the grid
    cdef int x_max =  min(x_grid+2, Grid.shape[0] - 1)
    cdef int x_min =  max(x_grid-2, 0)
    # get the max and min y to check within the grid
    cdef int y_max =  min(y_grid+2, Grid.shape[0] - 1)
    cdef int y_min =  max(y_grid-2, 0)

    cdef int k, l
    # loop over the points to check
    for k in range(x_min, x_max+1):
        for l in range(y_min, y_max):
            j = Grid[k, l]
            # only one side to avoir double counting the coolision and get rid of the -1 case
            if j>i:
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
             float t):
    cdef:
        int count = XY.shape[0]
        int i, j, dim
        int num_threads, chunksize
        FLOAT *XY1, *XY2, *V1, *V2
        int x_grid, y_grid, new_x_grid, new_y_grid
        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2
    chunksize = count/4
    num_threads = 4

    with nogil:

        # bounce off of walls
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, schedule = 'static', chunksize = chunksize, num_threads = num_threads):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        # for i in range(count):
        for i in prange(count, schedule = 'static', chunksize = chunksize, num_threads = num_threads):
            sub_update(XY, V, R, i, count, Grid, grid_spacing, locks)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        for i in prange(count, schedule = 'static', chunksize = chunksize, num_threads = num_threads):
            x_grid = <int>(XY[i, 0]/grid_spacing)
            y_grid = <int>(XY[i, 1]/grid_spacing)

            # Stay inbound if negative set to 0, if too large set to Grid.shape[dim] - 1
            x_grid =  max( min(x_grid, Grid.shape[0] - 1), 0)
            y_grid =  max( min(y_grid, Grid.shape[1] - 1), 0)

            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
            # update the grid coordinate by casting the integer value
            new_x_grid = <int>(XY[i, 0]/grid_spacing)
            new_y_grid = <int>(XY[i, 1]/grid_spacing)
            # Stay inbound if negative set to 0, if too large set to Grid.shape[dim] - 1
            new_x_grid =  max( min(new_x_grid, Grid.shape[0] - 1), 0)
            new_y_grid =  max( min(new_y_grid, Grid.shape[1] - 1), 0)

            #if the particule changed location
            if (x_grid != new_x_grid) or (y_grid != new_y_grid):
                #empty the former location
                Grid[x_grid, y_grid] = -1
                #populate the new one
                Grid[new_x_grid, new_y_grid] = i




def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
