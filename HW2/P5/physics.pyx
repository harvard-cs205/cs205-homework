#cython: boundscheck=False, wraparound=False
from libc.stdio cimport printf
cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
from cython.parallel import parallel, prange

# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT

# get function form hilbert_curve c code, taken from wikipedia page on Hilbert curve
cdef extern from "hilbert_curve.c":
    int xy2d(int n, int x, int y)

# returns the d values for the 2d positions array
cpdef get_ds_from_positions(int n, UINT[:, ::1] positions, UINT[:] result, int length):
    cdef:
        int i

    for i in range(length):
        result[i] = xy2d(n, positions[i, 0], positions[i, 1])


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
        int j, dim, xgrid, ygrid, gridsize, x, y
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    acquire(&(locks[i]))
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.

    # for j in range(i + 1, count):
    #     XY2 = &(XY[j, 0])
    #     V2 = &(V[j, 0])
    #     if overlapping(XY1, XY2, R):
    #         # SUBPROBLEM 4: Add locking
    #         if not moving_apart(XY1, V1, XY2, V2):
    #             collide(XY1, V1, XY2, V2)

    #         # give a slight impulse to help separate them
    #         for dim in range(2):
    #             V2[dim] += eps * (XY2[dim] - XY1[dim])

    xgrid = <int> (XY[i, 0] / grid_spacing)
    ygrid = <int> (XY[i, 1] / grid_spacing)
    
    # we only check x and y values greater than x in order to avoid colliding the same balls
    # twice, once for lower values of x and y, then for the higher values
    # check x, x+1 and x+2 
    for x in range(xgrid, xgrid+3):
        # check y values making sure that distance from original (x,y) is at most 2
        for y in range(ygrid, ygrid+3-(x-xgrid)):
            gridsize = <int> ((1.0 / grid_spacing) + 1)
            # check that we are checking for valid grid squares, and that (x,y) is not the original
            # square because then we might think that a ball is colliding with itself
            # check that ballnumber is a valid ball
            if x < gridsize and y < gridsize and (x != xgrid or y != ygrid) and Grid[x,y] < count:
                j = Grid[x, y]
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

# checks that a ball is in the grid
cdef int in_grid(int i, FLOAT[:, ::1] XY) nogil:
    return (XY[i, 0] >= 0 and XY[i, 0] <= 1 and XY[i, 1] >=0 and XY[i, 1] <= 1)

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
        int xgrid, ygrid
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
        # for i in range(count):
        for i in prange(count, num_threads = 4, schedule = 'static', chunksize = count/4):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        # for i in range(count):
        for i in prange(count, num_threads = 4, schedule = 'static', chunksize = count/4):
            sub_update(XY, V, R, i, count, Grid, grid_spacing, locks)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        # for i in range(count):
        for i in prange(count, num_threads = 4, schedule = 'static', chunksize = count/4):
            # initialize all currently used squares to -1
            if in_grid(i, XY):
                xgrid = <int> (XY[i, 0] / grid_spacing)
                ygrid = <int> (XY[i, 1] / grid_spacing)
                Grid[xgrid, ygrid] = -1
            
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
            
            # update the grid with new positions
            if in_grid(i, XY):
                xgrid = <int> (XY[i, 0] / grid_spacing)
                ygrid = <int> (XY[i, 1] / grid_spacing)
                Grid[xgrid, ygrid] = i    
            


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
