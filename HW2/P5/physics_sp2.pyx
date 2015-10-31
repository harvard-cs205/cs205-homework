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
                     float grid_spacing) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int k, j, dim
        int grid_x_dim, grid_y_dim
        FLOAT x1, y1
        int grid_x1, grid_y1, other_ball_grid_val
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])

    # Get the current particles coordinate to look up in the grid
    x1 = XY[i, 0]
    y1 = XY[i, 1]
    grid_x_dim = Grid.shape[0]
    grid_y_dim = Grid.shape[1]

    # And find where they are in the grid
    grid_x1 = int(x1 / grid_spacing)
    grid_y1 = int(y1 /grid_spacing)
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    for j in range(-2, 3):
        for k in range(-2, 3):
            if (j != 0) or (k != 0):
                # See if somebody is in one of these adjacent grid points
                # But first check our bounds:
                if (grid_x1 + j < grid_x_dim) and (grid_y1 + k < grid_y_dim) and (grid_x1 + j > 0) and (grid_y1 + k > 0):
                    other_ball_grid_val = Grid[grid_x1 + j, grid_y1 + k]

                    # If we have someone there...
                    if (other_ball_grid_val != -1):

                        # then treat it accordingly
                        XY2 = &(XY[other_ball_grid_val, 0])
                        V2 = &(V[other_ball_grid_val, 0])

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
             int nthreads):
    cdef:
        int count = XY.shape[0]
        int i, j, dim, new_grid_x, new_grid_y, old_grid_x, old_grid_y
        int chunk_size = count / nthreads 
        FLOAT *XY1, *XY2, *V1, *V2
        # SUBPROBLEM 4: uncomment this code.
        # omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    with nogil:
        # bounce off of walls
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, num_threads=nthreads, chunksize=chunk_size, schedule='static'):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, num_threads=nthreads, chunksize=chunk_size, schedule='static'):
            sub_update(XY, V, R, i, count, Grid, grid_spacing)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        for i in prange(count, num_threads=nthreads, chunksize=chunk_size, schedule='static'):

            old_grid_x = int(XY[i, 0] / grid_spacing)
            old_grid_y = int(XY[i, 1] / grid_spacing)

            for dim in range(2):
                XY[i, dim] += V[i, dim] * t

            # Now update the grid according to the new coordinates
            # If two fall in the same point, then just overwrite and take the latest ones
            new_grid_x = int(XY[i, 0] / grid_spacing)
            new_grid_y = int(XY[i, 1] / grid_spacing)

            # Also set our old point to empty since we aren't there any more
            # This may overwrite someone who IS there already - e.g., say particle 1 and particle 2
            # are in the same grid point, but particle 2 was put in there later, so its value is stored in grid.
            # This shouldn't matter because we are sweeping over the whole grid and updating everything anyways
            Grid[old_grid_x, old_grid_y] = -1
            Grid[new_grid_x, new_grid_y] = i


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
