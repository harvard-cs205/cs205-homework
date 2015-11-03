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

cdef extern from "math.h":
   double abs(double) nogil

cdef inline int overlapping(FLOAT *x1,
                            FLOAT *x2,
                            float R) nogil:
    cdef:
        float dx = x1[0] - x2[0]
        float dy = x1[1] - x2[1]
    return (dx * dx + dy * dy) < (4 * R * R)


cdef inline void get_grid_location( float ball_x,
                                    float ball_y,
                                    UINT[:, ::1] Grid,
                                    float grid_spacing,
                                    int* xy) nogil:

    cdef:
        int w, h

    w = Grid.shape[0]
    h = Grid.shape[1]
    xy[0] = <int>(ball_x / grid_spacing)
    xy[1] = <int>(ball_y / grid_spacing)
    
    xy[0] = max(0, min(xy[0], w - 1))
    xy[1] = max(0, min(xy[1], h - 1))


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
        int j, x_i, y_i, dim
        int grid_xy_1[2], grid_xy_2[2]
        int x_start, x_end, y_start, y_end
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])

    # get current grid location of the ball
    get_grid_location(XY[i, 0], XY[i, 1], Grid, grid_spacing, grid_xy_1)

    # get the grid neighbors within 2 cell distance to check for overlap
    x_start = max(0, grid_xy_1[0] - 3)
    x_end = min(grid_xy_1[0] + 3, Grid.shape[0] - 1)

    y_start = max(0, grid_xy_1[1] - 3)
    y_end = min(grid_xy_1[1] + 3, Grid.shape[1] - 1)

    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    # Loop through 2-cell distance neighbors to check for overlap
    for x_i in range(x_start, x_end + 1):
        for y_i in range(y_start, y_end + 1):
            
            # get index of ball from grid position
            j = Grid[x_i, y_i]
            if j == -1 or j <= i:
                continue
            
            XY2 = &(XY[j, 0])
            V2 = &(V[j, 0])

            # get_grid_location(XY[j, 0], XY[j, 1], Grid, grid_spacing, grid_xy_2)

            # # no need to check overlap if grid positions are more than 2 cells apart
            # if abs(grid_xy_1[0] - grid_xy_2[0]) > 2 and abs(grid_xy_1[1] - grid_xy_2[1]) > 2:
            #     continue
            
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
             float t):
    cdef:
        int parallelism = 4
        int chunk_size = 2500
        int count = XY.shape[0]
        int i, j, dim
        int grid_old_xy[2], grid_new_xy[2]
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
        for i in prange(count, schedule='static', chunksize=chunk_size, num_threads=parallelism):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, schedule='static', chunksize=chunk_size, num_threads=parallelism):
            sub_update(XY, V, R, i, count, Grid, grid_spacing)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        for i in prange(count, schedule='static', chunksize=chunk_size, num_threads=parallelism):
            get_grid_location(XY[i, 0], XY[i, 1], Grid, grid_spacing, grid_old_xy)

            for dim in range(2):
                XY[i, dim] += V[i, dim] * t

            get_grid_location(XY[i, 0], XY[i, 1], Grid, grid_spacing, grid_new_xy)

            if grid_old_xy[0] != grid_new_xy[0] or grid_old_xy[1] != grid_new_xy[1]:
                Grid[grid_old_xy[0], grid_old_xy[1]] = -1
                Grid[grid_new_xy[0], grid_new_xy[1]] = i


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
