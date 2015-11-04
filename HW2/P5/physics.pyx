#cython: boundscheck=False, wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
from cython.parallel import parallel, prange, threadid

# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT

# checks if two balls are overlapping via Euclidean distance
cdef inline int overlapping(FLOAT *x1,
                            FLOAT *x2,
                            float R) nogil:
    cdef:
        float dx = x1[0] - x2[0]
        float dy = x1[1] - x2[1]
    return (dx * dx + dy * dy) < (4 * R * R)

# checks if two balls are moving away from each other
cdef inline int moving_apart(FLOAT *x1, FLOAT *v1,
                             FLOAT *x2, FLOAT *v2) nogil:
    cdef:
        float deltax = x2[0] - x1[0]
        float deltay = x2[1] - x1[1]
        float vrelx = v2[0] - v1[0]
        float vrely = v2[1] - v1[1]
    # check if delta and velocity in same direction
    return (deltax * vrelx + deltay * vrely) > 0.0

# determine the new velocities (conservation of momentum) due to a collision
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


# for ball i, update the velocity vector given possible two-body collisions
cdef void sub_update(FLOAT[:, ::1] XY,
                     FLOAT[:, ::1] V,
                     float R,
                     int i, int count,
                     UINT[:, ::1] Grid,
                     float grid_spacing) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, k, dim
        float eps = 1e-5
        unsigned int gx, gy, grid_size

    # SUBPROBLEM 4: Add locking
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.

    grid_size = <unsigned int> (1.0 / grid_spacing + 1)
    gx = <unsigned int> (XY[i,0] / grid_spacing)
    gy = <unsigned int> (XY[i,1] / grid_spacing)

    # iterate through a 5 x 5 square on the grid
    for j in range(gx - 2, gx + 3):
        for k in range(gy - 2, gy + 3):
            # ensure j,k are in the grid, and only collide with larger indices (avoid double collisions)
            if ((0 <= j < grid_size) and (0 <= k < grid_size) and (i < Grid[j,k] < count)):
                # obtain position and velocity for nearby neighbors
                XY2 = &(XY[Grid[j,k], 0])
                V2 = &(V[Grid[j,k], 0])
                # check if overlapping
                if overlapping(XY1, XY2, R):
                    # if overlapping, check if moving away
                    # SUBPROBLEM 4: Add locking
                    if not moving_apart(XY1, V1, XY2, V2):
                        # adjust velocities accordingly due to the collision
                        collide(XY1, V1, XY2, V2)

                    # give a slight impulse to help separate them
                    for dim in range(2):
                        V2[dim] += eps * (XY2[dim] - XY1[dim])


# update the position and velocity vectors of all balls
cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t):
    cdef:
        int count = XY.shape[0]
        int i, j, dim, thrds = 4, chunk_size = count / 4
        FLOAT *XY1, *XY2, *V1, *V2
        # SUBPROBLEM 4: uncomment this code.
        # omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    with nogil:
        # bounce off of walls
        # parallelize this loop over 4 threads, with static scheduling.
        for i in prange(count, num_threads=thrds, schedule="static", chunksize=chunk_size):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        # parallelize this loop over 4 threads, with static scheduling.
        for i in prange(count, num_threads=thrds, schedule="static", chunksize=chunk_size):
            sub_update(XY, V, R, i, count, Grid, grid_spacing)

        # update the grid values
        # parallelize this loop over 4 threads (with static scheduling).
        for i in prange(count, num_threads=thrds, schedule="static", chunksize=chunk_size):
            # if the object is in the grid, then take it out of its current position
            if (0 <= XY[i,0] <= 1 and 0 <= XY[i,1] <= 1):
                Grid[<unsigned int> (XY[i,0] / grid_spacing), <unsigned int> (XY[i,1] / grid_spacing)] = -1
            # update positions based on updated velocities
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
            # update grid, if the object is now in the grid
            if (0 <= XY[i,0] <= 1 and 0 <= XY[i,1] <= 1):
                Grid[<unsigned int> (XY[i,0] / grid_spacing), <unsigned int> (XY[i,1] / grid_spacing)] = i


# initialize locks for use
def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
