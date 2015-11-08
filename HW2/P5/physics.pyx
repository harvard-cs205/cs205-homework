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
                     uintptr_t locks_ptr,) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, dim, gx, gy, grid_size
        float eps = 1e-5
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    # SUBPROBLEM 4: Add locking

    # original code
    #XY1 = &(XY[i, 0])
    #V1 = &(V[i, 0])
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

    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.

    # get grid index for current i
    grid_size = int((1.0 / grid_spacing) + 1)

    gx = min(grid_size - 1, max(0, <int>(XY[i, 0] / grid_spacing)))
    gy = min(grid_size - 1, max(0, <int>(XY[i, 1] / grid_spacing)))

    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])

    # update 5x5 neighborhood
    for x in range(max(0, gx - 2), min(gx + 3, grid_size)):
        for y in range(max(0, gy - 2), min(gy + 3, grid_size)):
            # retrieve index of cell
            j = Grid[x, y]

            # only update if pair not visited yet
            # (this here excludes also the case where gx == x, gy == y)
            if j > i:
                XY2 = &(XY[j, 0])
                V2 = &(V[j, 0])
                if overlapping(XY1, XY2, R):
                    # SUBPROBLEM 4: Add locking
                    if not moving_apart(XY1, V1, XY2, V2):

                        # use locks for both i, j
                        acquire(&locks[i])
                        acquire(&locks[j])
                        collide(XY1, V1, XY2, V2)
                        release(&locks[j])
                        release(&locks[i])

                    # lock only for j
                    acquire(&locks[j])        
                    # give a slight impulse to help separate them
                    for dim in range(2):
                        V2[dim] += eps * (XY2[dim] - XY1[dim])
                    release(&locks[j])

    

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t):
    cdef:
        int count = XY.shape[0]
        int i, j, dim, csize, nthreads, grid_size
        FLOAT *XY1, *XY2, *V1, *V2
        int gx, gy
        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    # set here threads & chunksize
    csize = int(0.25 * count)
    nthreads = 4

    grid_size = int((1.0 / grid_spacing) + 1)

    with nogil:
        # bounce off of walls
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        #for i in range(count):
        for i in prange(count, schedule='static', num_threads=nthreads, chunksize=csize):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, schedule='static', num_threads=nthreads, chunksize=csize):
            sub_update(XY, V, R, i, count, Grid, grid_spacing, locks_ptr)

        # reset all grid indices
        for i in prange(grid_size, schedule='static', num_threads=nthreads):
            for j in range(grid_size):
                Grid[i, j] = -1 # should be casted automatically

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.

        for i in prange(count, schedule='static', num_threads=nthreads, chunksize=csize):
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t

            # retrieve grid position
            gx = min(grid_size - 1, max(0, <int>(XY[i, 0] / grid_spacing)))
            gy = min(grid_size - 1, max(0, <int>(XY[i, 1] / grid_spacing)))

            # set grid to index
            Grid[gx, gy] = i


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
