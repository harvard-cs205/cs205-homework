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
                     float grid_spacing) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, dim
        int dx, dy
        int my = 2, mx = 2 # Max distance for collisions
        int gX1, gY1
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.

    # Collide with up to 2 squares away, in the right and down direction
    # gX1, gY1 = getGridXY(XY1, grid_spacing) # Get XY1's position in the grid
    gX1 = <int> (XY1[1] / grid_spacing)
    gY1 = <int> (XY1[0] / grid_spacing)

    # with gil:
    #     print gX1, XY1[0]/grid_spacing
    #     print gY1, XY1[1]/grid_spacing
    for dx in range(mx+1):
        for dy in range(my+1):
            # Up to 2 squares away (vertical + horizontal) and inside the grid
            if dx + dy > 2 or dx + dy == 0 or \
                    not inRange(gX1 + dx, gY1 + dy, Grid.shape[0]) or \
                    Grid[gY1 + dy, gX1 + dx] == i or \
                    Grid[gY1 + dy, gX1 + dx] == -1:
                continue
            XY2 = &(XY[Grid[gY1 + dy, gX1 + dx], 0])
            V2 = &(V[Grid[gY1 + dy, gX1 + dx], 0])

    # for j in range(i + 1, count):
    #     XY2 = &(XY[j, 0])
    #     V2 = &(V[j, 0])
            if overlapping(XY1, XY2, R):
                # SUBPROBLEM 4: Add locking
                if not moving_apart(XY1, V1, XY2, V2):
                    # if (XY1[0] - XY2[0]) < eps and (XY1[1] - XY2[1]) < eps:
                    #     # with gil:
                    #     #     print 'Colliding', XY1[0], XY1[1], XY2[0], XY2[1]
                    #     with gil:
                    #         print i, Grid[gY1 + dy, gX1 + dx]
                    # else:
                    #     with gil:
                    #         print 'No problem'


                    collide(XY1, V1, XY2, V2)

                # give a slight impulse to help separate them
                for dim in range(2):
                    V2[dim] += eps * (XY2[dim] - XY1[dim])

cdef inline int inRange(int x, int y, int N) nogil:
    return <int> (x >= 0 and y >= 0 and x < N and y < N)

cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t):
    cdef:
        int count = XY.shape[0]
        int chnk = count / 4
        int nt = 4
        int i, j, dim
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
        for i in prange(count, schedule='static', num_threads=nt, chunksize=chnk):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, schedule='static', num_threads=nt, chunksize=chnk):
            sub_update(XY, V, R, i, count, Grid, grid_spacing)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        for i in prange(count, schedule='static', num_threads=nt, chunksize=chnk):
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
            Grid[ <int> (XY[i, 0]/grid_spacing), <int> (XY[i, 1]/grid_spacing) ] = i



def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
