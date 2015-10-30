#cython: boundscheck=False, wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
from cython.parallel import prange, threadid
from openmp cimport omp_lock_t, \
    omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock, omp_get_thread_num

# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT

# Helper functions set hilbert order array list
cdef inline void set_hilbert_array_c(FLOAT[:, ::1] XY,
                                   UINT[:, :] hilbert_map, 
                                   UINT[:] hilbert_array,
                                   float grid_spacing):
    cdef:
        int i, x, y
    with nogil:
        for i in prange(XY.shape[0], num_threads=4):
            x = <int>(XY[i, 0] / grid_spacing)
            y = <int>(XY[i, 1] / grid_spacing)
            hilbert_array[i] = hilbert_map[x, y]

cpdef set_hilbert_array(FLOAT[:, ::1] XY,
                        UINT[:, :] hilbert_map, 
                        UINT[:] hilbert_array,
                        float grid_spacing):
    set_hilbert_array_c(XY, hilbert_map, hilbert_array, grid_spacing)

# Helper functions clear & set grid
cdef inline void cleargrid_c(FLOAT[:, ::1] XY,
                           UINT[:, ::1] Grid,
                           float grid_spacing) nogil:
    cdef:
        int i, x, y
    for i in range(XY.shape[0]):
        if XY[i, 0] < 0 or XY[i, 1] < 0 or XY[i, 0] > 1 or XY[i, 1] > 1:
            continue
        x = <int>(XY[i, 0] / grid_spacing)
        y = <int>(XY[i, 1] / grid_spacing)
        Grid[x, y] = -1

cdef inline void setgrid_c(FLOAT[:, ::1] XY,
                         UINT[:, ::1] Grid,
                         float grid_spacing) nogil:
    cdef:
        int i, x, y
    for i in range(XY.shape[0]):
        if XY[i, 0] < 0 or XY[i, 1] < 0 or XY[i, 0] > 1 or XY[i, 1] > 1:
            continue
        x = <int>(XY[i, 0] / grid_spacing)
        y = <int>(XY[i, 1] / grid_spacing)
        Grid[x, y] = i

cpdef cleargrid(FLOAT[:, ::1] XY,
                UINT[:, ::1] Grid,
                float grid_spacing):
    cleargrid_c(XY, Grid, grid_spacing)

cpdef setgrid(FLOAT[:, ::1] XY,
              UINT[:, ::1] Grid,
              float grid_spacing):
    setgrid_c(XY, Grid, grid_spacing)

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
        int x, y
        int ix = <int>(XY[i, 0] / grid_spacing)
        int iy = <int>(XY[i, 1] / grid_spacing)
        # x1, x2, y1, y2 is the bounding rect to detect collision
        int x1 = max(0, ix - 2)
        int y1 = max(0, iy - 2)
        int x2 = min(Grid.shape[0], ix + 2 + 1)
        int y2 = min(Grid.shape[1], iy + 2 + 1)
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    omp_set_lock(&locks[i])
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])

    #print("%d %d %d %d %d %u" % (i, x1, x2, y1, y2, Grid[ix, iy]))
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    for x in range(x1, x2):
        for y in range(y1, y2):
            j = Grid[x, y]
            # if location(x, y) is not occupied by a ball, continue
            # if j <= i, to avoid "twice collision", continue
            if j == -1 or j <= i:
                continue
            XY2 = &(XY[j, 0])
            V2 = &(V[j, 0])
            if overlapping(XY1, XY2, R):
                # SUBPROBLEM 4: Add locking
                omp_set_lock(&locks[j])
                if not moving_apart(XY1, V1, XY2, V2):
                    collide(XY1, V1, XY2, V2)
                omp_unset_lock(&locks[j])

                # give a slight impulse to help separate them
                for dim in range(2):
                    V2[dim] += eps * (XY2[dim] - XY1[dim])
    omp_unset_lock(&locks[i])

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
        int x, y
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
        for i in prange(count, num_threads=4, schedule="static", chunksize=count/4):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, num_threads=4, schedule="static", chunksize=count/4):
            sub_update(XY, V, R, i, count, Grid, grid_spacing, locks)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        cleargrid_c(XY, Grid, grid_spacing)

        for i in prange(count, num_threads=4, schedule="static", chunksize=count/4):
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t

        setgrid_c(XY, Grid, grid_spacing)


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
