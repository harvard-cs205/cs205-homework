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
        int j, dim, m, n
        UINT xidx, yidx
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    # with gil: print 'check0'
    xidx, yidx = <UINT> (XY[i, 0]/grid_spacing), <UINT> (XY[i, 1]/grid_spacing)
    # with gil: print 'checkMaybe'
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    # Need to check for collisions with the object o in grid squares marked with *:
    # *  *  *  *  *
    # *  *  *  *  *
    # *  *  o  *  *
    # *  *  *  *  *
    # *  *  *  *  *
    if (xidx < Grid.shape[0]) and (yidx < Grid.shape[0]): # object on grid?
        for m in range(max(2, xidx)-2, min(Grid.shape[0], xidx+3)): # handle objects near edge
            for n in range(max(2, yidx)-2, min(Grid.shape[0], yidx+3)): # same as above
                if (not (m == xidx and n == yidx)) and (i < Grid[m, n] < Grid.shape[0]):
                    XY2 = &(XY[Grid[m, n], 0])
                    V2 = &(V[Grid[m, n], 0])

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
        int count = XY.shape[0]
        int i, j, dim, num_thr, chunk, temp
        UINT tempidx1, tempidx2
        FLOAT *XY1, *XY2, *V1, *V2
        # SUBPROBLEM 4: uncomment this code.
        # omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2
    num_thr = 1
    chunk = 125

    with nogil:
        # bounce off of walls
        for i in prange(count, num_threads=num_thr, schedule='static', chunksize=chunk):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1
        # bounce off of each other
        for i in prange(count, num_threads=num_thr, schedule='static', chunksize=chunk):
            sub_update(XY, V, R, i, count, Grid, grid_spacing)
        # update positions
        #
        # SUBPROBLEM 2: update the grid values.
        # Whole bunch of if statements. The end result: only store an object's index in a 
        # grid square if the object is actually on the grid.
        for i in prange(count, num_threads=num_thr, schedule='static', chunksize=chunk):
            tempidx1 = <UINT> (XY[i, 0]/grid_spacing)
            tempidx2 = <UINT> (XY[i, 1]/grid_spacing)

            if (tempidx1 < Grid.shape[0]) and (tempidx2 < Grid.shape[0]):
                temp = Grid[tempidx1, tempidx2]

            for dim in range(2):
                XY[i, dim] += V[i, dim] * t

            if (((<UINT> (XY[i, 0]/grid_spacing)) < Grid.shape[0]) and
                ((<UINT> (XY[i, 1]/grid_spacing)) < Grid.shape[0])):
                Grid[<UINT> (XY[i, 0]/grid_spacing), <UINT> (XY[i, 1]/grid_spacing)] = i

                if (tempidx1 < Grid.shape[0]) and (tempidx2 < Grid.shape[0]):
                    Grid[tempidx1, tempidx2] = temp


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
