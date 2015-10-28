#cython: boundscheck=False, wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from cython.parallel import parallel, prange, threadid
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release

from libc.stdio cimport printf
cdef extern from "math.h" :
    double sqrt(double) nogil

# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT

cdef enum:
    NO_INDEX = 4294967295

cdef inline int cuttail(FLOAT XY,
                        float grid_spacing,
                        int grid_size) nogil:
    cdef:
        int gridpos = int(XY/grid_spacing)
    if gridpos < 0:
        return 0
    elif gridpos >= grid_size:
        return grid_size - 1
    else:
        return gridpos

    
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
                     int grid_size,
                     omp_lock_t *locks) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, dim
        int XYG[2]
        float eps = 1e-5
        int gsx, gsy
        int gs = 3 # the size of the circle we are going to search

    # SUBPROBLEM 4: Add locking
    acquire(&locks[i])
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    #===original code===#
    # for j in range(i + 1, count):
        # XY2 = &(XY[j, 0])
        # V2 = &(V[j, 0])
        # if overlapping(XY1, XY2, R):
            # # SUBPROBLEM 4: Add locking
            # if not moving_apart(XY1, V1, XY2, V2):
                # collide(XY1, V1, XY2, V2)

            # # give a slight impulse to help separate them
            # for dim in range(2):
                # V2[dim] += eps * (XY2[dim] - XY1[dim])
    #===================#
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    # obtain the grid index of XY1
    XYG[0] = int(XY1[0]/grid_spacing)
    XYG[1] = int(XY1[1]/grid_spacing)
    # printf("grid_spacing %f\n", grid_spacing)
    # printf("position: %f %f\n", XY1[0], XY1[1])
    # printf("grid index: %i %i \n", XYG[0], XYG[1])
    for gsx in range(XYG[0]-gs, XYG[0]+gs):
        if gsx < 0 or gsx >= grid_size: #out of boundary
            continue
        for gsy in range(XYG[1]-gs, XYG[1]+gs):
            if gsy < 0 or gsy >= grid_size: #out of boundary
                continue
            j = Grid[gsx, gsy]
            if j == NO_INDEX or j <= i: #no ball at this grid or the ball is itself; prevent collide twice: only collide with larger index
                continue
            #printf("after j %u gsx %i gsy %i\n", j, gsx, gsy)
            acquire(&locks[j])
            XY2 = &(XY[j, 0])
            V2 = &(V[j, 0])
            if overlapping(XY1, XY2, R):
                # SUBPROBLEM 4: Add locking
                if not moving_apart(XY1, V1, XY2, V2):
                    collide(XY1, V1, XY2, V2)

                # give a slight impulse to help separate them
                for dim in range(2):
                    V2[dim] += eps * (XY2[dim] - XY1[dim])
            release(&locks[j])
    release(&locks[i])
                    

cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             int grid_size,
             uintptr_t locks_ptr,
             float t):
    cdef:
        int count = XY.shape[0], n_th = 4
        int chunk = count/n_th
        int i, j, dim, id
        int XYG0, XYG1
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
        for i in prange(count, schedule='static', chunksize=chunk, num_threads=n_th ):
        #for i in range(count):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, schedule='static', chunksize=chunk, num_threads=n_th ):
        #for i in range(count):
            sub_update(XY, V, R, i, count, Grid, grid_spacing, grid_size, locks)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # for i in prange(count, schedule='static', chunksize=chunk, num_threads=n_th ):
        # #for i in range(count):
            # for dim in range(2):
                # XY[i, dim] += V[i, dim] * t
        # SUBPROBLEM 2: update the grid values.
        for i in prange(count, schedule='static', chunksize=chunk, num_threads=n_th ):
        #for i in range(count):
            # obtain the grid index of XYi
            id = threadid()
            XYG0 = cuttail(XY[i, 0], grid_spacing, grid_size)
            XYG1 = cuttail(XY[i, 1], grid_spacing, grid_size)
            #printf("%i: XYG0=%i XYG1=%i grid_size=%i\n", id, XYG0, XYG1, grid_size)
            if Grid[XYG0, XYG1] == i: # if this position has no other ball, set to no ball
                Grid[XYG0, XYG1] = NO_INDEX
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
            XYG0 = cuttail(XY[i, 0], grid_spacing, grid_size)
            XYG1 = cuttail(XY[i, 1], grid_spacing, grid_size)
            Grid[XYG0, XYG1] = i


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
