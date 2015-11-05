#cython: boundscheck=True, wraparound=False

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
                     int grid_size,
                     float grid_spacing,
                     omp_lock_t *locks) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int dim, j
        float eps = 1e-5
        int grid_x, grid_y, x_start, x_end, y_start, y_end, x, y

    # SUBPROBLEM 4: Add locking
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    grid_x = <int>(XY[i,0]/ grid_spacing)
    grid_y = <int>(XY[i,1]/ grid_spacing)
    
    # Create the 5x5 bounding box
    x_start = grid_x - 2
    x_end = grid_x + 2
    y_start = grid_y - 2
    y_end = grid_y + 2
    
    # Ensure bounding box does not exceed edges of grid
    if x_start < 0:
        x_start = 0
    if x_end > grid_size - 1:
        x_end = grid_size - 1
    if y_start < 0:
        y_start = 0
    if y_end > grid_size - 1:
        y_end = grid_size - 1
        
    # Iterate over the bounding box and check for other balls within 2 boxes
    for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
            if Grid[x, y] != -1:
                # Make sure we only handle each collision once and we don't collide with ourselves
                if (grid_x < x) or (grid_x == x and grid_y < y):
                    j = Grid[x, y]
                    if j != i:
                        XY2 = &(XY[j, 0])
                        V2 = &(V[j, 0])
                        if overlapping(XY1, XY2, R):
                            # SUBPROBLEM 4: Add locking
                            # Acquire the locks (lower id first)
                            if i < j:
                                acquire(&(locks[i]))
                                acquire(&(locks[j]))
                            else:
                                acquire(&(locks[j]))
                                acquire(&(locks[i]))
                            
                            if not moving_apart(XY1, V1, XY2, V2):
                                collide(XY1, V1, XY2, V2)
                                
                            # Release the locks
                            release(&(locks[i]))
                            release(&(locks[j]))

                            # give a slight impulse to help separate them
                            for dim in range(2):
                                V2[dim] += eps * (XY2[dim] - XY1[dim])

cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_size,
             uintptr_t locks_ptr,
             float t,
             float grid_spacing):
    cdef:
        int count = XY.shape[0]
        int i, j, dim
        FLOAT *XY1, *XY2, *V1, *V2
        int num_threads = 4
        int chunksize = count/4
        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2
    
    with nogil:
        # bounce off of walls
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, schedule='static', chunksize=chunksize, num_threads=num_threads):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, schedule='static', chunksize=chunksize, num_threads=num_threads):
            sub_update(XY, V, R, i, count, Grid, <int>grid_size, grid_spacing, locks)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        for i in prange(count, schedule='static', chunksize=chunksize, num_threads=num_threads):
            # Remove the ball from this position on the grid
            Grid[<int>(XY[i,0]/ grid_spacing), <int>(XY[i,1]/ grid_spacing)] = -1
            # Move the ball
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
            # After the position for this ball been updated, update the grid value with the new positions
            Grid[<int>(XY[i,0]/ grid_spacing), <int>(XY[i,1]/ grid_spacing)] = i

def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks