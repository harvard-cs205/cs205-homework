# Note: includes all my comments from reviewing the skeleton code

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

# Check if two balls are overlapping - used in original code version
cdef inline int overlapping(FLOAT *x1,
                            FLOAT *x2,
                            float R) nogil:
    cdef:
        float dx = x1[0] - x2[0]
        float dy = x1[1] - x2[1]
    return (dx * dx + dy * dy) < (4 * R * R)

# Check if two balls are moving apart
cdef inline int moving_apart(FLOAT *x1, FLOAT *v1,
                             FLOAT *x2, FLOAT *v2) nogil:
    cdef:
        float deltax = x2[0] - x1[0]
        float deltay = x2[1] - x1[1]
        float vrelx = v2[0] - v1[0]
        float vrely = v2[1] - v1[1]
    # check if delta and velocity in same direction
    return (deltax * vrelx + deltay * vrely) > 0.0

# Update ball velocities if they collide
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

# Helper function to acquire locks in order
cdef void acquire_locks(int idx1, int idx2, omp_lock_t *locks) nogil:

    # Acquire in ascending order - to avoid deadlock
    if idx1 < idx2:
        acquire(&(locks[idx1]))
        acquire(&(locks[idx2]))
    else:
        acquire(&(locks[idx2]))
        acquire(&(locks[idx1]))

# Helper function to release locks in order
cdef void release_locks(int idx1, int idx2, omp_lock_t *locks) nogil:

    # Release in ascending order - to avoid deadlock
    if idx1 < idx2:
        release(&(locks[idx1]))
        release(&(locks[idx2]))
    else:
        release(&(locks[idx2]))
        release(&(locks[idx1]))

# Handles ball collisions
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
        int j, dim
        int check_x, check_y, x, y
        float eps = 1e-5

    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])

    #############################################################
    # Updated code - based on grid
    ############################################################

    check_x = int(XY[i, 0] / grid_spacing)
    check_y = int(XY[i, 1] / grid_spacing)

    # Check area of potential overlap (x coordinates)
    for x in range(check_x - 2, check_x + 3):

        # Check that x coordinate is in bounds
        if x >= 0 and x < grid_size:

            # Check area of potential overlap (y coordinates)
            for y in range(check_y - 2, check_y + 3):

                # Check that y coordinate is in bounds
                if y >= 0 and y < grid_size:

                    # Look up corresponding ball index (if any)
                    j = Grid[x, y]

                    # Check that grid square isn't empty and id is different
                    if j != -1 and j != i:

                        # Look up position and velocity of ball for comparison
                        XY2 = &(XY[j, 0])
                        V2 = &(V[j, 0])

                        # Acquire locks (before if statement to avoid another
                        # thread affecting the result of moving_apart)
                        acquire_locks(i, j, locks)

                        # Update velocities if balls aren't moving apart
                        if not moving_apart(XY1, V1, XY2, V2):
                            collide(XY1, V1, XY2, V2)
                        
                        # Release locks
                        release_locks(i, j, locks)

                        # Give a slight impulse to help separate the balls
                        for dim in range(2):
                            V2[dim] += eps * (XY2[dim] - XY1[dim])

    #############################################################
    # Original code - based on positions
    ############################################################

    # Range set to avoid colling two balls twice.
    # for j in range(i + 1, count):
        
    #     XY2 = &(XY[j, 0])
    #     V2 = &(V[j, 0])
        
    #     if overlapping(XY1, XY2, R):
            
    #         if not moving_apart(XY1, V1, XY2, V2):
    #             collide(XY1, V1, XY2, V2)

    #         for dim in range(2):
    #             V2[dim] += eps * (XY2[dim] - XY1[dim])

# Updates ball positions & velocities
cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             int grid_size,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t,
             int num_threads,
             int chunk_size):
    cdef:
        int count = XY.shape[0] # Number of balls
        int i, dim
        int grid_x, grid_y
        FLOAT *XY1, *XY2, *V1, *V2
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    with nogil:

        # Bounce off of walls - updates velocity, not position
        for i in prange(count, schedule='static', chunksize=chunk_size, num_threads=num_threads):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # Bounce off of each other - updates velocity, not position
        for i in prange(count, schedule='static', chunksize=chunk_size, num_threads=num_threads):
            sub_update(XY, V, R, i, count, Grid, grid_size, grid_spacing, locks)

        # Reset grid values - we can't just 'move' ball coordinates as we are only keeping track
        # of any one ball in a given grid position.
        Grid[:] = -1

        # Update positions
        for i in prange(count, schedule='static', chunksize=chunk_size, num_threads=num_threads):
            
            # Update x, y coordinates
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t

            # Determine grid indices for new coordinates
            grid_x = int(XY[i, 0] / grid_spacing)
            grid_y = int(XY[i, 1] / grid_spacing)

            # Check that indices aren't out of bounds
            if grid_x >= 0 and grid_x < grid_size and grid_y >=0 and grid_y < grid_size:
                
                # Update grid value
                Grid[grid_x, grid_y] = i

# Creates a given number of locks
def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
