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

# Function to rectify grid index boundaries        
cdef int correct_bounds(int grid_ind, int grid_sz) nogil:
        # Make sure we're not indexing outside the grid    
    if grid_ind < 0:
        grid_ind = 0
    if grid_ind > grid_sz - 1:
        grid_ind = grid_sz - 1
    return grid_ind
    
cdef void sub_update(FLOAT[:, ::1] XY,
                     FLOAT[:, ::1] V,
                     float R,
                     int i, int count,
                     UINT[:, ::1] Grid,
                     float grid_size,
                     omp_lock_t *locks) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, dim, grid_ind_x1, grid_ind_y1, 
        int x_search_grid_i, y_search_grid_i, probe_grid_x, probe_grid_y
        float eps = 1e-5
        UINT obj_ind, dummy_max = -1
        
    # SUBPROBLEM 4: Add locking
    acquire(&(locks[i]))
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################


#==============================================================================
#     for j in range(i + 1, count):
#         XY2 = &(XY[j, 0])
#         V2 = &(V[j, 0])
#         
# #        overlap_wrap(XY1, V1, XY2, V2, R, eps, grid_size, i, j)
#         if overlapping(XY1, XY2, R):
#             # SUBPROBLEM 4: Add locking
#             if not moving_apart(XY1, V1, XY2, V2):
#                 collide(XY1, V1, XY2, V2)
# 
#             # give a slight impulse to help separate them
#             for dim in range(2):
#                 V2[dim] += eps * (XY2[dim] - XY1[dim])
#==============================================================================
                
    # Find the grid coordinates of the current particle
    grid_ind_x1 = <int>(XY1[0] * grid_size)
    grid_ind_y1 = <int>(XY1[1] * grid_size)
        
    # Make sure we're not indexing outside the grid   
    grid_ind_x1 = correct_bounds(grid_ind_x1, Grid.shape[0])
    grid_ind_y1 = correct_bounds(grid_ind_y1, Grid.shape[0])
        
    # Only check the 5 by 5 window around the center space
    for x_search_grid_i in range(-2, 3):
        for y_search_grid_i in range(-2, 3):
            # Determine which square we are looking at
            probe_grid_x = grid_ind_x1 + x_search_grid_i
            probe_grid_y = grid_ind_y1 + y_search_grid_i

            # Check the grid location only if the probing location is a valid one
            # and if it's not in the center
            if (probe_grid_x == correct_bounds(probe_grid_x, Grid.shape[0])
            and probe_grid_y == correct_bounds(probe_grid_y, Grid.shape[0])
            and not (x_search_grid_i == 0 and y_search_grid_i == 0)
            and Grid[probe_grid_x, probe_grid_y] != i):

                # Check out the probing spot
                obj_ind = Grid[probe_grid_x, probe_grid_y]

                # If it's not empty, 
                # and make sure we don't count collisions twice.
                if (obj_ind != dummy_max
                and obj_ind >= i + 1 
                and obj_ind < count): 
                    acquire(&(locks[obj_ind]))
                    XY2 = &(XY[obj_ind, 0])
                    V2 = &(V[obj_ind, 0])
                    if overlapping(XY1, XY2, R):
                        # SUBPROBLEM 4: Add locking
                        if not moving_apart(XY1, V1, XY2, V2):
                            collide(XY1, V1, XY2, V2)
    
                        # give a slight impulse to help separate them
                        for dim in range(2):
                            V2[dim] += eps * (XY2[dim] - XY1[dim])
                    release(&(locks[obj_ind]))
    release(&(locks[i]))



cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_size,
             uintptr_t locks_ptr,
             float t):
    cdef:
        int count = XY.shape[0]
        int i, j, dim, n_threads = 4, chunksz
        FLOAT *XY1, *XY2, *V1, *V2, prev_pos[2]
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr
        
    chunksz = <int>(round(R / n_threads))
    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    
    with nogil:
        # bounce off of walls
        for i in prange(count, 
                    schedule='static', 
                    chunksize=chunksz,
                    num_threads=n_threads):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        for i in prange(count, 
                        schedule='static', 
                        chunksize=chunksz,
                        num_threads=n_threads):
            sub_update(XY, V, R, i, count, Grid, grid_size, locks)

        # update positions
        for i in prange(count, 
                        schedule='static', 
                        chunksize=chunksz,
                        num_threads=n_threads):
            for dim in range(2):
                # Record the previous particle position 
                # so that we can update the grid
                prev_pos[dim] = XY[i, dim]

                XY[i, dim] += V[i, dim] * t
            # Move object i to the new grid spot                
            Grid[correct_bounds(<int>(prev_pos[0] * grid_size), 
                                Grid.shape[0]),
                 correct_bounds(<int>(prev_pos[1] * grid_size), 
                                Grid.shape[0])] = -1
            Grid[correct_bounds(<int>(XY[i, 0] * grid_size), Grid.shape[0]),
                 correct_bounds(<int>(XY[i, 1] * grid_size), Grid.shape[0])] = i

def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
