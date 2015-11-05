# cython: boundscheck=False, wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
from cython.parallel import prange
import math
#

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
                     omp_lock_t *locks) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2, XY1_x, XY1_y
        int j, k, dim, grid1_x, grid1_y, min_x, max_x, min_y, max_y, XY2index
        # The maximum grid index is the number of rows minus one.  The grid is always square.
        int maxindex = Grid.shape[0]-1  
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    acquire(&locks[i]) # Acquire the lock that corresponds to ball i.

    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    """
    for objects that are within two grid spaces:
    need grid coordinates of i
    make a 5X5 box around i by adding and subtracting to i's grid coordinates
    need to make sure not to mess up by checking out of the grid bounds
    """    
    XY1_x = XY1[0] # Extract x value of XY1
    XY1_y = XY1[1] # Extract y value of XY1
    
    # Find grid indices the same way as  in driver.py
    grid1_x = <int>(XY1_x/grid_spacing) 
    grid1_y = <int>(XY1_y/grid_spacing)

    ### Define min and max grid indices to check below
    # If the 5X5 grid box will go out of bounds on the left, set the min to zero.
    if grid1_x < 2: 
        min_x = 0
        max_x = grid1_x + 2
    # If it will go out of bounds on the right, set the max to the maximum index.
    elif grid1_x > maxindex - 2:
        min_x = grid1_x - 2
        max_x = maxindex
    # Otherwise, define the box as normal.  
    else:
        min_x = grid1_x - 2
        max_x = grid1_x + 2    
    
    # Analogous process for y-values:
    if grid1_y < 2: 
        min_y = 0
        max_y = grid1_y + 2
    elif grid1_y > maxindex - 2:
        min_y = grid1_y - 2
        max_y = maxindex  
    else:
        min_y = grid1_y - 2
        max_y = grid1_y + 2    
    
    # Note that this could cause an error if Grid is smaller than 5X5
    
    # Now iterate over the 5X5 grid
    for j in range(min_x, max_x):
        for k in range(min_y, max_y):
            # Take the value inside the grid. 
            XY2index = Grid[j,k]
            # This value is either a ball position index, or a -1 value.  I look only at balls that
            #    I haven't already looked at.  
            if XY2index > i:
                XY2 = &(XY[XY2index, 0])
                V2 = &(V[XY2index, 0])

                if overlapping(XY1, XY2, R):

                    # SUBPROBLEM 4: Add locking
                    acquire(&locks[XY2index]) # Acquire the lock for the second ball

                    if not moving_apart(XY1, V1, XY2, V2):
                        collide(XY1, V1, XY2, V2)

                    # give a slight impulse to help separate them
                    for dim in range(2):
                        V2[dim] += eps * (XY2[dim] - XY1[dim])
                    release(&locks[XY2index]) # Release the lock for the second ball

    release(&locks[i]) # Release the lock corresponding to ball i.  

    """ Original code:
    for j in range(i + 1, count):
        XY2 = &(XY[j, 0])
        V2 = &(V[j, 0])
        if overlapping(XY1, XY2, R):


            # SUBPROBLEM 4: Add locking
            if not moving_apart(XY1, V1, XY2, V2):
                collide(XY1, V1, XY2, V2)

            # give a slight impulse to help separate them
            for dim in range(2):
                V2[dim] += eps * (XY2[dim] - XY1[dim])"""

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
        int chunksize = math.ceil(count/4.) #Add this to divide chunks if #of balls isn't multiple of 4
        unsigned int x_grid, y_grid
        FLOAT *XY1, *XY2, *V1, *V2
        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    with nogil:
        ############## bounce off of walls ################################
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.

        ### 4 Threads
        #for i in prange(count, schedule='static', chunksize=chunksize, num_threads=4):

        ### Single Thread
        for i in range(count):

            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        ############### bounce off of each other ##########################
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling. 

        ### 4 Threads
        #for i in prange(count, schedule='static', chunksize=chunksize, num_threads=4):

        ### Single Thread
        for i in range(count):

            sub_update(XY, V, R, i, count, Grid, grid_spacing, locks)

        ############### update positions ##################################
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        # I do this within the same prange() (or range()) as the update step. 

        ### 4 Threads
        #for i in prange(count, schedule='static', chunksize=chunksize, num_threads=4): 

        ### Single Thread       
        for i in range(count):

            # If the ball is still within the grid, clear its position out of the grid before updating
            if (XY[i,0]>=0 and XY[i,0]<=1) and (XY[i,1]>=0 and XY[i,1]<=1): 
                x_grid = <unsigned int>(XY[i,0] / grid_spacing)
                y_grid = <unsigned int>(XY[i,1] / grid_spacing)
                Grid[x_grid, y_grid] = -1
            # Update position
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
            # If the ball is still within the grid, update its Grid position after the update.  
            if (XY[i,0]>=0 and XY[i,0]<=1) and (XY[i,1]>=0 and XY[i,1]<=1): 
                x_grid = <unsigned int>(XY[i,0] / grid_spacing)
                y_grid = <unsigned int>(XY[i,1] / grid_spacing)
                Grid[x_grid, y_grid] = i
 
def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
