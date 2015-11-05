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
                     float grid_spacing,
                     omp_lock_t *locks) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, dim, x_grid1, y_grid1, ball_index, curr_grid_x, curr_grid_y
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    acquire(&locks[i])
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])

    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    # in order to check the overlapping,
    # we check if grids have the same index

    # first convert XY1 to grid 1
    # and calcualte the grid_size
    x_grid1 = <int>(XY[i,0]/grid_spacing)
    y_grid1 = <int>(XY[i,1]/grid_spacing)
    grid_size = <int>((1.0 / grid_spacing) + 1)


    #then we shall check any grid which is 2 away from this one, it's like 5 * 5
    #and our grid is in the center of this grid
    for curr_grid_x in range(x_grid1 - 2, x_grid1 + 3):

        #we have to check if curr_grid_x is in range [0,1] before we move on
        if (curr_grid_x > 0) and (curr_grid_x < grid_size):
            #now we move y-axis to check
            for curr_grid_y in range(y_grid1-2,y_grid1 + 3):

                #we also have to check if y values still in range
                    if (curr_grid_y > 0) and (curr_grid_y < grid_size):

                        #after all conditions are satisfied
                        #now we check if they have the same index as the original one
                        #if they are overlapping, so they will have the different index, go into the loop
                        #we have to check the grid index here

                        ball_index = Grid[curr_grid_x,curr_grid_y]

                        if(ball_index != -1 and ball_index != i):
                            #acquire(&locks[ball_index])
                            XY2 = &(XY[ball_index,0])
                            V2 = &(V[ball_index,0])

                            ################################
                            #original code, ignore it :)
                            ################################
                            #for j in range(i + 1, count):
                            #    XY2 = &(XY[j, 0])
                            #    V2 = &(V[j, 0])

                            if overlapping(XY1, XY2, R):

                                # SUBPROBLEM 4: Add locking

                                if not moving_apart(XY1, V1, XY2, V2):
                                    collide(XY1, V1, XY2, V2)
                                # give a slight impulse to help separate them
                                for dim in range(2):
                                    V2[dim] += eps * (XY2[dim] - XY1[dim])
                            #release(&locks[ball_index])
    release(&locks[i])


cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t):
    cdef:
        int count = XY.shape[0]
        int i, j, dim,xgrid,ygrid,grid_size
        FLOAT *XY1, *XY2, *V1, *V2
        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2


    # bounce off of walls
    #
    # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
    # scheduling.
    #for i in range(count):
    for i in prange(count, nogil=True, schedule='static', chunksize = count/4, num_threads = 4):
        for dim in range(2):
            if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                V[i, dim] *= -1

    # bounce off of each other
    #
    # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
    # scheduling.
    #for i in range(count):
    for i in prange(count,nogil=True, schedule='static', chunksize = count/4, num_threads = 4):
        sub_update(XY, V, R, i, count, Grid, grid_spacing,locks)

    # update positions
    #
    # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
    #    scheduling).
    # SUBPROBLEM 2: update the grid values.
    #for i in range(count):

    #so at first, all grids should be set to -1
    #then update with the newest location
    Grid[:] = -1


    #grid_size will be used for checking boundary later
    grid_size = <int>((1.0 / grid_spacing) + 1)

    for i in prange(count,nogil=True, schedule='static', chunksize = count/4, num_threads = 4):
        #we use the Grid to label the balls' location
        #the position array get updated
        #then update the grid according to the position array
        for dim in range(2):
            XY[i, dim] += V[i, dim] * t


        #put the new position into grid and check
        xgrid = <int>(XY[i,0] / grid_spacing)
        ygrid = <int>(XY[i,1] / grid_spacing)


        #check if they are out of the boundary
        #only update the grid if they are in the boundary
        #that object shouldn't leave [0,1] square
        #each square stores the index of that ball
        if (xgrid > 0) and (xgrid < grid_size) and (ygrid > 0) and (ygrid < grid_size):
            Grid[xgrid,ygrid] = i


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
