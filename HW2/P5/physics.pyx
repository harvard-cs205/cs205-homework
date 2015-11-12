# cython: boundscheck=True, wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
from cython.parallel import parallel, prange
from libc.stdio cimport printf

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


#A simple function I wrote to confirm that I am in the grid. 

cdef inline int inGrid(FLOAT[:, ::1] XY, int i) nogil:
    
    return (XY[i, 0] <= 1 and XY[i, 0] >=0 and XY[i, 1] <= 1 and XY[i, 1] >= 0)


#The same functions from the Wiki Page, included here, but not used. 

cpdef inline int xy2d(int n, int x, int y): 
    cdef:
        int rx, ry, s, d
    d = 0
    s = n/2
    #for (s=n/2; s>0; s/=2):
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        rot(s, &x, &y, rx, ry)
        s /= 2
    return d


cdef inline void rot(int n, int *x, int *y, int rx, int ry) nogil:
    cdef:
        int t
    if (ry == 0):
        if (rx == 1):
            x[0] = n-1 - x[0]
            y[0] = n-1 - y[0]

        t  = x[0]
        x[0] = y[0]
        y[0] = t



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
                     float grid_spacing, omp_lock_t *locks) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, x1, y1, g_x, g_y, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound, theMax, dim
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    #set didn't work, turns out we only import acquire
    #omp_set_lock(i)

    #Get the lock for the current "ith" ball

    acquire(&locks[i])

    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    
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

    #Get the grid x value and the grid y value using the same technique
    #we used in update. 

    g_x = int(XY[i, 0]/grid_spacing)
    g_y = int(XY[i, 1]/grid_spacing)

    #Figure out what the bounds of the grid are. Was experiencing problems
    #earlier when using grid_size, then saw the piazza post about the minor bug.

    theMax = int((1.0/grid_spacing) + 1)

    #Only explore things to the "right" or "below" you to not double-count. 
    #So, we start at our current index, check 2 squares to the right of it 
    #and also modify our y values so the further "right" in the grid, we don't
    #go as far down. This makes sense if you draw out a grid and a circle with a 
    #particular radius in a square and think about all the squares that it could
    #potentially be neighbors with. 

    x_lower_bound = g_x 
    x_upper_bound = g_x + 3
    y_lower_bound = g_y


    for x1 in range(x_lower_bound, x_upper_bound):
        for y1 in range(y_lower_bound, g_y + 3 - (x1 - x_lower_bound)):

            #make sure that the x1 and y1 are not outside the grid, 
            #are no the exact same thing as the ball you were already looking at.
            #Also, make sure that the values are within the # of balls we actually have.
            #I have no idea how this wouldn't be true, but I was getting a seg-fault before
            #I added this condition. 

            if x1 < theMax and y1 < theMax and not(x1 == g_x and y1 == g_y) and Grid[x1, y1] < count:
                j = Grid[x1, y1]
                XY2 = &(XY[j, 0])
                acquire(&locks[j])
                V2 = &(V[j, 0])
                if overlapping(XY1, XY2, R):
                    if not moving_apart(XY1, V1, XY2, V2):
                        collide(XY1, V1, XY2, V2)
                    for dim in range(2):
                        V2[dim] += eps * (XY2[dim] - XY1[dim])
                release(&locks[j])
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
        int i, j, dim, g_x, g_y, g_x_orig, g_y_orig
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
        for i in prange(count, num_threads=4, schedule='static', chunksize=count/4):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count, num_threads=4, schedule='static', chunksize=count/4):
            sub_update(XY, V, R, i, count, Grid, grid_spacing, locks)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.
        
        # for a in xrange(Grid.shape[0]):
        #     for b in xrange(Grid.shape[0]):
        #         Grid[a, b] = -1

        #Parallelize the code below and make sure that you are
        #first initializing the values in the grid to -1 (since you are updating the grid
        #with all new values). Then go through, and for each value i that is in the grid
        #bounds, add it to the grid by putting it in the appropriate "bucket" by dividing
        #by grid_spacing. 

        for i in prange(count, num_threads=4, schedule='static', chunksize=count/4):
            if inGrid(XY, i):
                g_x_orig = int(XY[i, 0]/grid_spacing)
                g_y_orig = int(XY[i, 1]/grid_spacing)
                Grid[g_x_orig, g_y_orig] = -1

            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
                
            if inGrid(XY, i):
                g_x = int(XY[i, 0]/grid_spacing) 
                g_y = int(XY[i, 1]/grid_spacing)

                Grid[g_x, g_y] = i



def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
