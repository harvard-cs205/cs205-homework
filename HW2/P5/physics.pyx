#cython: boundscheck=False, wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
from openmp cimport omp_lock_t, \
    omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock, omp_get_thread_num
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

#took hilbert curve generation functions from https://en.wikipedia.org/wiki/Hilbert_curve
cdef extern from "hilbert.c":
    int xy2d (int n, int x, int y)

cpdef find_hilbert(int n, 
                UINT[:,::1] XY, 
                int num_balls, 
                UINT[:] order):
    cdef int i
    for i in range(num_balls):
        order[i]=xy2d(n,XY[i,0],XY[i,1])

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
                     int grid_length,
                     uintptr_t locks_ptr) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, k, l, dim 
        UINT search_idx[49], grid_idx1[2]
        int m=0
        float eps = 1e-5
        float grid_idx[2]
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    # SUBPROBLEM 4: Add locking
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.
    
    #The grid spacing is ~.7*ball radius, so another ball can be overlapping if it is lying on a grid edge. In this case, it with extend into a grid spot two spaces away (the distance will be 1.4 grid spaces). If another ball is just touching the original ball with its center aligned with the grid axis, its center will be 2.8 grid spaces away, so its center will be in the grid space 3 spaces away. 
    #Go +/- 3 grid spots in x and y, and store in list search_idx. When we do the search, we'll only call overlapping for elements that are greater than i (have not been parsed yet) and that are not -1 (no ball in that grid element). 

    #initialize search index to be -1, since we don't take any special care for edges. 
        
    for j in range(49):
        search_idx[j]=-1
    #find the grid index of the given position
    grid_idx[0]=XY1[0]/grid_spacing 
    grid_idx[1]=XY1[1]/grid_spacing
    grid_idx1[0]=<UINT> grid_idx[0]
    grid_idx1[1]=<UINT> grid_idx[1]

    #do the search for nearby objects
    for k in range(grid_idx1[0]-3,grid_idx1[0]+4):
        for l in range(grid_idx1[1]-3,grid_idx1[1]+4):
            if k>=0 and l>=0 and k<grid_length and l<grid_length:
                search_idx[m]=Grid[k,l]
                m+=1

    for j in range(49):
        if search_idx[j]>i and search_idx[j]!=-1: #only check real indicies that haven't been searched yet
            # with gil:
            #     print('point {}, nearby point, {}'.format(i,search_idx[j]))
            XY2 = &(XY[search_idx[j], 0])
            V2 = &(V[search_idx[j], 0])
            if overlapping(XY1, XY2, R):
                # SUBPROBLEM 4: Add locking
                omp_set_lock(&(locks[i]))
                omp_set_lock(&(locks[search_idx[j]]))
                if not moving_apart(XY1, V1, XY2, V2):
                    collide(XY1, V1, XY2, V2)
                # give a slight impulse to help separate them
                for dim in range(2):
                    V2[dim] += eps * (XY2[dim] - XY1[dim])
                omp_unset_lock(&(locks[search_idx[j]]))
                omp_unset_lock(&(locks[i]))

cpdef update(FLOAT[:, ::1] XY, 
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t):
    cdef:
        int count = XY.shape[0]
        int i, j, dim, grid_length=Grid.shape[0], numthreads=4
        FLOAT *XY1, *XY2, *V1, *V2, grid_pos[2], grid_pos_old[2]
        # SUBPROBLEM 4: uncomment this code.

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    with nogil:
        # bounce off of walls
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count,num_threads=numthreads,schedule='static',chunksize=count/4):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
        # scheduling.
        for i in prange(count,num_threads=numthreads,schedule='static',chunksize=count/4):
            sub_update(XY, V, R, i, count, Grid, grid_spacing,grid_length,locks_ptr)

        # update positions
        #
        # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
        #    scheduling).
        # SUBPROBLEM 2: update the grid values.

        for i in prange(count,num_threads=numthreads,schedule='static',chunksize=count/4):
            for dim in range(2):
                grid_pos_old[dim]=XY[i,dim]/grid_spacing
                XY[i, dim] += V[i, dim] * t
                grid_pos[dim]=XY[i,dim]/grid_spacing
            Grid[<int> grid_pos_old[0],<int> grid_pos_old[1]]=-1
            Grid[<int> grid_pos[0],<int> grid_pos[1]]=i


def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
