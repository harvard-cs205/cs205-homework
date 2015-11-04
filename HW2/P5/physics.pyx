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

    for dim in range(2):
        x1_minus_x2[dim] = x1[dim] - x2[dim]
        v1_minus_v2[dim] = v1[dim] - v2[dim]
    len_x1_m_x2 = x1_minus_x2[0] * x1_minus_x2[0] + x1_minus_x2[1] * x1_minus_x2[1]
    dot_v_x = v1_minus_v2[0] * x1_minus_x2[0] + v1_minus_v2[1] * x1_minus_x2[1]
    for dim in range(2):
        change_v1[dim] = (dot_v_x / len_x1_m_x2) * x1_minus_x2[dim]
    for dim in range(2):
        v1[dim] -= change_v1[dim]
        v2[dim] += change_v1[dim]  
        
cdef void sub_update(FLOAT[:, ::1] XY,
                     FLOAT[:, ::1] V,
                     float R,
                     int i, int count,
                     UINT[:, ::1] Grid,
                     omp_lock_t *locks,
                     float grid_spacing) nogil:
    cdef:
        FLOAT *XY1, *XY2, *V1, *V2
        int j, dim, xx, yy
        unsigned int grid_x, grid_y, grid_xnew, grid_ynew
        float eps = 1e-5
        
    # SUBPROBLEM 4: Add locking
   
    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
            
    # SUBPROBLEM 2:
    grid_x = <unsigned int>(XY[i,0]/grid_spacing)
    grid_y = <unsigned int>(XY[i,1]/grid_spacing)    

    #account for grid positions up two 2 squares away
    for xx in range(-2,3):
        for yy in range(-2,3):
            grid_xnew = grid_x+xx
            grid_ynew = grid_y+yy
            
            #make sure no outlying points are used (e.g., withing bounds of problem, not comparing the same position to itself)
            if (grid_xnew > 0) & (grid_xnew < <unsigned int>(1/grid_spacing) + 1) & (grid_ynew > 0) & (grid_ynew < <unsigned int>(1/grid_spacing) + 1) & (xx!=0) & (yy!=0):
                if (Grid[grid_x + xx,grid_y + yy] < count) & (Grid[grid_x + xx,grid_y + yy] > 0) & (xx!=0) & (yy!=0):

                    j = Grid[grid_xnew,grid_ynew]
                    
                    XY2 = &(XY[j, 0])
                    V2 = &(V[j, 0])
                    # SUBPROBLEM 4: Add locking
                    
                    if (XY1 != XY2) & (overlapping(XY1, XY2, R)):    
                    
                    #add locks before we call collide.
                    #ensure we lock/release the smaller of the two positions
                    #to avoid issues
                    
                        if i > j:   
                            acquire(&(locks[i]))
                            acquire(&(locks[j]))
                        else:
                            acquire(&(locks[j]))
                            acquire(&(locks[i]))  
   
                         if not moving_apart(XY1, V1, XY2, V2):
                                                    
                            collide(XY1, V1, XY2, V2)
                            
                        for dim in range(2):
                                                     
                            V2[dim] += eps * (XY2[dim] - XY1[dim])
                    
                        if i > j:   
                            release(&(locks[j]))
                            release(&(locks[i]))
                        else:
                            release(&(locks[i]))
                            release(&(locks[j]))             
                    
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
        unsigned int grid_x, grid_y, grid_xnew, grid_ynew
        
        FLOAT *XY1, *XY2, *V1, *V2
        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2

    with nogil:
        # bounce off of walls
        for i in prange(count,num_threads=4,schedule='static',chunksize=count/4):
            for dim in range(2):
                if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                    ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                    V[i, dim] *= -1

        # bounce off of each other
        for i in prange(count,num_threads=4,schedule='static',chunksize=count/4):
            sub_update(XY, V, R, i, count, Grid, locks, grid_spacing)

        # update positions
        # SUBPROBLEM 2: 
        for i in prange(count, num_threads=4, schedule='static', chunksize=count/4):
            
            #reset initial grid position value to -1 when ball i exits
            #grid position determiend by taking XY coordinates and
            #dividing by the grid spacing
            #make sure to use unsigned int for grid_x/y to convert from float
            
            if (XY[i,0] > 0) & (XY[i,0] < 1) & (XY[i,1] > 0) & (XY[i,1] < 1):
                grid_x = <unsigned int>(XY[i,0]/grid_spacing)
                grid_y = <unsigned int>(XY[i,1]/grid_spacing)   
                Grid[grid_x,grid_y] = -1
         
            for dim in range(2):
                XY[i, dim] += V[i, dim] * t
 
             #update ball i's new grid position to grid_xnew,grid_ynew
             #make sure to use unsigned int for grid_x/y to convert from float           
            if (XY[i,0] > 0) & (XY[i,0] < 1) & (XY[i,1] > 0) & (XY[i,1] < 1):
                grid_xnew = <unsigned int>(XY[i,0]/grid_spacing)
                grid_ynew = <unsigned int>(XY[i,1]/grid_spacing)
                Grid[grid_xnew, grid_ynew] = i

def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
