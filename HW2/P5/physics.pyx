#cython: boundscheck=False, wraparound=False

cimport numpy as np
from libc.math cimport sqrt
from libc.stdint cimport uintptr_t
from cython.parallel cimport prange
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
from libc.stdio cimport printf, stdout, fprintf

# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT

cdef UINT negative_one=4294967295

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
        int j, dim
        float eps = 1e-5

    # SUBPROBLEM 4: Add locking
    # GET THE LOCK HERE SO THAT THE TWO THREADS DO NOT ATTEMPT TO WRITE WITHIN THE 
    # SAME MEMORY LOCATION. GIVEN THE THREADING SCHEME, DEADLOCKING CAN BE AVOIDED BY
    # THE COMPARISON DONE BELOW
    acquire(&(locks[i]))

    XY1 = &(XY[i, 0])
    V1 = &(V[i, 0])
    
    #############################################################
    # IMPORTANT: do not collide two balls twice.
    ############################################################
    # SUBPROBLEM 2: use the grid values to reduce the number of other
    # objects to check for collisions.

    cdef int xCenter = <int>(XY[i, 0]/grid_spacing)
    cdef int yCenter = <int>(XY[i, 1]/grid_spacing)
    
    # OBTAIN THE BALLS THAT ARE WITHIN THE GRIDS DEFINED ABOVE
    # THE GRID SPACE IS SUPPOSED TO BE MUCH SMALLER THAN HAVING
    # TO LOOK AT EACH BALL EVERYWERE FOR EVERY COLLISION
    cdef int distCheck = 3
    cdef int xMaxLim = xCenter+distCheck
    cdef int xMinLim = xCenter-distCheck
    cdef int yMaxLim = yCenter+distCheck
    cdef int yMinLim = yCenter-distCheck

    cdef int gridXMax = Grid.shape[0]-1
    cdef int gridYMax = Grid.shape[1]-1

    # CHECK TO SEE IF THE GRID BOUNDARIES EXIT THE DOMAIN BOUNDARIES, 
    # IF THEY DO, BRING THE GRID BOUNDARIES BACK INTO THE DOMAIN
    if xMaxLim>=gridXMax: xMaxLim = gridXMax
    if xMinLim<0: xMinLim=0
    if yMaxLim>=gridYMax: yMaxLim = gridYMax
    if yMinLim<0: yMinLim=0
    
    cdef int ball_grids
    cdef int xx, yy
    
    # NOW INSTEAD OF CHECKING BETWEEEN EACH PARTICLE IN THE ENTIRE DOMAIN
    # CUT DOWN ON ONLY THE PARTICLES THAT ARE IN THE GRIDS NEXT TO THE 
    # CURRENT GRID OF INTEREST. THIS REDUCES THE COMPUTATION FROM ORDER N^2 TO 
    # ORDER N
    for xx in range(xMinLim,xMaxLim+1):
        for yy in range(yMinLim,yMaxLim+1):
            ball_grids= <int> Grid[xx,yy]
            # THE STATEMENT ON THE RIGHT AVOIDS DEADLOCKING AND ENSURES THAT BALLS 
            # ARE DO NOT ACCOUNT FOR COLIDING TWICE, AS IT ONLY ALLOWS IT TO CONTINUE
            # AT A CERTIAN ORDER.
            if ball_grids != negative_one and ball_grids>i: 
                XY2 = &(XY[ball_grids, 0])
                V2 = &(V[ball_grids, 0])
                if overlapping(XY1, XY2, R):
                    # SUBPROBLEM 4: Add locking
                    if not moving_apart(XY1, V1, XY2, V2) and XY2!=XY1:
                        # OBTAIN THE LOCK BEFORE ALTERING ANY OF THE MEMORY
                        acquire(&(locks[ball_grids]))
                        collide(XY1, V1, XY2, V2)
                        release(&(locks[ball_grids]))

                    # give a slight impulse to help separate them
                    for dim in range(2):
                        V2[dim] += eps * (XY2[dim] - XY1[dim])
    release(&(locks[i]))
  
cpdef update(FLOAT[:, ::1] XY,
             FLOAT[:, ::1] V,
             UINT[:, ::1] Grid,
             float R,
             float grid_spacing,
             uintptr_t locks_ptr,
             float t,
             int num_threads):
    cdef:
        int count = XY.shape[0]
        int i, j, dim
        FLOAT *XY1, *XY2, *V1, *V2
        # SUBPROBLEM 4: uncomment this code.
        omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr
    
    cdef int chunksize=count/num_threads

    cdef int oldxCenter, oldyCenter, newxCenter, newyCenter

    assert XY.shape[0] == V.shape[0]
    assert XY.shape[1] == V.shape[1] == 2


    # bounce off of walls
    #
    # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
    # scheduling.
    for i in prange(count,num_threads=num_threads, schedule='static', chunksize=chunksize, nogil=True):
        for dim in range(2):
            if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
                ((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
                V[i, dim] *= -1

    # bounce off of each other
    #
    # SUBPROBLEM 1: parallelize this loop over 4 threads, with static
    # scheduling.
    #for i in range(count):
    for i in prange(count,num_threads=num_threads, schedule='static', chunksize=chunksize, nogil=True):
        sub_update(XY, V, R, i, count, Grid, grid_spacing, locks)

    # update positions
    #
    # SUBPROBLEM 1: parallelize this loop over 4 threads (with static
    #    scheduling).
    # SUBPROBLEM 2: update the grid values.
    for i in prange(count,num_threads=num_threads, schedule='static', chunksize=chunksize, nogil=True):

        #  OBTAININ THE OLD GRID LOCATIONS FOR THE PARTICALES
        oldxCenter = <int>(XY[i, 0]/grid_spacing)
        oldyCenter = <int>(XY[i, 1]/grid_spacing)

        # UPDATING THE PARTICLES LOCATIONS 
        for dim in range(2):
            XY[i, dim] += V[i, dim] * t
            # THIS PART ENSURES THAT THE POSITIONS DO NOT EXIT THE BOUNDARIES OF 
            # THE DOMAIN. THIS WILL ENSURE THAT THE COLISION WITH THE WALL IS 
            # SUCCESSFUL IF THE GRIDS ATTEMPT TO ASSIGN THE POSITION OF THE BALL
            # TO A PART OUTSIDE OF THE DOMAIN.
            if XY[i, dim] < 0:
                    XY[i, dim] = 0
            if XY[i,0] > 1:
                    XY[i, dim] = 1

        # OBTAININGTHE NEW GRID LOCATIONS FOR THE PARTICLES
        newxCenter = <int>(XY[i, 0]/grid_spacing)
        newyCenter = <int>(XY[i, 1]/grid_spacing)

        # IF THE PARTICLE HAS LEFT THE  OLD GRID WE MUST UPDATE SUCH THAT IF THE NEW
        # COORDINATES ARE NOT THE SAME AS THE OLD COORDINATES WE NEED TO UPDATE
        if oldxCenter!=newxCenter or oldyCenter!=newyCenter:
            #UPDATE THE PARTICLE LOCATIONS
            Grid[oldxCenter,oldyCenter]=negative_one
            Grid[newxCenter,newyCenter]=i

def preallocate_locks(num_locks):
    cdef omp_lock_t *locks = get_N_locks(num_locks)
    assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
    return <uintptr_t> <void *> locks
