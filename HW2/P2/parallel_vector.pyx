# turn off bounds checking & wraparound for arrays
#cython: boundscheck=False, wraparound=False

##################################################
# setup and helper code
##################################################


from cython.parallel import parallel, prange
from openmp cimport omp_lock_t, \
    omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock, omp_get_thread_num
from libc.stdlib cimport malloc, free

import numpy as np
cimport numpy as np


# lock helper functions
cdef void acquire(omp_lock_t *l) nogil:
    omp_set_lock(l)

cdef void release(omp_lock_t *l) nogil:
    omp_unset_lock(l)

# helper function to fetch and initialize several locks
cdef omp_lock_t *get_N_locks(int N) nogil:
    cdef:
        omp_lock_t *locks = <omp_lock_t *> malloc(N * sizeof(omp_lock_t))
        int idx

    if not locks:
        with gil:
            raise MemoryError()
    for idx in range(N):
        omp_init_lock(&(locks[idx]))

    return locks

cdef void free_N_locks(int N, omp_lock_t *locks) nogil:
    cdef int idx

    for idx in range(N):
        omp_destroy_lock(&(locks[idx]))

    free(<void *> locks)


##################################################
# Your code below
##################################################

cpdef move_data_serial(np.int32_t[:] counts,
                       np.int32_t[:] src,
                       np.int32_t[:] dest,
                       int repeat):
   cdef:
       int idx, r

   assert src.size == dest.size, "Sizes of src and dest arrays must match"
   with nogil:
       for r in range(repeat):
           for idx in range(src.shape[0]):
               if counts[src[idx]] > 0:
                   counts[dest[idx]] += 1
                   counts[src[idx]] -= 1


cpdef move_data_fine_grained(np.int32_t[:] counts,
                             np.int32_t[:] src,
                             np.int32_t[:] dest,
                             int repeat):
    cdef:
        int idx, r
        omp_lock_t *locks = get_N_locks(counts.shape[0])
        int srcIdx, destIdx
 
    ##########
    # Your code here
    # Use parallel.prange() and a lock for each element of counts to parallelize
    # data movement.  Be sure to avoid deadlock, and double-locking.
    ##########
    with nogil:
        for r in prange(repeat, num_threads=4):
            for idx in range(src.shape[0]):
                srcIdx, destIdx = src[idx], dest[idx]
                
                if srcIdx < destIdx:
                    acquire(&(locks[srcIdx]))
                    acquire(&(locks[destIdx]))
                elif srcIdx > destIdx:
                    acquire(&(locks[destIdx]))
                    acquire(&(locks[srcIdx]))
                else:
                    acquire(&(locks[srcIdx]))

                if counts[srcIdx] > 0:
                    counts[destIdx] += 1
                    counts[srcIdx] -= 1

                if srcIdx < destIdx:
                    release(&(locks[srcIdx]))
                    release(&(locks[destIdx]))
                elif srcIdx > destIdx:
                    release(&(locks[destIdx]))
                    release(&(locks[srcIdx]))
                else:
                    release(&(locks[srcIdx]))                
    
    free_N_locks(counts.shape[0], locks)


cpdef move_data_medium_grained(np.int32_t[:] counts,
                               np.int32_t[:] src,
                               np.int32_t[:] dest,
                               int repeat,
                               int N):
    cdef:
        int idx, r
        int num_locks = (counts.shape[0] + N - 1) / N  # ensure enough locks
        omp_lock_t *locks = get_N_locks(num_locks)
        int srcLockIdx, destLockIdx
 
    ##########
    # Your code here
    # Use parallel.prange() and a lock for every N adjacent elements of counts
    # to parallelize data movement.  Be sure to avoid deadlock, as well as
    # double-locking.
    ##########
    with nogil:
        for r in prange(repeat, num_threads=4):
            for idx in range(src.shape[0]):
                srcLockIdx, destLockIdx = src[idx] / N, dest[idx] / N

                if srcLockIdx < destLockIdx:
                    acquire(&(locks[srcLockIdx]))
                    acquire(&(locks[destLockIdx]))
                elif srcLockIdx > destLockIdx:
                    acquire(&(locks[destLockIdx]))
                    acquire(&(locks[srcLockIdx]))
                else:
                    acquire(&(locks[srcLockIdx]))

                if counts[src[idx]] > 0:
                    counts[dest[idx]] += 1
                    counts[src[idx]] -= 1

                if srcLockIdx < destLockIdx:
                    release(&(locks[srcLockIdx]))
                    release(&(locks[destLockIdx]))
                elif srcLockIdx > destLockIdx:
                    release(&(locks[destLockIdx]))
                    release(&(locks[srcLockIdx]))
                else:
                    release(&(locks[srcLockIdx])) 

    free_N_locks(num_locks, locks)
