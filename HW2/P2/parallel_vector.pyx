# Note: includes all my comments from reviewing the skeleton code

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

# NOTE: cpdef functions can be called from both Python and C

cpdef move_data_serial(np.int32_t[:] counts,
                       np.int32_t[:] src,
                       np.int32_t[:] dest,
                       int repeat):

   # NOTE: cdef makes C-level declarations

    cdef:
       int idx, r

    assert src.size == dest.size, "Sizes of src and dest arrays must match"
    with nogil:
        for r in range(repeat): # Repeat 100 times
            for idx in range(src.shape[0]): # Repeat 1 mil. times
                if counts[src[idx]] > 0: # i.e. avoid negative numbers
                    counts[dest[idx]] += 1 # Increase one value by 1
                    counts[src[idx]] -= 1 # Reduce another value by 1


cpdef move_data_fine_grained(np.int32_t[:] counts,
                             np.int32_t[:] src,
                             np.int32_t[:] dest,
                             int repeat):
    cdef:
        int idx, r, sum = 0
        omp_lock_t *locks = get_N_locks(counts.shape[0]) # Creates 1,000 locks (1 per index in counts array)

    for r in range(repeat):

        for idx in prange(src.shape[0], nogil=True, num_threads=4):

            if counts[src[idx]] > 0:

                acquire(&locks[dest[idx]]) # Acquire lock for index from dest array
                counts[dest[idx]] += 1 # Update value
                release(&locks[dest[idx]]) # Release lock

                acquire(&locks[src[idx]]) # Acquire lock for index from src array
                counts[src[idx]] -= 1 # Update value
                release(&locks[src[idx]]) # Release lock

    free_N_locks(counts.shape[0], locks) # Frees all locks


cpdef move_data_medium_grained(np.int32_t[:] counts,
                               np.int32_t[:] src,
                               np.int32_t[:] dest,
                               int repeat,
                               int N):
    cdef:
        int idx, r
        int num_locks = (counts.shape[0] + N - 1) / N  # Ensure enough locks
        omp_lock_t *locks = get_N_locks(num_locks) # Creates all locks

    for r in range(repeat):

        for idx in prange(src.shape[0], nogil=True, num_threads=4):

            if counts[src[idx]] > 0:

                acquire(&locks[dest[idx]/N]) # Acquire lock for index from dest array
                counts[dest[idx]] += 1 # Update value
                release(&locks[dest[idx]/N]) # Release lock

                acquire(&locks[src[idx]/N]) # Acquire lock for index from src array
                counts[src[idx]] -= 1 # Update value
                release(&locks[src[idx]/N]) # Release lock

    free_N_locks(num_locks, locks) # Frees all locks
