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
        int idx, r, si, di, first_lock, second_lock, same_lock
        omp_lock_t *locks = get_N_locks(counts.shape[0])

   ##########
   # Your code here
   # Use parallel.prange() and a lock for each element of counts to parallelize
   # data movement.  Be sure to avoid deadlock, and double-locking.
   ##########
    with nogil:
        for r in range(repeat):
            for idx in prange(src.shape[0], num_threads=4):

                # Use new variables just to clear up the syntax
                si = src[idx]
                di = dest[idx]

                # Boolean value to let us know when si = di
                same_lock = 0

                # We want to always lock in order of smallest index
                if si < di:
                    first_lock = si
                    second_lock = di
                elif di < si:
                    first_lock = di
                    second_lock = si
                else:
                    # And if they're the same, we only lock once
                    same_lock = 1
                    first_lock = si

                if counts[src[idx]] > 0:
                    # Get our locks to modify the data
                    acquire(&locks[first_lock])
                    if not same_lock:
                        acquire(&locks[second_lock])

                    # Increment and decrement accordingly
                    counts[dest[idx]] += 1
                    counts[src[idx]] -= 1

                    # Release locks, depending on whether we locked once or twice
                    # In the same order, but it shouldn't actually matter at this point
                    if not same_lock:
                        release(&locks[second_lock])
                        release(&locks[first_lock])
                    else:
                        release(&locks[first_lock])
 
    free_N_locks(counts.shape[0], locks)


cpdef move_data_medium_grained(np.int32_t[:] counts,
        np.int32_t[:] src,
        np.int32_t[:] dest,
        int repeat,
        int N):
    cdef:
        int idx, r, si, di, first_lock, second_lock, same_lock
        int num_locks = (counts.shape[0] + N - 1) / N  # ensure enough locks
        omp_lock_t *locks = get_N_locks(num_locks)

   ##########
   # Your code here
   # Use parallel.prange() and a lock for every N adjacent elements of counts
   # to parallelize data movement.  Be sure to avoid deadlock, as well as
   # double-locking.
   ##########
    with nogil:
        for r in range(repeat):
            for idx in prange(src.shape[0], num_threads=4):

                # Same code as above, except now we wrap our indices around
                si = src[idx] % num_locks
                di = dest[idx] % num_locks

                # Lock smallest first, and only lock once if theyre the same
                same_lock = 0
                if si < di:
                    first_lock = si
                    second_lock = di
                elif di < si:
                    first_lock = di
                    second_lock = si
                else:
                    first_lock = si
                    same_lock = 1

                if counts[src[idx]] > 0:

                   # Get our locks
                    acquire(&locks[first_lock])
                    if not same_lock:
                        acquire(&locks[second_lock])

                   # Increment/decrement accordingly
                    counts[dest[idx]] += 1
                    counts[src[idx]] -= 1

                   # And now release the beast from its cage
                    if not same_lock:
                        release(&locks[second_lock])
                        release(&locks[first_lock])
                    else:
                        release(&locks[first_lock])

    free_N_locks(num_locks, locks)
