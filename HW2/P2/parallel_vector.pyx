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


cdef void lock_and_increment(int idx, omp_lock_t *lock, int increment_by, np.int32_t[:] counts) nogil:
    acquire(lock)
    counts[idx] += increment_by
    release(lock)


cpdef move_data_fine_grained(np.int32_t[:] counts,
                             np.int32_t[:] src,
                             np.int32_t[:] dest,
                             int repeat):
    cdef:
        int idx, r, src_idx, dest_idx
        omp_lock_t *locks = get_N_locks(counts.shape[0])
        omp_lock_t *min_lock
        omp_lock_t *max_lock

    ##########s
    # Your code here
    # Use parallel.prange() and a lock for each element of counts to parallelize
    # data movement.  Be sure to avoid deadlock, and double-locking.
    ##########
    with nogil:
        for r in range(repeat):
            for idx in prange(src.shape[0], num_threads=4):

                # checking the locks instead of ids allows this code to
                # work almost unchanged for coarse-grained locking
                min_lock = &locks[src[idx]] if src[idx] < dest[idx] else &locks[dest[idx]]
                max_lock = &locks[src[idx]] if src[idx] > dest[idx]  else &locks[dest[idx]]

                # acquire the min_lock first
                acquire(min_lock)

                # avoid double locking if they're the same lock
                if min_lock != max_lock:
                    acquire(max_lock)
                
                if counts[src[idx]] > 0:
                    counts[src[idx]] -= 1
                    counts[dest[idx]] += 1

                release(min_lock)

                if min_lock != max_lock:
                    release(max_lock)



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
        omp_lock_t *min_lock
        omp_lock_t *max_lock

    ##########
    # Your code here
    # Use parallel.prange() and a lock for every N adjacent elements of counts
    # to parallelize data movement.  Be sure to avoid deadlock, as well as
    # double-locking.
    ##########
    with nogil:
        for r in range(repeat):
            for idx in prange(src.shape[0], num_threads=4):

                # only modification compared to fine-grained locking is to divide indices by N
                # so that adjacent cells map to the proper lock
                min_lock = &locks[src[idx] / N] if src[idx] < dest[idx] else &locks[dest[idx] / N]
                max_lock = &locks[src[idx] / N] if src[idx] > dest[idx]  else &locks[dest[idx] / N]

                # acquire the min_lock first
                acquire(min_lock)

                # avoid double locking if they're the same lock
                if min_lock != max_lock:
                    acquire(max_lock)
                
                if counts[src[idx]] > 0:
                    counts[src[idx]] -= 1
                    counts[dest[idx]] += 1

                release(min_lock)

                if min_lock != max_lock:
                    release(max_lock)

    free_N_locks(num_locks, locks)
