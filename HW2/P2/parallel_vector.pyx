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

    ##########
    # Your code here
    # Use parallel.prange() and a lock for each element of counts to parallelize
    # data movement.  Be sure to avoid deadlock, and double-locking.
    ##########
    with nogil:
        for r in range(repeat):
            for idx in prange(src.shape[0], num_threads=5):

                # LOCK COUNTS
                # aquire here so locked while checking if source>0
                #prevent double locking
                if src[idx] == dest[idx]:
                    continue

                # make sure you lock in account order to avoid
                # deadlocking
                if src[idx] < dest[idx]:
                    acquire(&locks[src[idx]])
                    acquire(&locks[dest[idx]])
                else:
                    acquire(&locks[dest[idx]])
                    acquire(&locks[src[idx]])

                # ACTION: do transfering of counts here
                if (counts[src[idx]] > 0):

                    counts[dest[idx]] += 1
                    counts[src[idx]] -= 1

                # RELEASE LOCKS
                # release locks in the right order
                if src[idx] < dest[idx]:
                    release(&locks[src[idx]])
                    release(&locks[dest[idx]])
                else:
                    release(&locks[dest[idx]])
                    release(&locks[src[idx]])

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

   ##########
   # Your code here
   # Use parallel.prange() and a lock for every N adjacent elements of counts
   # to parallelize data movement.  Be sure to avoid deadlock, as well as
   # double-locking.
   ##########
   with nogil:
       for r in range(repeat):
           for idx in prange(src.shape[0],num_threads=5):

               # don't do anything if source and dest same
               if src[idx] == dest[idx]:
                   continue

               # LOCK
               # acquire locks in order
               if src[idx]/N < dest[idx]/N:
                   acquire(&locks[src[idx]/N])
                   acquire(&locks[dest[idx]/N])
               elif dest[idx]/N < src[idx]/N:
                   acquire(&locks[dest[idx]/N])
                   acquire(&locks[src[idx]/N])
               # don't double lock
               elif src[idx]/N == dest[idx]/N:
                   acquire(&locks[src[idx]/N])


               # ACTION
               if counts[src[idx]] > 0:
                   counts[dest[idx]] += 1
                   counts[src[idx]] -= 1


               # RELEASE LOCK
               # release in order
               if src[idx]/N < dest[idx]/N:
                   release(&locks[src[idx]/N])
                   release(&locks[dest[idx]/N])
               elif dest[idx]/N < src[idx]/N:
                   release(&locks[dest[idx]/N])
                   release(&locks[src[idx]/N])
               # don't double lock
               elif src[idx]/N == dest[idx]/N:
                   release(&locks[src[idx]/N])

   free_N_locks(num_locks, locks)
