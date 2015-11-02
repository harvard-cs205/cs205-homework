# turn off bounds checking & wraparound for arrays
#cython: boundscheck=False, wraparound=False, cdivision=True

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
        int source, destination, case

   ##########
   # Your code here
   # Use parallel.prange() and a lock for each element of counts to parallelize
   # data movement.  Be sure to avoid deadlock, and double-locking.
   ##########

    with nogil:
        for r in prange(repeat, num_threads=4):
            # Everything in here is parallel...so we have to be very careful. We can easily cause deadlock...
            for idx in range(src.shape[0]):
                # We need to lock here...but we can accidentally do terrible things. We need a lock on both
                # the source and the destination...

                source = src[idx]
                destination = dest[idx]

                # To avoid deadlock, always lock the smaller one first.
                if source < destination: case = -1
                elif source > destination: case=1
                else: case=0

                if case==-1:
                    acquire(&locks[source])
                    acquire(&locks[destination])
                elif case==0:
                    acquire(&locks[source])
                elif case==1:
                    acquire(&locks[destination])
                    acquire(&locks[source])

                # Do the transfer
                if counts[src[idx]] > 0:
                    counts[destination] += 1
                    counts[source] -= 1

                # Now release in the appropriate order
                if case==-1:
                    release(&locks[source])
                    release(&locks[destination])
                elif case==0:
                    release(&locks[source])
                elif case==1:
                    release(&locks[destination])
                    release(&locks[source])

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
        int source, destination, case, source_lock, dest_lock

    ##########
    # Your code here
    # Use parallel.prange() and a lock for every N adjacent elements of counts
    # to parallelize data movement.  Be sure to avoid deadlock, as well as
    # double-locking.
    ##########

    print 'num_locks in pyx', num_locks

    with nogil:
        for r in prange(repeat, num_threads=4):
            for idx in range(src.shape[0]):
                source = src[idx]
                destination = dest[idx]

                source_lock = source/num_locks
                dest_lock = destination/num_locks

                # The last lock will have a couple extra...can't always divide equally with a given number of locks
                if source_lock >= num_locks: # Should never be greater than or equal to num_locks
                    source_lock = num_locks - 1
                if dest_lock >= num_locks:
                    dest_lock = num_locks - 1

                # To avoid deadlock, always grab the smaller indexed lock first
                if source_lock < dest_lock: case = -1
                elif source_lock > dest_lock: case=1
                else: case=0

                if case==-1:
                    acquire(&locks[source_lock])
                    acquire(&locks[dest_lock])
                elif case==0:
                    acquire(&locks[source_lock])
                elif case==1:
                    acquire(&locks[dest_lock])
                    acquire(&locks[source_lock])

                if counts[source] > 0:
                    counts[destination] += 1
                    counts[source] -= 1

                # Now release in the appropriate order
                if case==-1:
                    release(&locks[source_lock])
                    release(&locks[dest_lock])
                elif case==0:
                    release(&locks[source_lock])
                elif case==1:
                    release(&locks[dest_lock])
                    release(&locks[source_lock])

        free_N_locks(num_locks, locks)
