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
       int idx, r, min_idx, max_idx
       omp_lock_t *locks = get_N_locks(counts.shape[0])

   ##########
   # Your code here
   # Use parallel.prange() and a lock for each element of counts to parallelize
   # data movement.  Be sure to avoid deadlock, and double-locking.
   ##########
   for r in range(repeat):
       for idx in prange(src.shape[0], nogil=True, schedule=dynamic, num_threads=4):
           #  Corner case.  This doesn't do anything to the counts, but
           #  will cause a double lock.
           if src[idx] == dest[idx] :
               continue

           # Prevent deadlock by acquiring both potential locks
           # in order of index.
           if src[idx] < dest[idx] :
               min_idx = src[idx]
               max_idx = dest[idx]
           else :
               min_idx = dest[idx]
               max_idx = src[idx]

           acquire(&(locks[min_idx]))
           acquire(&(locks[max_idx]))

           if counts[src[idx]] > 0:
               counts[dest[idx]] += 1
               counts[src[idx]] -= 1

           # Release locks.
           release(&(locks[max_idx]))
           release(&(locks[min_idx]))

   free_N_locks(counts.shape[0], locks)


cpdef move_data_medium_grained(np.int32_t[:] counts,
                               np.int32_t[:] src,
                               np.int32_t[:] dest,
                               int repeat,
                               int N):
   cdef:
       int idx, r, l1, l2
       int num_locks = (counts.shape[0] + N - 1) / N  # ensure enough locks
       omp_lock_t *locks = get_N_locks(num_locks)

   ##########
   # Your code here
   # Use parallel.prange() and a lock for every N adjacent elements of counts
   # to parallelize data movement.  Be sure to avoid deadlock, as well as
   # double-locking.
   ##########
   for r in range(repeat):
       for idx in prange(src.shape[0], nogil=True, schedule=dynamic, num_threads=4):
           
           # Shortcut this case since this effectively is a no-op
           if src[idx] == dest[idx] :
               continue

           # Prevent deadlock by acquiring both potential locks
           # in order of index size
           if src[idx] < dest[idx] :
               l1 = src[idx] / N
               l2 = dest[idx] / N
           else :
               l1 = dest[idx] / N
               l2 = src[idx] / N

           acquire(&(locks[l1]))
           # Prevent double lock
           if l1 != l2 :
               acquire(&(locks[l2])) 
           if counts[src[idx]] > 0:
               counts[dest[idx]] += 1
               counts[src[idx]] -= 1
           # Prevent double release
           if l1 != l2 :
               release(&(locks[l2])) 
           release(&(locks[l1])) 

   free_N_locks(num_locks, locks)