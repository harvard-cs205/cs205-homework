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

   # note: counts begins as a numpy array with index = value
   # note: src and dest represent arrays of random indices of the counts array
   with nogil:
       # execute the following script 100 times (repeat = 100)
       for r in range(repeat):
       	   # for each index in src (and dest; total = 1,000,000)
           for idx in range(src.shape[0]):
	       # check if the value of counts is greater than 0 at the specified src index
               if counts[src[idx]] > 0:
	       	   # if > 0, add 1 to counts src index and subtract 1 to counts dest index, else do nothing
		   # if dest[idx] == src[idx], then effectively nothing happens
                   counts[dest[idx]] += 1
                   counts[src[idx]] -= 1


cpdef move_data_fine_grained(np.int32_t[:] counts,
                             np.int32_t[:] src,
                             np.int32_t[:] dest,
                             int repeat):
   cdef:
       int idx, r, ind1, ind2
       omp_lock_t *locks = get_N_locks(counts.shape[0])

   # &locks is an array of 1000 omp locks, with each lock corresponding to each position in the count array
   # note: counts begins as a numpy array with index = value
   # note: src and dest represent arrays of random indices of the counts array
   
   with nogil:
       # execute the following script 100 times (repeat = 100)
       for r in range(repeat):
           # loop through each index in src (and dest; total = 1,000,000) - use four threads here
           for idx in prange(src.shape[0], num_threads=4):
               # need to order locks to avoid deadlocking
               if src[idx] > dest[idx]:
                   ind1 = dest[idx]
                   ind2 = src[idx]
               else:
                   ind1 = src[idx]
                   ind2 = dest[idx]

               # acquire locks accordingly
               acquire(&locks[ind1])
	       # check for double-locking
               if ind1 != ind2:
                   acquire(&locks[ind2])
		   
               # run the code (need to lock before the conditional to avoid potential negative counts)
               if counts[src[idx]] > 0:
                   counts[dest[idx]] += 1
                   counts[src[idx]] -= 1
		   
               # release the locks
               release(&locks[ind1])
               if ind1 != ind2:
                   release(&locks[ind2])

   free_N_locks(counts.shape[0], locks)


cpdef move_data_medium_grained(np.int32_t[:] counts,
                               np.int32_t[:] src,
                               np.int32_t[:] dest,
                               int repeat,
                               int N):
   cdef:
       int idx, r, ind1, ind2
       int num_locks = (counts.shape[0] + N - 1) / N  # ensure enough locks
       omp_lock_t *locks = get_N_locks(num_locks)

   with nogil:
       # execute the following script 100 times (repeat = 100)
       for r in range(repeat):
           # loop through each index in src (and dest; total = 1,000,000) - use four threads here
           for idx in prange(src.shape[0], num_threads=4):
               # need to order locks to avoid deadlocking
               if src[idx]/N > dest[idx]/N:
                   ind1 = dest[idx]/N
                   ind2 = src[idx]/N
               else:
                   ind1 = src[idx]/N
                   ind2 = dest[idx]/N

               # acquire locks accordingly
               acquire(&locks[ind1])
	       # check for double-locking
               if ind1 != ind2:
                   acquire(&locks[ind2])

               # run the code (need to lock before the conditional to avoid potential negative counts)
               if counts[src[idx]] > 0:
                   counts[dest[idx]] += 1
                   counts[src[idx]] -= 1

               # release the locks
               release(&locks[ind1])
               if ind1 != ind2:
                   release(&locks[ind2])

   free_N_locks(num_locks, locks)
