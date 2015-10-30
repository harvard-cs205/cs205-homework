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

############# FINE GRAINED #############
cpdef move_data_fine_grained(np.int32_t[:] counts,
                             np.int32_t[:] src,
                             np.int32_t[:] dest,
                             int repeat):
   cdef:
       int idx, r, src_loc, dest_loc

       int lock_num_src, lock_num_dest, N = 1 # Useless variables
        
       omp_lock_t *locks = get_N_locks(counts.shape[0])

#   print("Number of locks: {}".format(counts.shape[0]))
   for r in range(repeat):
       for idx in prange(src.shape[0], 
                         nogil=True, 
                         num_threads=4):
            src_loc = src[idx]
            dest_loc = dest[idx]

            # Perform a useless computation here to match the computation 
            # of the medium grained version
            lock_num_src = (src_loc / N)
            lock_num_dest = (dest_loc / N)

            # If source and destination are the same, 
            # there's nothing to do.
            if src_loc != dest_loc: 
                # To prevent deadlock, let's lock from the lower index
                # This costs some overhead
                if src_loc < dest_loc:
                    acquire(&(locks[src_loc]))
                    acquire(&(locks[dest_loc]))
                elif src_loc > dest_loc:
                    acquire(&(locks[dest_loc]))
                    acquire(&(locks[src_loc]))
                if counts[src_loc] > 0:
                    counts[src_loc] -= 1
                    counts[dest_loc] += 1
                release(&(locks[src_loc]))
                release(&(locks[dest_loc]))

   free_N_locks(counts.shape[0], locks)

############# MEDIUM GRAINED #############
cpdef move_data_medium_grained(np.int32_t[:] counts,
                               np.int32_t[:] src,
                               np.int32_t[:] dest,
                               int repeat,
                               int N):
   cdef:
       int idx, r, src_loc, dest_loc, lock_num_src, lock_num_dest
       int num_locks = (counts.shape[0] + N - 1) / N  # ensure enough locks
       omp_lock_t *locks = get_N_locks(num_locks)

#   print("Number of locks: {}".format(num_locks))

        
   for r in range(repeat):
       for idx in prange(src.shape[0], 
                         nogil=True, 
                         num_threads=4):
            src_loc = src[idx]
            dest_loc = dest[idx]

            lock_num_src = (src_loc / N)
            lock_num_dest = (dest_loc / N)

            # If source and destination are the same, 
            # there's nothing to do.
            if src_loc != dest_loc: 
                # Lock, starting with the smaller index
                if lock_num_src < lock_num_dest:
                    acquire(&(locks[lock_num_src]))
                    acquire(&(locks[lock_num_dest]))
                elif lock_num_src > lock_num_dest:
                    acquire(&(locks[lock_num_dest]))
                    acquire(&(locks[lock_num_src]))
                else:
                    # Sometimes both source and destination are in the same lock
                    acquire(&(locks[lock_num_src]))
                if counts[src_loc] > 0:
                    counts[src_loc] -= 1
                    counts[dest_loc] += 1
                    
                # Release locks
                if lock_num_src != lock_num_dest:
                    release(&(locks[lock_num_src]))
                    release(&(locks[lock_num_dest]))
                else:
                    release(&(locks[lock_num_src]))

   free_N_locks(num_locks, locks)
