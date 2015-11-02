# turn off bounds checking & wraparound for arrays
# cython: boundscheck=False, wraparound=False

######################
#
# Submission by Kendrick Lo (Harvard ID: 70984997) for
# CS 205 - Computing Foundations for Computational Science (Prof. R. Jones)
# 
# Homework 2 - Problem 2
#
######################

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
        int first_to_lock, second_to_lock  # to implement ordered locking
        omp_lock_t *locks = get_N_locks(counts.shape[0])

    ##########
    # Your code here
    # Use parallel.prange() and a lock for each element of counts to
    # parallelize
    # data movement.  Be sure to avoid deadlock, and double-locking.
    ##########

    with nogil:

        # with gil:
        #    print "entering fine grained"
       
        for r in range(repeat):
            for idx in prange(src.shape[0], nogil=False, num_threads=4):

                #####
                #
                # Lock-ordering

                # We define a total ordering on the set of objects eligible
                # for locking and use this ordering to choose the sequence of
                # lock acquisition. In our case, src[idx] and dest[idx]
                # represent elements of our counts array; we always lock a
                # lower index of the counts array before a higher index of the
                # counts array, otherwise we may encounter deadlock.
                #
                # Note we also add a check to ensure that we do not lock (or
                # do anything) if the src and dest indices are the same.

                if src[idx]!=dest[idx]:

                    if src[idx]<dest[idx]:
                        first_to_lock = src[idx]
                        second_to_lock = dest[idx]
                    else:
                        first_to_lock = dest[idx]
                        second_to_lock = src[idx]
                
                    omp_set_lock(&(locks[first_to_lock]))
                    omp_set_lock(&(locks[second_to_lock]))
               
                    ### critical section, increment and decrement counters
                    if counts[src[idx]] > 0:
                        counts[src[idx]] -= 1
                        counts[dest[idx]] += 1
                    ##### #####

                    # remove locks in reverse order
                    omp_unset_lock(&(locks[second_to_lock]))
                    omp_unset_lock(&(locks[first_to_lock]))
                
    free_N_locks(counts.shape[0], locks)


cpdef move_data_medium_grained(np.int32_t[:] counts,
                               np.int32_t[:] src,
                               np.int32_t[:] dest,
                               int repeat,
                               int N):
    cdef:
        int idx, r
        int num_locks = (counts.shape[0] + N - 1) / N  # ensure enough locks
        int lock_section1, lock_section2  # to implement ordered locking
        int first_to_lock, second_to_lock  # to implement ordered locking
        omp_lock_t *locks = get_N_locks(num_locks)

    ##########
    # Your code here
    # Use parallel.prange() and a lock for every N adjacent elements of counts
    # to parallelize data movement.  Be sure to avoid deadlock, as well as
    # double-locking.
    ##########

    with nogil:

        # with gil:
        #    print "entering medium grained"

        for r in range(repeat):
            for idx in prange(src.shape[0], nogil=False, num_threads=4):

                #####
                #
                # Lock-ordering

                # We define a total ordering on the set of objects eligible
                # for locking and use this ordering to choose the sequence of
                # lock acquisition. 
                #
                # In this section, instead of each element of the counts
                # array having a lock, each set of (elements div N) is 
                # assigned the same lock. To figure out which lock to use, 
                # we divide a src or dest index by N (i.e., counts[i] uses
                # locks[i / N]).
                # 
                # We always lock a lower indexed section of the counts array
                # before a higher indexed section of the counts array,
                # otherwise we may encounter deadlock.
                #
                # Note we also add a check to ensure that we do not lock (or
                # do anything) if the src and dest indices are the same.

                if src[idx]!=dest[idx]:

                    # identify sections to be locked
                    lock_section1 = src[idx] / N
                    lock_section2 = dest[idx] / N

                    if lock_section1<lock_section2:
                        first_to_lock = lock_section1
                        second_to_lock = lock_section2
                    elif lock_section1>lock_section2:
                        first_to_lock = lock_section2
                        second_to_lock = lock_section1
                    else:
                        # both src and dest are in the same section
                        first_to_lock = lock_section1
                        second_to_lock = -1
                    
                    # lock sections
                    omp_set_lock(&(locks[first_to_lock]))
                    if second_to_lock>=0:
                        # only use second lock if locking different section
                        omp_set_lock(&(locks[second_to_lock]))

                    ### critical section, increment and decrement counters
                    if counts[src[idx]] > 0:
                        counts[dest[idx]] += 1
                        counts[src[idx]] -= 1
                    ##### #####

                    # remove locks in reverse order
                    if second_to_lock>=0:
                        # do not unlock if second lock not employed
                        omp_unset_lock(&(locks[second_to_lock]))
                    omp_unset_lock(&(locks[first_to_lock]))

    free_N_locks(num_locks, locks)
