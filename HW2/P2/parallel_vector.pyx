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

	print('Number of locks for fine-grained: {}'.format(counts.shape[0])) 


	for r in range(repeat):

		for idx in prange(src.shape[0], nogil = True , num_threads = 4 ):


			#--------------------------------------
			# need to separate to avoid deadlocking
			# ex: src[idx] = 5 and dest[idx] = 6 on thread 1
			# while src[idx] = 6 and dest[idx] = 5 on thread 2
			#--------------------------------------
			if src[idx] < dest[idx]:
           	# grab the lock 
				acquire(&(locks[src[idx]])) 
				acquire(&(locks[dest[idx]])) 
           	# move data 
				if counts[src[idx]] > 0:
					counts[dest[idx]] += 1 
					counts[src[idx]] -= 1    
           	# release the lock 
				release(&(locks[src[idx]]))
				release(&(locks[dest[idx]]))    


			elif src[idx] > dest[idx]:
           	# grab the lock
				acquire(&(locks[dest[idx]]))            
				acquire(&(locks[src[idx]])) 
           	# move data 
				if counts[src[idx]] > 0:
					counts[dest[idx]] += 1 
					counts[src[idx]] -= 1    
           	# release the lock 
				release(&(locks[dest[idx]]))    
				release(&(locks[src[idx]]))


			#--------------------------------------
            # here is to avoid double locking
            #--------------------------------------
			else:
				acquire(&(locks[src[idx]])) 
           	# move data
				if counts[src[idx]] > 0:
					counts[dest[idx]] += 1
					counts[src[idx]] -= 1
           	# release the lock
				release(&(locks[src[idx]]))

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
		# also add new integers
		int lock_src, lock_dest

   ##########
   # Your code here
   # Use parallel.prange() and a lock for every N adjacent elements of counts
   # to parallelize data movement.  Be sure to avoid deadlock, as well as
   # double-locking.
   ##########
	print('Number of locks for medium-grained: {}'.format(num_locks)) 

	for r in range(repeat):
		for idx in prange(src.shape[0], nogil= True, num_threads = 4):

			# Each (entries / num_locks) share a lock
			# that is an integer from 0 to 99
			lock_src = src[idx] / num_locks		
			lock_dest = dest[idx] / num_locks


			#--------------------------------------
			# same as above - avoid deadlock
			#--------------------------------------

			if lock_src < lock_dest:

				acquire( &locks[lock_src])
				acquire( &locks[lock_dest])

				if counts[src[idx]] > 0:
					counts[dest[idx]] += 1
					counts[src[idx]] -= 1

				release( &locks[lock_src])
				release( &locks[lock_dest])

			elif lock_dest < lock_src:

				acquire( &locks[lock_dest])
				acquire( &locks[lock_src])

				if counts[src[idx]] > 0:
					counts[dest[idx]] += 1
					counts[src[idx]] -= 1

				release( &locks[lock_dest])
				release( &locks[lock_src])		

			else:

				acquire( &locks[lock_src])

				if counts[src[idx]] > 0:
					counts[dest[idx]] += 1
					counts[src[idx]] -= 1

				release( &locks[lock_src])			

	free_N_locks(num_locks, locks)




