#cython: boundscheck=True, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, round
from libc.stdint cimport uintptr_t
cimport cython
from omp_defs cimport omp_lock_t, get_N_locks, free_N_locks, acquire, release
from cython.parallel import parallel, prange

# Useful types
ctypedef np.float32_t FLOAT
ctypedef np.uint32_t UINT

cdef inline int overlapping(FLOAT *x1,
							FLOAT *x2,
							float R) nogil:
	cdef:
		float dx = x1[0] - x2[0]
		float dy = x1[1] - x2[1]
	return (dx * dx + dy * dy) < (4 * R * R)


cdef inline int moving_apart(FLOAT *x1, FLOAT *v1,
							 FLOAT *x2, FLOAT *v2) nogil:
	cdef:
		float deltax = x2[0] - x1[0]
		float deltay = x2[1] - x1[1]
		float vrelx = v2[0] - v1[0]
		float vrely = v2[1] - v1[1]
	# check if delta and velocity in same direction
	return (deltax * vrelx + deltay * vrely) > 0.0


cdef inline void collide(FLOAT *x1, FLOAT *v1,
						 FLOAT *x2, FLOAT *v2) nogil:
	cdef:
		float x1_minus_x2[2]
		float v1_minus_v2[2]
		float change_v1[2]
		float len_x1_m_x2, dot_v_x
		int dim
	# https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
	for dim in range(2):
		x1_minus_x2[dim] = x1[dim] - x2[dim]
		v1_minus_v2[dim] = v1[dim] - v2[dim]
	len_x1_m_x2 = x1_minus_x2[0] * x1_minus_x2[0] + x1_minus_x2[1] * x1_minus_x2[1]
	with gil:
		if len_x1_m_x2 == 0:
			print 
			print 'len_x1_m_x2 is zero!'
			print 'x1[0]:',x1[0],' ... x2[0]:',x2[0]
			print 'x1[1]:',x1[1],' ... x2[1]:',x2[1]
			print 'v1[0]:',v1[0],' ... v2[0]:',v2[0]
			print 'v1[1]:',v1[1],' ... v2[1]:',v2[1]
			print 'x1_minus_x2[0]:',x1_minus_x2[0]
			print 'x1_minus_x2[1]:',x1_minus_x2[1]
			print
	dot_v_x = v1_minus_v2[0] * x1_minus_x2[0] + v1_minus_v2[1] * x1_minus_x2[1]
	for dim in range(2):
		change_v1[dim] = (dot_v_x / len_x1_m_x2) * x1_minus_x2[dim]
	for dim in range(2):
		v1[dim] -= change_v1[dim]
		v2[dim] += change_v1[dim]  # conservation of momentum

cdef void sub_update(FLOAT[:, ::1] XY,
					 FLOAT[:, ::1] V,
					 float R,
					 int i, int count,
					 UINT[:, ::1] Grid,
					 float grid_spacing) nogil:
	cdef:
		FLOAT *XY1, *XY2, *V1, *V2
		int j, k, ct, dim, center_x, center_y, offset, gridmax, gridmin, neighbors[48] 
		float eps = 1e-5
		#bool xsafe_low, xsafe_high, ysafe_low, ysafe_high, both_low, both_high, xlow_yhigh, ylow_xhigh

	# SUBPROBLEM 4: Add locking
	#with gil:
	#	print 'check 0'

	XY1 = &(XY[i, 0])
	V1  = &( V[i, 0])

	# get neighbors on grid
	gridmax = Grid.shape[0]
	gridmin = 0
	offset = 3
	neighborhood_size = offset*2 + 1 # size of grid for checking up to 3 cells away in any direction
	#with gil:
	#	print 'check 1'
	center_x = <UINT> round(XY[i,0]/grid_spacing)
	center_y = <UINT> round(XY[i,1]/grid_spacing)
	#with gil:
	#	print center_x
	#	print center_y
	#	print

	ct = 0
	for j in range(-offset,offset+1):
		for k in range(-offset,offset+1):
			if (j!=0) | (k!=0):
				if ( center_x+j < gridmin ) | ( center_x+j >= gridmax ) | ( center_y+k < gridmin ) | ( center_y+k >= gridmax ):
					neighbors[ct] = 999
				else:
					neighbors[ct] = Grid[center_x+j,center_y+k]
					if neighbors[ct] == -1:
						neighbors[ct] = 999
				ct += 1
	

	#############################################################
	# IMPORTANT: do not collide two balls twice.
	############################################################
	# SUBPROBLEM 2: use the grid values to reduce the number of other
	# objects to check for collisions.

	#with gil:
	#	print "neighbors:"
	#	print np.array(neighbors)
	#	print
	#with gil:
	#	print 'check 2'
	for neighbor in neighbors:
		if (neighbor < 999) & (neighbor != i):
			XY2 = &(XY[neighbor, 0])
			V2  = &( V[neighbor, 0])
			if overlapping(XY1, XY2, R):
				# SUBPROBLEM 4: Add locking
				if not moving_apart(XY1, V1, XY2, V2):
					collide(XY1, V1, XY2, V2)

				# give a slight impulse to help separate them
				for dim in range(2):
					V2[dim] += eps * (XY2[dim] - XY1[dim])

	#with gil:
	#	print 'check 3'
cpdef update(FLOAT[:, ::1] XY,
			 FLOAT[:, ::1] V,
			 UINT[:, ::1] Grid,
			 float R,
			 float grid_spacing,
			 uintptr_t locks_ptr,
			 float t,
			 int nthread,
			 int chunk):
	cdef:
		int count = XY.shape[0]
		int i, j, dim, coord[2], old_coord[2]
		FLOAT *XY1, *XY2, *V1, *V2
		# SUBPROBLEM 4: uncomment this code.
		# omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

	assert XY.shape[0] == V.shape[0]
	assert XY.shape[1] == V.shape[1] == 2

	#print "Grid info:"
	##print "Grid shape:",Grid.shape
	#print "np.array(Grid) shape:", np.array(Grid).shape
	#print "Grid contents:"
	#print np.array(Grid)
	#return "done."

	with nogil:
		# bounce off of walls
		#
		# SUBPROBLEM 1: parallelize this loop over 4 threads, with static
		# scheduling.
		for i in range(count):
			for dim in range(2):
				if (((XY[i, dim] < R) and (V[i, dim] < 0)) or
					((XY[i, dim] > 1.0 - R) and (V[i, dim] > 0))):
					V[i, dim] *= -1

		# bounce off of each other
		#
		# SUBPROBLEM 1: parallelize this loop over 4 threads, with static
		# scheduling.
		for i in prange(count, num_threads=nthread, schedule='static', chunksize=chunk):
			sub_update(XY, V, R, i, count, Grid, grid_spacing)

		# update positions
		#
		# SUBPROBLEM 1: parallelize this loop over 4 threads (with static
		#    scheduling).
		# SUBPROBLEM 2: update the grid values.
		for i in range(count):       
			
			for dim in range(2):
				old_coord[dim] = <UINT> round(XY[i,dim]/grid_spacing)
				XY[i, dim] += V[i, dim] * t
				coord[dim] = <UINT> round(XY[i,dim]/grid_spacing)
				if (coord[dim] < 0): # keeps from falling off top or left of grid
					coord[dim] = 0
				if (coord[dim] >= Grid.shape[0]): # keeps from falling off bottom or right of grid
					coord[dim] = Grid.shape[0] - 1
			Grid[ coord[0],coord[1] ] = i
			# figure out where the ball just was and set that cell to -1.  otherwise weird shit happens.
		with gil:
			print
			print "Cells with balls:", np.sum( ((np.array(Grid) < 501).all() & (np.array(Grid) > -1).all()) )
			print
			print np.array(Grid)


def preallocate_locks(num_locks):
	cdef omp_lock_t *locks = get_N_locks(num_locks)
	assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
	return <uintptr_t> <void *> locks
