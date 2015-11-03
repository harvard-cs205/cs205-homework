#cython: boundscheck=True, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, round
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc
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
		int j, k, ct, dim, center_x, center_y, offset, gridmax, gridmin, neighbors[48] # boo hard-coding!
		float eps = 1e-5

	# SUBPROBLEM 4: Add locking

	XY1 = &(XY[i, 0])
	V1  = &( V[i, 0])

	'''
	SUBPROBLEM 2: use the grid values to reduce the number of other objects to check for collisions.
	
	I defined a neighbors array with a slot for each neighbor in the 7x7 area we need to cover (see Piazza @642).
	I used the center_x and center_y variables to define the Grid coordinates of the current ball.
	Then I looped through all of the neighbor coordinates, and recorded the ball number in each one, if there was one.
	Skip down to the next block comment for more.
	'''
	# get neighbors on grid
	gridmax = Grid.shape[0]
	gridmin = 0
	offset = 3
	neighborhood_size = offset*2 + 1 # size of grid for checking up to 3 cells away in any direction

	center_x = <UINT> round(XY[i,0]/grid_spacing)
	center_y = <UINT> round(XY[i,1]/grid_spacing)

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

	''' Now that we have all of the neighborhood ball IDs, we can check for overlapping, collisions, etc.
		Here we loop over each ball and check (a) that this neighbor cell actually has a ball in it, and
		(b) that it's actually a neighbor, and not the current ball itself.

		This allows us to reduce our search space from i+1:num_balls, down to the 7x7 grid of neighbors who 
		could conceivably be colliding or overlapping. 
		This effectively reduces the algorithm runtime from O(N^2) (a full nested for-loop), 
		down to a linear multiple of N.
	'''
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
		int i, j, dim, coord[2]
		np.int64_t[:,:] old_coord = np.zeros((XY.shape[0],2), dtype=int)
		FLOAT *XY1, *XY2, *V1, *V2
		# SUBPROBLEM 4: uncomment this code.
		# omp_lock_t *locks = <omp_lock_t *> <void *> locks_ptr

	assert XY.shape[0] == V.shape[0]
	assert XY.shape[1] == V.shape[1] == 2

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
			
			''' Updating the Grid:
				We have three main tasks here:
					(1) Record the new grid position of each ball, based on its center, after its V*t update
					(2) Remove the record of the ball in its previous grid position
					(3) Ensure that the conversion from XY position to grid coordinates doesn't go out of bounds

				We round the normalized XY position to the nearest integer by importing round() from libc.math. 
					Note: This round does not behave in exactly the same way as the .astype(int) method from 
					Numpy which we are attempting to emulate (this is how we round/type-convert to int in driver.py). 
					The .astype(int) method rounds down halves (eg. 2.5 -> 2 and 3.5 -> 3), but libc.math round() 
					"rounds halfway cases away from zero" (http://goo.gl/AucW1I).

				The old_coord (num_balls x 2) array keeps track of a ball's grid coordinates from the most recent
				update. If a ball's center changes to a new grid cell, the old coordinates are marked as empty.

				In thoses cases where a ball's center strays outside the grid boundary, we adjust its center to 
				the nearest legal cell (eg. (-1,0) -> (0,0)).
			'''
			for dim in range(2):
				old_coord[i][dim] = <UINT> round(XY[i,dim]/grid_spacing)
				if old_coord[i][dim] >= Grid.shape[0]:
					old_coord[i][dim] = Grid.shape[0] - 1
				XY[i, dim] += V[i, dim] * t
				coord[dim] = <UINT> round(XY[i,dim]/grid_spacing)
				if (coord[dim] < 0): # keeps from falling off top or left of grid
					coord[dim] = 0
				if (coord[dim] >= Grid.shape[0]): # keeps from falling off bottom or right of grid
					coord[dim] = Grid.shape[0] - 1
			
			if (coord[0]!=old_coord[i,0]) | (coord[1]!=old_coord[i,1]):
				Grid[ coord[0],coord[1] ] = i # update grid with new coords for ball i
				Grid[ old_coord[i,0],old_coord[i,1] ] = -1 # remove ball from previous grid coords



def preallocate_locks(num_locks):
	cdef omp_lock_t *locks = get_N_locks(num_locks)
	assert 0 != <uintptr_t> <void *> locks, "could not allocate locks"
	return <uintptr_t> <void *> locks
