import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
from timer import Timer
from animator import Animator
from physics import update, preallocate_locks

def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

#Code copied from wikipedia
def cmp_zorder(a, b):
    j = 0
    k = 0
    x = 0
    for k in range(2):
        #Since passing the index as well
        y = a[1][k] ^ b[1][k]
        if less_msb(x, y):
            j = k
            x = y
    return a[1][j] - b[1][j]
    
def less_msb(x, y):
        return x < y and x < (x ^ y)

if __name__ == '__main__':
    num_balls = 10000
    radius = 0.002
    positions = np.random.uniform(0 + radius, 1 - radius,
                                  (num_balls, 2)).astype(np.float32)

    # make a hole in the center
    while True:
        distance_from_center = np.sqrt(((positions - 0.5) ** 2).sum(axis=1))
        mask = (distance_from_center < 0.25)
        num_close_to_center = mask.sum()
        if num_close_to_center == 0:
            # everything is out of the center
            break
        positions[mask, :] = np.random.uniform(0 + radius, 1 - radius,
                                               (num_close_to_center, 2)).astype(np.float32)

    velocities = np.random.uniform(-0.25, 0.25,
                                   (num_balls, 2)).astype(np.float32)

    # Initialize grid indices:
    #
    # Each square in the grid stores the index of the object in that square, or
    # -1 if no object.  We don't worry about overlapping objects, and just
    # store one of them.
    grid_spacing = radius / np.sqrt(2.0)
    grid_size = int((1.0 / grid_spacing) + 1)
    grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
    grid[(positions[:, 0] / grid_spacing).astype(int),
         (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)
    # A matplotlib-based animator object
    animator = Animator(positions, radius * 2)

    # simulation/animation time variablees
    physics_step = 1.0 / 100  # estimate of real-time performance of simulation
    anim_step = 1.0 / 30  # FPS
    total_time = 0

    frame_count = 0

    # SUBPROBLEM 4: uncomment the code below.
    # preallocate locks for objects
    locks_ptr = preallocate_locks(num_balls)

    #frame_list = []
    while True:
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_spacing, locks_ptr,
                   physics_step)

        # udpate our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval
        frame_count += 1
        if total_time > anim_step:
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            #frame_list.append(frame_count/total_time)
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            #velocities with their object positions!    
            grid_positions = (positions/grid_spacing).astype(int)
            #Adding index
            grid_positions = zip(range(grid_positions.shape[0]),grid_positions)
            #Sorting as per z-order function
            grid_positions.sort(cmp_zorder)
            #Extracting new indices
            new_idx = map(lambda x:x[0],grid_positions)
            #Using new indices to get sorted positions and velocities
            new_pos = np.take(positions,new_idx,axis=0)
            new_v = np.take(velocities,new_idx,axis=0)
            #Resetting the grid completely
            grid.fill(-1)
            positions = new_pos
            velocities = new_v
            #Assigning the grid positions
            for i in range(num_balls):
                if positions[i,0] >= 0 and positions[i,0] <=1 and\
                        positions[i,1] >= 0 and positions[i,1] <=1:
                    grid_index1 = (positions[i,0]/grid_spacing)\
                                    .astype('int')
                    grid_index2 = (positions[i,1]/grid_spacing)\
                                    .astype('int')
                    grid[grid_index1,grid_index2] = i
    
    #For averaging frames per second
    #print "Average frames per second:",np.mean(frame_list)
    
