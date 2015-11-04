import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import matplotlib.pyplot as plt
from timer import Timer
from animator import Animator
from physics import update, preallocate_locks


def cmp_zorder(a, b):
    '''
    Adapted from https://en.wikipedia.org/wiki/Z-order_curve#cite_note-5
    Takes two coord pairs as input and calculates the z order (Morton order)
    Returns -1 if a < b, 0 if a == b, and 1 if a > b
    '''
    #translate the floats to ints (approximately)
    a_int = (int(a[0]* 10000),int(a[1] *10000))
    b_int = (int(b[0]* 10000),int(b[1] *10000))
    max_idx = 0
    max_xor = 0
    # for each coord in a, b
    for idx in range(2):
        #take the xor
        xor = a_int[idx] ^ b_int[idx]
        #check if the xor is greater than the previous max_xor
        if less_msb(max_xor, xor):
            #update max
            max_idx = idx
            max_xor = xor
    diff = a_int[max_idx] - b_int[max_idx]
    if diff > 0:
        return 1
    elif diff == 0:
        return 0
    else:
        return -1
    
def less_msb(max_xor, xor):
    # Adapted from https://en.wikipedia.org/wiki/Z-order_curve#cite_note-5
    # if the most significant bit of xor is greater than max_xor, returns true 
    return max_xor < xor and max_xor < (max_xor ^ xor)


def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

if __name__ == '__main__':
    num_balls = 10000
    radius = 0.002
    #num_balls = 500
    #radius = .01
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
    
    while True:
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_spacing, locks_ptr,
                   physics_step) # changed this from grid_size to grid_spacing

        # udpate our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval

        frame_count += 1
        if total_time > anim_step:
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!
            
            #idea hash table where key = position, value = (old_index, new_index)
            # use the hash table to update indices in the grid
            #store old index
            position_to_idx = dict()# hash table used to update positions, grid, and velocities when sorting
            for i in range(num_balls):
                coords = (positions[i,0], positions[i,1])
                position_to_idx[coords] = i
                
            #sort positions using Morton Order
            positions = np.array(sorted(list(positions),cmp=cmp_zorder))
            for j in range(num_balls):
                coords = (positions[j,0], positions[j,1])
                #store new index
                #position to index is updated so that key = position coords, value = (old index, new index)
                position_to_idx[coords] = (position_to_idx[coords],j)
                #update grid
                grid_x, grid_y = int(coords[0]/grid_spacing), int(coords[1]/grid_spacing)
                if ((grid_x >= 0) and (grid_x < grid_size) and (grid_y >= 0) and (grid_y < grid_size)):
                    grid[grid_x,grid_y] = j
            
            #list where of old indexes, but in indexed by new index: e.g. old_to_new[new_idx] = old_idx
            old_to_new = list(zip(*sorted(position_to_idx.values(),key=lambda x: x[1]))[0])
            #update velocities
            velocities = velocities[old_to_new]