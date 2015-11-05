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

################################################################  
#implementation of z-order from https://en.wikipedia.org/wiki/Z-order_curve 

def z_sort(x,y):
    return cmp_zorder(x[1],y[1])

def cmp_zorder(a, b):
    j = 0
    k = 0
    x = 0
    for k in range(2):
        y = a[k] ^ b[k]
        if less_msb(x, y):
            j = k
            x = y
    return a[j] - b[j]

def less_msb(x, y):
    return x < y and x < (x ^ y)

################################################################

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

    while True:
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_spacing, locks_ptr,
                   physics_step)

        # udpate our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval
        #print positions
        frame_count += 1
        if total_time > anim_step:
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            frame_count = 0
            total_time = 0

            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!
            
            #convert position to int for sorting
            old_positions = (positions * 100).astype(int)
            #convert (x,y) position into one order index
            index_positions = zip(range(len(old_positions)), old_positions)
            index_positions.sort(z_sort)
            order_index = [x[0] for x in index_positions]
            
            #sort position and velocity list based on order_index
            new_positions =np.array([positions[i] for i in order_index])
            new_velocities = np.array([velocities[i] for i in order_index])
            #print new_positions
            #reset grid
            positions = new_positions
            velocities = new_velocities
            #print positions[12,0],positions[12,1]
            grid[:,:] = 4294967295
            for i in range(num_balls):
                if (positions[i,0] >= 0 and positions[i,0] <= grid_size 
                    and positions[i,1] >= 0 and positions[i,1] <= grid_size):
                    grid_x = int(positions[i,0] / grid_spacing)
                    grid_y = int(positions[i,1] / grid_spacing)
                    grid[grid_x, grid_y] = i


