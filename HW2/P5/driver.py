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

def morton_grid(grid_size): 
    grid = np.array([0, 1, 2, 3], dtype=np.int32).reshape((2,2))
    size = 2
    while size < grid_size:
        newgrid = np.zeros((size*2, size*2), dtype=np.int32)
        newgrid[0:size, 0:size] = grid
        newgrid[0:size, size:size*2] = grid + size*size
        newgrid[size:size*2, 0:size] = grid + size*size*2
        newgrid[size:, size:] = grid + size*size*3
        grid = newgrid
        size *= 2
    return grid

if __name__ == '__main__':
    num_balls, radius = 10000, 0.002
    # num_balls, radius = 200, 0.01
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

    morton_order = morton_grid(grid_size)[:grid_size, :grid_size]
    
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
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!
            '''
            # stack the current pos label with its Morton priority together
            pos_mrt_dstack = np.dstack((grid, morton_order)).reshape((grid_size*grid_size, 2))
            # filter out those -1
            pos_mrt_dstack = filter(lambda x: x[0] != np.iinfo(np.uint32).max, pos_mrt_dstack)
            # sort by morton order
            pos_mrt_dstack = np.array(sorted(pos_mrt_dstack, cmp=lambda x, y: x[1] - y[1]))
            # get the new order of each position
            new_posindex = pos_mrt_dstack[:, 0]
            ''' 
            positions = positions[new_posindex, :]
            new_posindex = morton_order[(positions[:, 0] / grid_spacing).astype(int),
                                        (positions[:, 1] / grid_spacing).astype(int)]
            
            new_posindex = np.argsort(new_posindex)
            velocities = velocities[new_posindex, :]
            
            grid[(positions[:, 0] / grid_spacing).astype(int),
                (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)
