import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
from timer import Timer
import matplotlib
matplotlib.use('Qt4agg')
import matplotlib.pyplot as plt
from animator import Animator
from physics import update, preallocate_locks

from morton import zenumerate, zorder

UINT32_MAX = 4294967295

def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

if __name__ == '__main__':
    num_balls = 500
    radius = 0.01
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

    # Create a good sorting solution using morton indexing
    print 'Creating morton index...grid size is' , grid_size
    index_array = np.arange(grid_size*grid_size, dtype=np.int)
    index_matrix = index_array.reshape((grid_size, grid_size))
    zorder(index_matrix)
    zordered_indices = index_matrix.ravel()
    print 'Done!'

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

        frame_count += 1

        ball_indices = np.arange(num_balls)
        if total_time > anim_step:
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!

            # We cannot unravel the grid to sort, as overlapping elements will be missed! We need to sort on position
            # unfortunately.

            positions_in_grid = (positions/grid_spacing).astype(np.int)
            # Adjust the positions in grid, so that you are not actually outside the grid
            positions_in_grid[positions_in_grid >= grid_size] = grid_size -1
            positions_in_grid[positions_in_grid < 0] = 0
            logical_positions = grid_size*positions_in_grid[:, 0] + positions_in_grid[:, 1]
            order_fixer = np.argsort(zordered_indices[logical_positions])

            # Now index based on the new order
            positions = positions[order_fixer]
            velocities = velocities[order_fixer]

            # Based on the new positions, update the grid
            ordered_grid_positions = positions_in_grid[order_fixer]
            grid[ordered_grid_positions[:, 0], ordered_grid_positions[:, 1]] = ball_indices
