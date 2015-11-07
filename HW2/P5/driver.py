import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import math
from timer import Timer
from animator import Animator
from physics import update, preallocate_locks, xy2hilbert, get_sorted_order

def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

if __name__ == '__main__':
   #num_balls = 500
   #radius = 0.01
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
    grid_spacing = radius * np.sqrt(2.0)
    grid_size = int((1.0 / grid_spacing) + 1)
    grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
    grid[(positions[:, 0] / grid_spacing).astype(int),
         (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)

    # Set up Hilbert ordering for subproblem 3
    # n is greatest power of 2 less than number of objects
    n = 2 ** (math.frexp(num_balls)[1] - 1)
    hilbert = np.zeros_like(grid, dtype=np.uint32)
    for x in range(hilbert.shape[0]):
        for y in range(hilbert.shape[1]):
            hilbert[x, y] = xy2hilbert(n, x, y)
    hilbert_1d = np.zeros(num_balls, dtype=np.uint32)

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
        if total_time > anim_step:
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!
            get_sorted_order(positions, hilbert_1d, hilbert, grid_spacing)
            indices_sorted = np.argsort(hilbert_1d)
            positions[:,:] = positions[indices_sorted]
            velocities[:,:] = velocities[indices_sorted]
