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
import pdb
# import the Hilbert ordering
# check this website for details:
# http://www.tiac.net/~sw/2008/10/Hilbert
from hilbert import Hilbert_to_int

def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

if __name__ == '__main__':
    num_balls = 1e4
    radius = 0.002
    n_threads = 4
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
    # int_max (=4294967295) if no object.  We don't worry about overlapping objects, and just
    # store one of them.



    #  size of one side = grid_spacing
    #  --------
    # |        |
    # |        |
    # |        |
    # |        |
    #  --------
    # size of a diagonal is R
    # need 4 grids to fit one ball
    grid_spacing = radius / np.sqrt(2.0)
    grid_size = int((1.0 / grid_spacing) + 1)   
    grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
    # look at the positions of the balls in the grid
    # attribute a number to each ball
    grid[(positions[:, 0] / grid_spacing).astype(int),
         (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)


    # A matplotlib-based animator object
    animator = Animator(positions, radius * 2, grid_size)

    # simulation/animation time variablees
    physics_step = 1.0 / 100  # estimate of real-time performance of simulation
    anim_step = 1.0 / 30  # FPS
    total_time = 0

    frame_count = 0

    # SUBPROBLEM 4: uncomment the code below.
    # preallocate locks for objects
    locks_ptr = preallocate_locks(num_balls)

    iterations = 0

    while True:
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_spacing, locks_ptr,
                   physics_step, n_threads)

        # udpate our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval
        iterations += 1

        frame_count += 1
        if total_time > anim_step:
            animator.update(positions)  
            print("{} simulation frames per second".format(frame_count / total_time))
            #pdb.set_trace()              
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!

            # we choose to order using Hilbert curve
            positions_hilbert = [Hilbert_to_int([int(x_grid),int(y_grid)]) for x_grid, y_grid in zip(positions[:,0]/grid_spacing,positions[:,1]/grid_spacing)]

            # sort this : grid_sorted gives an array with the index of positions_hilbert
            # once sorted
            grid_sorted = np.argsort(positions_hilbert)

            # reorder the positions with this
            positions = positions[grid_sorted]
            velocities = velocities[grid_sorted]

            # because we changed the ordering
            # we update the grid --> change the colors in animation.py
            grid[(positions[:,0]/grid_spacing).astype(int),
                (positions[:,0]/grid_spacing).astype(int)] = np.arange(num_balls)


