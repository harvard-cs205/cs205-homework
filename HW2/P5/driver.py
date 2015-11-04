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

# taken from http://blog.notdot.net/2009/11/Damn-Cool-Algorithms-Spatial-indexing-with-Quadtrees-and-Hilbert-Curves
# all copyright belongs there
# (used on a fair use basis)
# hilbert curve generator
hilbert_map = {
    'a': {(0, 0): (0, 'd'), (0, 1): (1, 'a'), (1, 0): (3, 'b'), (1, 1): (2, 'a')},
    'b': {(0, 0): (2, 'b'), (0, 1): (1, 'b'), (1, 0): (3, 'a'), (1, 1): (0, 'c')},
    'c': {(0, 0): (2, 'c'), (0, 1): (3, 'd'), (1, 0): (1, 'c'), (1, 1): (0, 'b')},
    'd': {(0, 0): (0, 'a'), (0, 1): (3, 'c'), (1, 0): (1, 'd'), (1, 1): (2, 'd')},
}

# get for a point with 2D indices x,y a Hilbert curve rank. order=10 defines a 2^10 x 2^10 grid
def point_to_hilbert(x, y, order=10):
  current_square = 'a'
  position = 0
  for i in range(order - 1, -1, -1):
    position <<= 2
    quad_x = 1 if x & (1 << i) else 0
    quad_y = 1 if y & (1 << i) else 0
    quad_position, current_square = hilbert_map[current_square][(quad_x, quad_y)]
    position |= quad_position
  return position


def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

if __name__ == '__main__':
    num_balls = 10000
    radius = 0.002

    # for test purposes
    #num_balls = 500
    #radius = 0.01

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

    # construct Hilbert curve
    hilbert_curve_order = int(np.ceil(np.log(grid_size) / np.log(2)))


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

    total_steps = 0
    while True:
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_size, locks_ptr,
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

            # Solution here uses the hilbert curve
            # compute the hilbert ranks for all points
            ranks = np.zeros(positions.shape[0])
            for i in range(positions.shape[0]):
                x = int(min(grid_size - 1, max(0, positions[i, 0] / grid_spacing)))
                y = int(min(grid_size - 1, max(0, positions[i, 1] / grid_spacing)))
                ranks[i] = point_to_hilbert(x, y, order=10) 

            # sort ranks and retrieve indices
            indices = np.argsort(ranks)
            # assign sorting to positions, velocities & update grid!
            positions = np.array(positions)[indices]
            velocities = np.array(velocities)[indices]

            # update grid
            grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
            for i in range(positions.shape[0]):
                x = int(min(grid_size - 1, max(0, positions[i, 0] / grid_spacing)))
                y = int(min(grid_size - 1, max(0, positions[i, 1] / grid_spacing)))
                grid[x,y] = i

            # in order to limit total computation
            total_steps +=1

        # quit after some measurements
        if total_steps > 20:
            break