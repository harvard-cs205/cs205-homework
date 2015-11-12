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

# Convert a set of coordinates to a distance.
# Taken from: https://en.wikipedia.org/wiki/Hilbert_curve
# and adapted to python.
def xy2d (n, x, y):
    d = 0
    s = n/2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(s, x, y, rx, ry)
        s /= 2
    return d

# Rotate/flip a quadrant appropriately, helper
# function for xy2d.
def rot(n, x, y, rx, ry):
    if (ry == 0):
        if (rx == 1):
            x = n-1 - x;
            y = n-1 - y;
    return y, x

def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

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
                   physics_step, grid_size)

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

            # Map every point to its corresponding d in the hilbert ordering.
            hilbs = [xy2d(grid_size, int(b[0]), int(b[1])) for b in positions]
            # Sort based on this ordering.
            res = np.argsort(hilbs)

            # Reorder the positions and velocities lists (praise numpy indexing syntax).
            positions = positions[res]
            velocities = velocities[res]

            # Reinitialize the grid and make sure that nothing out of bounds is attempting to
            # be added to the grid.
            grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
            x_positions = (positions[:,0]/grid_spacing).astype(int)
            y_positions = (positions[:,1]/grid_spacing).astype(int)
            filtered_x = x_positions[((x_positions < grid_size) & (x_positions >= 0)) \
                     & ((y_positions < grid_size) & (y_positions >= 0))]
            filtered_y = y_positions[((x_positions < grid_size) & (x_positions >= 0)) & \
                     ((y_positions < grid_size) & (y_positions >= 0))]
            grid[filtered_x, filtered_y] = np.arange(num_balls)[((x_positions < grid_size) \
                     & (x_positions >= 0)) & ((y_positions < grid_size) & (y_positions >= 0))]
