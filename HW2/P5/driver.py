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
from physics import set_hilbert_array, cleargrid, setgrid, update, preallocate_locks

def hilbert_xy2d(n, x, y):
    d = 0
    s = n / 2
    rx = 0
    ry = 0
    while s > 0:
        if (x & s) > 0:
            rx = 1
        else:
            rx = 0
        if (y & s) > 0:
            ry = 1
        else:
            ry = 0

        d += s * s * ((3 * rx) ^ ry)
        if ry == 0: 
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            t = x
            x = y
            y = t
        s = s / 2

    return d


def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

if __name__ == '__main__':
    num_balls = 10 
    radius = 0.1
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

    # Construct hilbert_map for sorting in SUBPROBLEM 3
    hilbert_array = np.ones(num_balls, dtype=np.uint32)
    hilbert_map = np.zeros((grid_size, grid_size), dtype=np.uint32)
    n = 1
    while n < grid_size:
        n = n * 2
    for x in range(grid_size):
        for y in range(grid_size):
            hilbert_map[x, y] = hilbert_xy2d(n, x, y)

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
        #print("t: %f" % t.interval)

        # udpate our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval

        frame_count += 1
        if total_time > anim_step:
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            frame_count = 0
            total_time = 0
            # uncomment the "continue" line to disable reorder of positions 
            # continue

            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!
            set_hilbert_array(positions, hilbert_map, hilbert_array, grid_spacing)
            sort_idx = np.argsort(hilbert_array, kind="quiksort")

            cleargrid(positions, grid, grid_spacing)
            positions[:, :] = positions[sort_idx]
            velocities[:, :] = velocities[sort_idx]
            setgrid(positions, grid, grid_spacing)
