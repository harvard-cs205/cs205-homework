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
import matplotlib.pyplot as plt


def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1


def xy2d(x, n):
    x = x.copy()
    # Find the nearest power of 2
    p = 2**(n-1).bit_length()
    d = 0
    s = (p + 1)/2
    while s > 0:
        rx = (x[0] & s) > 0
        ry = (x[1] & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        x[0], x[1] = rot(s, x[0], x[1], rx, ry)
        s /= 2
    return d


def rot(n, x, y, rx, ry):
    if (ry == 0):
        if (rx == 1):
            x = n-1 - x
            y = n-1 - y

        # Swap x and y
        x, y = y, x
    return x, y


if __name__ == '__main__':
    # Prod value
    num_balls = 10000
    radius = 0.002

    # Test valiue
    # num_balls = 500
    # radius = 0.01

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
    # animator = Animator(positions, radius * 2)

    # simulation/animation time variablees
    physics_step = 1.0 / 100  # estimate of real-time performance of simulation
    anim_step = 1.0 / 30  # FPS
    total_time = 0

    frame_count = 0

    # SUBPROBLEM 4: uncomment the code below.
    # preallocate locks for objects
    locks_ptr = preallocate_locks(num_balls)

    # Average number for performance evaluation
    iterations = 200
    num_threads = 1
    i = 0
    sfps_cum = 0
    sfps = []
    while i < iterations:
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_spacing, locks_ptr,
                   physics_step, num_threads)

        # udpate our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval

        frame_count += 1
        if total_time > anim_step:
            i += 1
            # animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            sfps_cum += frame_count / total_time
            sfps.append(frame_count / total_time)
            frame_count = 0
            total_time = 0

            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!

            # Changing positions to have int coordinates
            coor = np.apply_along_axis(lambda x: [int(x[0] / grid_spacing),
                                       int(x[1] / grid_spacing)], 1, positions)
            # Computing the list of Hilbert rank
            rank = np.apply_along_axis(xy2d, 1, coor, grid_size)
            args = np.argsort(rank)

            # Updating arrays
            positions = positions[args, :]
            velocities = velocities[args, :]
            grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
            grid[(positions[:, 0] / grid_spacing).astype(int),
                 (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)

    average = sfps_cum/iterations
    print('Average simulation frames per second is {}'.format(average))

    # Plotting the histogramm
    plt.hist(sfps, bins=100)

    plt.xlabel('Frequency')
    plt.ylabel('simulation frames per second')

    plt.title('Performance with {} threads'.format(num_threads))

    plt.show()
