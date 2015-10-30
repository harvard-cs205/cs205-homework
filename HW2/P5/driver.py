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

def interleave(a, b):
    '''
    Interleave the bits of integers a and b and return
    an integer representation of the result.
    '''
    abin = '{0:016b}'.format(a)
    bbin = '{0:016b}'.format(b)
    inter = ''
    for i in range(16):
        inter += abin[-i-1]
        inter += bbin[-i-1]
    return int(inter, 2)

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
    interlist = []
    speedtest = 0
    total_iters = 0

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
            total_iters += 1
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            # maintain a running average to make it easier to analyze performance
            speedtest += frame_count / total_time
            print("{} average frames per second").format(speedtest/total_iters)
            frame_count = 0
            total_time = 0

            # This is Z-ordering / Morton ordering. Note: as far as I know, you can't really
            # interleave the bits of a float, so I just multiply the floats by 10,000 and
            # convert them to integers, dropping the trailing (and hopefully unimportant)
            # digits. There's probably a better way, but this seems to work pretty well.
            interlist = []
            for i in range(num_balls):
                interlist.append(interleave(int(positions[i, 0]*10000), 
                    int(10000*positions[i, 1])))
            
            idx = np.argsort(interlist)
            positions = positions[idx]
            velocities = velocities[idx]
            xidx, yidx = np.where(np.invert((0 <= grid)-(grid<num_balls))) # where 0<=grid<num_balls
            for m in zip(xidx, yidx):
                grid[m] = idx[grid[m]]
