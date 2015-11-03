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

# from wiki
def zorder(p, q):
    j, k, x = 0, 0, 0
    a = p[1]
    b = q[1]
    for k in range(2):
        y = a[k] ^ b[k]
        if (x < y and x < (x ^ y)):
            j = k
            x = y
    return a[j] - b[j]


if __name__ == '__main__':
    num_balls = 10000
    radius = 0.002
    pos = np.random.uniform(0 + radius, 1 - radius,
                                  (num_balls, 2)).astype(np.float32)


    # make a hole in the center
    while True:
        distance_from_center = np.sqrt(((pos - 0.5) ** 2).sum(axis=1))
        mask = (distance_from_center < 0.25)
        num_close_to_center = mask.sum()
        if num_close_to_center == 0:
            # everything is out of the center
            break
        pos[mask, :] = np.random.uniform(0 + radius, 1 - radius,
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
    grid[(pos[:, 0] / grid_spacing).astype(int),
         (pos[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)

    # A matplotlib-based animator object
    animator = Animator(pos, radius * 2)

    # simulation/animation time variablees
    physics_step = 1.0 / 100  # estimate of real-time performance of simulation
    anim_step = 1.0 / 30  # FPS
    total_time = 0

    frame_count = 0

    # SUBPROBLEM 4: uncomment the code below.
    # preallocate locks for objects
    locks_ptr = preallocate_locks(num_balls)
    interval = []

    while True:
        with Timer() as t:
            update(pos, velocities, grid,
                   radius, grid_spacing, locks_ptr,
                   physics_step)


        # udpate our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval

        frame_count += 1
        if total_time > anim_step:
            animator.update(pos)
            print("{} simulation frames per second".format(frame_count / total_time))
            interval += [frame_count / total_time]
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object pos!
            # Sort the grid using morton order
            grid_pos = (pos / grid_spacing).astype(int)
            i_pos = zip(range(len(grid_pos)), grid_pos).sort(zorder)
            index = [x[0] for x in i_pos]
            pos = pos[index]
            velocities = velocities[index]
            grid_bound = np.array(filter(lambda x: x[0] > 0 and x[1] < 1, pos))
            grid = - np.ones((grid_size, grid_size), dtype = np.uint32)
            grid[(grid_bound[:, 0] / grid_spacing).astype(int),
                 (grid_bound[:, 1] / grid_spacing).astype(int)] = np.arange(len(grid_bound))




