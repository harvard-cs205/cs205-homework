
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
from physics import update, preallocate_locks#, xy2d

# import matplotlib
# matplotlib.use('TkAgg')
# import pylab as plt

#Worked with Sami Goche on this part. 

#These functions are for Hilbert Ordering for 5.3, rewritten in Python from
#https://en.wikipedia.org/wiki/Hilbert_curve

def xy2d(n, x, y):
    d = 0
    s = n/2
    while s > 0:
        x1 = x[0]
        y1 = y[0]
        rx = (x1 & s) > 0
        ry = (y1 & s) > 0
        d += s*s*((3*rx)^ry)
        rot(s, x, y, rx, ry)
        s /= 2
    return d

def rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x[0] = n-1 - x[0]
            y[0] = n-1 - y[0]
        t = x[0]
        x[0] = y[0]
        y[0] = t

#When we do spatially ordering, according to the wiki-entry I needed to rescale
#my coordinates to be integers along some axis bounded by values that were perfect
#squares. Just arbitraily chose 256. 
nValue = 256

def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

if __name__ == '__main__':
    # num_balls = 500
    # radius = 0.01
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

            #The way I do distvalues is that I map the xy2d function over positions, which basically
            #gets the a single "distance" value for each of the (x, y) coordinates. Because of some of
            #the restrictions found on the wiki page about how this function worked, I needed to rescale
            #the values by nValue (explained above) and make sure they were ints.

            distValues = map(lambda x: xy2d(nValue, [int(x[0]*nValue)], [int(x[1]*nValue)]), positions)

            #Once I get the distValues above, I want to sort the positions by their relative distances.
            #I use np.argsort as recommended in the pset directions to do this, 
            #the re-evaluate the positions and the grid. 
            
            sortedIndices = np.argsort(distValues)
            positions = positions[sortedIndices, :]
            grid[(positions[:, 0] / grid_spacing).astype(int),
                (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)
            velocities = velocities[sortedIndices, :]



