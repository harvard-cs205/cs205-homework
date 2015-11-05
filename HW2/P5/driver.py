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

# Morton ordering, with code obtained from Wikipedia
def cmp_zorder(a, b):
    j = 0
    k = 0
    x = 0
    for k in range(2):
        y = a[k] ^ b[k]
        if less_msb(x, y):
            j = k
            x = y
    return a[j] - b[j]

def less_msb(x, y):
    return x < y and x < (x ^ y)

if __name__ == '__main__':
    num_balls = 10000
    radius = 0.002
#    num_balls = 500
#    radius = 0.01
    # create a uniformly randomized list of (x,y) coordinates for balls within a [0,1]^2 grid
    positions = np.random.uniform(0 + radius, 1 - radius,
                                  (num_balls, 2)).astype(np.float32)

    # make a hole in the center
    while True:
        # vector of distances from position (0.5, 0.5)
        distance_from_center = np.sqrt(((positions - 0.5) ** 2).sum(axis=1))
        # obtain list of indicator values if ball is 0.25 units away from the center
        mask = (distance_from_center < 0.25)
        # count the number of balls close to the center
        num_close_to_center = mask.sum()
        # break if there is a hole
        if num_close_to_center == 0:
            # everything is out of the center
            break
        # else reshuffle only those balls that need reshuffling
        positions[mask, :] = np.random.uniform(0 + radius, 1 - radius,
                                               (num_close_to_center, 2)).astype(np.float32)

    # create a uniformly randomized list of (vx,vy) velocities for balls
    velocities = np.random.uniform(-0.25, 0.25,
                                   (num_balls, 2)).astype(np.float32)

    # Initialize grid indices:
    #
    # Each square in the grid stores the index of the object in that square, or
    # -1 if no object.  We don't worry about overlapping objects, and just
    # store one of them.
    
    # grid spacing of R/sqrt(2) (area of subdivision is R^2/2)
    grid_spacing = radius / np.sqrt(2.0)
    # number of subdivisions across the grid given the radius
    grid_size = int((1.0 / grid_spacing) + 1)
    # initialize the grid with dimensions grid_size x grid_size
    # random note: these values come out to be 4294967295, not -1 as I would suspect
    grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
    # add indices of existing balls into the grid (possible collisions?)
    grid[(positions[:, 0] / grid_spacing).astype(int),
         (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)
    
    # A matplotlib-based animator object
    animator = Animator(positions, radius * 2)

    # simulation/animation time variablees
    physics_step = 1.0 / 100  # estimate of real-time performance of simulation
    anim_step = 1.0 / 30  # FPS
    total_time = 0

    frame_count = 0

    # preallocate locks for objects
    locks_ptr = preallocate_locks(num_balls)

    # initialize totals for final average
    total_total_time = 0
    total_frame_count = 0
    i = 0
    
    # number of iterations to average over
    iterations = 100
    
    while True:
        with Timer() as t:
            # input grid_spacing instead of grid_size
            update(positions, velocities, grid,
                   radius, grid_spacing, locks_ptr,
                   physics_step)

        # update our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval

        frame_count += 1
        if total_time > anim_step:
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            total_frame_count += frame_count
            total_total_time += total_time
            frame_count = 0
            total_time = 0
            i += 1
            if i == iterations:
                print("Average: {} simulation frames per second".format(total_frame_count / total_total_time))
                break
            
            # sort objects by location using a Morton ordering procedure            
            grid_positions = (positions / grid_spacing).astype("int")
            # map grid_positions + offset to the morton ordering function, then obtain the sorted indices
            grid_indices = np.argsort(map(lambda x, y: cmp_zorder(x, y), grid_positions, np.array(list(grid_positions[1:]) + list(grid_positions[:1]))))

            # update positions and velocities with the new indices
            positions = np.array(positions)[grid_indices]
            velocities = np.array(velocities)[grid_indices]

            # refresh the grid
            grid[:,:] = -1

            # add back positions into the new grid
            for k in range(num_balls):
                if ((0 <= positions[k,0] < 1) and (0 <= positions[k,1] < 1)):
                    grid[(positions[k,0] / grid_spacing).astype(int), (positions[k,1] / grid_spacing).astype(int)] = k
