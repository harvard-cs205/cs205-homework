import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import matplotlib.pyplot as plt
from timer import Timer
from animator import Animator
from physics import update, preallocate_locks, grid_index

def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

#Hibert sorting for part 3
#Adapted from http://blog.notdot.net/2009/11/Damn-Cool-Algorithms-Spatial-indexing-with-Quadtrees-and-Hilbert-Curves

#Starting map
hilbert_map = {
    'a': {(0, 0): (0, 'd'), (0, 1): (1, 'a'), (1, 0): (3, 'b'), (1, 1): (2, 'a')},
    'b': {(0, 0): (2, 'b'), (0, 1): (1, 'b'), (1, 0): (3, 'a'), (1, 1): (0, 'c')},
    'c': {(0, 0): (2, 'c'), (0, 1): (3, 'd'), (1, 0): (1, 'c'), (1, 1): (0, 'b')},
    'd': {(0, 0): (0, 'a'), (0, 1): (3, 'c'), (1, 0): (1, 'd'), (1, 1): (2, 'd')},
}

def point_to_hilbert(coords, spacing, max_index):
    #Convert coords to ints for bit manipulation
    floatx, floaty = tuple(coords)
    x = grid_index(floatx, spacing, max_index)
    y = grid_index(floaty, spacing, max_index)
    #For a Hilbert curve of order n, each dimension should range between 0 and 2n - 1
    n = (max_index + 1) / 2
    current_square = 'a'
    position = 0
    #Hilbert calculations
    for i in range(n - 1, -1, -1):
        position <<= 2
        quad_x = 1 if x & (1 << i) else 0
        quad_y = 1 if y & (1 << i) else 0
        quad_position, current_square = hilbert_map[current_square][(quad_x, quad_y)]
        position |= quad_position
    return position

if __name__ == '__main__':
    num_balls = 10000 #500
    radius = 0.002 #0.01
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
    grid_spacing = radius / np.sqrt(2.0) #Distance between 2 grid points
    grid_size = int((1.0 / grid_spacing) + 1) #Number of points on each axis
    grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
    grid[(positions[:, 0] / grid_spacing).astype(int),
         (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)

    # A matplotlib-based animator object
    #animator = Animator(positions, radius * 2)

    # simulation/animation time variablees
    physics_step = 1.0 / 100  # estimate of real-time performance of simulation
    anim_step = 1.0 / 30  # FPS
    total_time = 0

    frame_count = 0

    # SUBPROBLEM 4: uncomment the code below.
    # preallocate locks for objects
    locks_ptr = preallocate_locks(num_balls)

    histogram_vals = []
    
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
            #animator.update(positions)
            fps = frame_count / total_time
            print("{} simulation frames per second".format(fps))
            histogram_vals.append(fps)
            if len(histogram_vals) == 30:
                plt.hist(histogram_vals)
                plt.savefig('SCS-1thread.png')
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!
            
            #Sort according to Hilbert curve distance
            distances = np.apply_along_axis(point_to_hilbert, 1, positions, grid_spacing, grid_size)
            order = np.argsort(distances)
            
            #Update velocities and positions (grid is updated in physics.pyx)
            velocities = np.array(velocities)[order]
            positions = np.array(positions)[order]
