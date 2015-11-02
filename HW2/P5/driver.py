# Note: includes all my comments from reviewing the skeleton code

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

# Helper function - calculates binary Morton codes for (x, y) coordinates
# Code from http://www.thejach.com/view/2011/9/playing_with_morton_numbers
def tomorton(x,y):
    x = bin(x)[2:]
    lx = len(x)
    y = bin(y)[2:]
    ly = len(y)
    L = max(lx, ly)
    m = 0
    for j in xrange(1, L+1):
    # note: ith bit of x requires x[lx - i] since our bin numbers are big endian
        xi = int(x[lx-j]) if j-1 < lx else 0
        yi = int(y[ly-j]) if j-1 < ly else 0
        m += 2**(2*j)*xi + 2**(2*j+1)*yi
    return m/4

# Helper function - calculates spatially coherent sorting
def get_mapped_index(mapped, pos, num_balls, grid_spacing):

    # Map coordinates to 1D Morton codes
    for i in range(num_balls):
        mapped[i] = tomorton(int(positions[i, 0] / grid_spacing), int(positions[i, 1] * grid_spacing))

    # Return index for sorting
    return np.argsort(mapped)

def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

if __name__ == '__main__':

    # Set the number of balls
    num_balls = 10000

    # Set the radius of each ball
    radius = 0.002
    
    # Randomly determine x, y coordinates for each ball
    # +/- radius ensures that the balls remain in range
    positions = np.random.uniform(0 + radius, 1 - radius,
                                  (num_balls, 2)).astype(np.float32)

    # Manipulate the position to make a hole in the center
    while True:
        
        # Check which balls' coordinates are in the center
        distance_from_center = np.sqrt(((positions - 0.5) ** 2).sum(axis=1))
        mask = (distance_from_center < 0.25)
        num_close_to_center = mask.sum()

        # Everything is out of the center
        if num_close_to_center == 0:
            break

        # Pick new random coordinates for balls that are in the center
        positions[mask, :] = np.random.uniform(0 + radius, 1 - radius,
                                               (num_close_to_center, 2)).astype(np.float32)

    # Determine x, y velocities for each ball
    velocities = np.random.uniform(-0.25, 0.25,
                                   (num_balls, 2)).astype(np.float32)

    ################################################################################
    # Initialize grid indices:
    # Each square in the grid stores the index of the object in that square, or
    # -1 if no object.  We don't worry about overlapping objects, and just
    # store one of them.
    ################################################################################

    # Size of grid spaces depends on radius (smaller radius -> smaller grid spaces)
    grid_spacing = radius / np.sqrt(2.0)

    # Size of grid depends on radius (smaller radius -> greater grid size)
    grid_size = int((1.0 / grid_spacing) + 1)

    # Initialize grid indices to -1
    grid = - np.ones((grid_size, grid_size), dtype=np.uint32)

    # Assign grid indices for balls based on x, y coordinates
    grid[(positions[:, 0] / grid_spacing).astype(int),
         (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)

    # Initialize matplotlib-based animator object
    animator = Animator(positions, radius * 2)

    # Simulation/animation time variablees
    physics_step = 1.0 / 100  # Estimate of real-time performance of simulation
    anim_step = 1.0 / 30  # FPS

    # Initialize runtime results
    total_time = 0
    frame_count = 0

    # Preallocate locks (1 per ball)
    locks_ptr = preallocate_locks(num_balls)

    # Initialize variables related to number of threads
    num_threads = 4
    chunk_size = num_balls/num_threads

    # Initialize array for sorting
    mapped_index = np.zeros(num_balls, dtype=np.uint32)

    while True:

        # Update ball positions & velocities
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_size, grid_spacing, locks_ptr,
                   physics_step, num_threads, chunk_size)

        # Update estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval

        # Update total runtime
        total_time += t.interval

        # Update number of frames
        frame_count += 1

        if total_time > anim_step:
            
            # Animator draws new frame
            animator.update(positions)

            # Print & store results
            print("{} simulation frames per second".format(frame_count / total_time))

            # Reset runtime results
            frame_count = 0
            total_time = 0

            # Sort indices by location
            mapped_index = get_mapped_index(mapped_index, positions, num_balls, grid_spacing)

            # Apply mapping to positions & velocities arrays
            positions = positions[mapped_index]
            velocities = velocities[mapped_index]

            # Update grid indices for balls based on x, y coordinates
            grid[(positions[:, 0] / grid_spacing).astype(int),
                 (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)