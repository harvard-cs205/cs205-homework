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

import matplotlib.pylab as plt


def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

#### Copied these two functions from https://en.wikipedia.org/wiki/Z-order_curve to use for Morton ordering.
#### Modified to take inputs of the form (pos_index, [grid_x, grid_y]) as used below  
def cmp_zorder(a, b): 
    j = 0
    k = 0
    x = 0
    for k in range(2):  # <---- Changed from dim to 2 from Wikipedia code (2 dimensions)
        y = a[k] ^ b[k] 
        if less_msb(x, y):
            j = k
            x = y
    return a[j] - b[j]
    
def less_msb(x, y):
        return x < y and x < (x ^ y)

# This function is passed to the sorted function to extract the grid coordinate tuple portion of a list of
#     indexed coordinates
def get_key(a):
    return a[1]

#---------------------------------------------------------------------------------------------------------#

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
    
    # Keep track of the speed of each iteration
    speeds = []  

    while True: 
    # while len(speeds) < 31: # I get 30 speeds to put in my histogram
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_spacing, locks_ptr,
                   physics_step)
                   #  Changed the argument from grid_size to grid_spacing as mentioned in Piazza

        # update our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval

        frame_count += 1
        if total_time > anim_step:
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            
            # Add the speed for this iteration to my list
            speeds.append(frame_count/total_time) 

            frame_count = 0
            total_time = 0
            
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!
            
            # I use the Morton ordering, borrowing the code from the Wikipedia article cited in the
            #      assignment.            
            
            # Make tuples of position indices and grid indices of the form (pos_index, [x_grid, y_grid]).
            gridindex = zip(range(len(positions)), (positions/grid_spacing).astype(int))

            # Apply Morton sort using cmp_zorder.
            gridindex = sorted(gridindex, cmp_zorder, key=get_key) 

            # Extract just the position index. This is the new order.  
            gridindex = map(lambda x: x[0], gridindex) 

            # Reorder positions and velocities by the new sorted index
            positions = positions[gridindex]
            velocities=velocities[gridindex]

            # Empty the grid out before refilling it
            grid[:,:] = -1

            # Reset the grid with the updated positions
            for i in range(len(positions)):
                grid[(positions[i, 0] / grid_spacing).astype(int), 
                     (positions[i, 1] / grid_spacing).astype(int)] = i
        
    # Create a histogram of the speeds        
    plt.pause(10.**-6.)
    fig, ax = plt.subplots()
    ax.hist(speeds, bins=8)
    #ax.set_xlim([2,8])
    ax.set_xlabel('Simulation Frames Per Second')
    ax.set_ylabel('Frequency')
    ax.set_title('Serial')
    print(np.mean(speeds))
    plt.pause(10.**10.) # Have to pause or else the histogram disappears immediately
    
