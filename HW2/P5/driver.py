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

if __name__ == '__main__':
    #num_balls = 10000
    #radius = 0.002f
    num_balls = 500
    radius = 0.01
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
    expUse = np.ceil(np.log2(grid_size))
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
    count0 = 0
    fCounts = []

    while True:
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_spacing, locks_ptr,
                   physics_step)

        # udpate our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval

        grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
        
        positionsTest = np.copy((positions/grid_spacing).astype(int))
        positionsTest[positionsTest < 0] = 0
        positionsTest[positionsTest > grid_size - 1] = grid_size - 1
        grid[positionsTest[:, 0], positionsTest[:, 1]] = np.arange(num_balls)
       
        frame_count += 1
        if total_time > anim_step:
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            fCounts.append(frame_count / total_time)
            count0 = count0 + 1
            if count0 == 100:
                print min(fCounts),np.median(fCounts),max(fCounts)
                plt.figure()
                plt.hist(fCounts)
                plt.show()
                plt.title('Locking, Sorted, 4 Threads')
                plt.xlabel('Frames/second')
                plt.savefig('4_thread_locking_sorted.png')
                break
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!

            # Use Morton Ordering
            positionsY = map(lambda x: (expUse-len(bin(x)[2:]))*'0' + bin(x)[2:],(positions[:,0]/grid_spacing).astype(int))
            positionsX = map(lambda x: (expUse-len(bin(x)[2:]))*'0' + bin(x)[2:],(positions[:,1]/grid_spacing).astype(int))
            positionsY2 = map(lambda x: int(x[0]+'0'+x[1]+'0'+x[2]+'0'+x[3]+'0'+x[4]+'0'+x[5]+'0'+x[6]+'0'+x[7]+'0'),positionsY)
            positionsX2 = map(lambda x: int('0'+x[0]+'0'+x[1]+'0'+x[2]+'0'+x[3]+'0'+x[4]+'0'+x[5]+'0'+x[6]+'0'+x[7]),positionsX)
            zOrder = [int(str(sum(x)),2) for x in zip(positionsY2,positionsX2)]
            newOrder = np.argsort(zOrder)
            positions = np.copy(positions[newOrder])
            velocities = np.copy(velocities[newOrder])
    
            grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
            positionsTest = np.copy((positions/grid_spacing).astype(int))
            positionsTest[positionsTest < 0] = 0
            positionsTest[positionsTest > grid_size - 1] = grid_size - 1
            grid[positionsTest[:, 0], positionsTest[:, 1]] = np.arange(num_balls)
            
            
