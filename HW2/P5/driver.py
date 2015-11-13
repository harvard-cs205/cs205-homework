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
from physics import update, preallocate_locks, find_hilbert
import pdb
import matplotlib.pyplot as plt

def randcolor():
    return np.random.uniform(0.0, 0.89, (3,)) + 0.1

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
    hilbert_size=64
    # Initialize grid indices:
    #
    # Each square in the grid stores the index of the object in that square, or
    # -1 if no object.  We don't worry about overlapping objects, and just
    # store one of them.



    grid_spacing = radius / np.sqrt(2.0)
    grid_size = int((1.0 / grid_spacing) + 1)
    grid = - np.ones((grid_size, grid_size), dtype=np.uint32)
    order=np.ones(num_balls,dtype=np.uint32)
    
    find_hilbert(hilbert_size,(hilbert_size*positions).astype(np.uint32),num_balls,order)
    order_idx=np.argsort(order)
    positions=positions[order_idx,:]
    velocities=velocities[order_idx,:]
    grid[(positions[:, 0] / grid_spacing).astype(int),
         (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)
    # fig = plt.figure(1)
    # fig.clf()
    # ax = fig.add_subplot(1, 1, 1)
    # cmap = cm.jet
    # clist = np.linspace(0, 10, positions.shape[0])
    
    # ax.scatter(positions[:,0],positions[:,1],c=clist, cmap=cmap)
    # fig2 = plt.figure(2)
    # fig2.clf()
    # ax2 = fig2.add_subplot(1, 1, 1)
    # plt.plot(positions[np.argsort(order),0],positions[np.argsort(order),1])
    # plt.show()
    #pdb.set_trace()
    # A matplotlib-based animator object
    animator = Animator(positions, radius * 2)

    # simulation/animation time variablees
    physics_step = 1.0 / 100  # estimate of real-time performance of simulation
    anim_step = 1.0 / 30  # FPS
    total_time = 0
    runtime=0
    frame_count = 0

    # SUBPROBLEM 4: uncomment the code below.
    # preallocate locks for objects
    locks_ptr = preallocate_locks(num_balls)
    x=True
    while x==True:
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_spacing, locks_ptr,
                   physics_step)

        # update our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval
        runtime += t.interval
        frame_count += 1
        if total_time > anim_step:
            
            #pdb.set_trace()
            print("{} simulation frames per second -- {} frames, {} seconds".format(frame_count / total_time,frame_count,total_time))
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!
            #7th order hilbert ordering, returned in array 'order'
            find_hilbert(hilbert_size,(hilbert_size*positions).astype(np.uint32),num_balls,order)
            order_idx=np.argsort(order)
            #update position and velocity order
            positions=positions[order_idx,:]
            velocities=velocities[order_idx,:]
            #update grid positions. Since we are only reordering, balls aren't moving to a new grid space, only swapping positions, so we don't have to do anything annoying like reinitialize the grid. 
            grid[(positions[:, 0] / grid_spacing).astype(int),
                 (positions[:, 1] / grid_spacing).astype(int)] = np.arange(num_balls)
            animator.update(positions)
            
        if runtime>10:
            x=False