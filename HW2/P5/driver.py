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

# Hilbert_curve
#convert (x,y) to d
def xy2d(n, x, y):
    s = int(n/2)
    d = 0;
    while (s > 0):
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(s, x, y, rx, ry)
        s /= 2
    return d

#convert d to (x,y)
def d2xy(n, d):
    s = 1
    t = d
    x = y = 0
    while(s<n):
        rx = 1 & (t/2)
        ry = 1 & (t ^ rx)
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t /= 4
        s *= 2
    return x, y


#rotate/flip a quadrant appropriately
def rot(n, x, y, rx, ry):
    if ry == 0 :
        if rx == 1:
            x = n-1 - x
            y = n-1 - y
        #Swap x and y
        temp = x
        x = y
        y = temp
    return x, y    
    
    
    
if __name__ == '__main__':
    num_balls = 10000
    radius = 0.002
    #num_balls = 50
    #radius = 0.05
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
    print("grid_spacing: {} grid_size:{}".format(grid_spacing, grid_size))

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
    stats = np.empty(500)
    ll = 0
    while True:
        with Timer() as t:
            update(positions, velocities, grid,
                   radius, grid_spacing, grid_size, locks_ptr,
                   physics_step)

        # udpate our estimate of how fast the simulator runs
        physics_step = 0.9 * physics_step + 0.1 * t.interval
        total_time += t.interval

        frame_count += 1
        if total_time > anim_step:
            animator.update(positions)
            print("{} simulation frames per second".format(frame_count / total_time))
            stats[ll] = frame_count / total_time
            ll += 1
            frame_count = 0
            total_time = 0
            # SUBPROBLEM 3: sort objects by location.  Be sure to update the
            # grid if objects' indices change!  Also be sure to sort the
            # velocities with their object positions!
            order = np.empty(positions.shape[0])
            for i,idx in enumerate((positions/grid_spacing).astype(int)):
                order[i] = xy2d(positions.shape[0], idx[0], idx[1])
            neworder = np.argsort(order)
            positions = positions[neworder]
            velocities = velocities[neworder]
            
        if ll >= len(stats):
            #ll = 0
            break
    print "avg:", stats.mean()
    print "std:", stats.std()
